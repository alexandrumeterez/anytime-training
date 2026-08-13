from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp
from tqdm import tqdm

from olmo.config import TrainConfig
from olmo.model import OLMo
from olmo.data import build_train_dataloader
from olmo.optim import build_optimizer


TensorTree = Dict[str, torch.Tensor]

def extract_params_buffers(model: torch.nn.Module) -> Tuple[TensorTree, TensorTree]:
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    return params, buffers


def zeros_like_tree(tree: TensorTree, *, device=None, dtype=None) -> TensorTree:
    return {
        k: torch.zeros_like(v, device=device or v.device, dtype=dtype or v.dtype)
        for k, v in tree.items()
    }


def tree_scale(tree: TensorTree, s: float) -> TensorTree:
    return {k: v * s for k, v in tree.items()}

@torch.no_grad()
def build_frozen_denom_from_adamw(
    optimizer,
    model_params_in_order: Iterable[torch.nn.Parameter],
    device: torch.device,
    out_dtype=torch.float32,
    eps: float = 1e-8,
    bias_correct: bool = True,
    beta2: float = 0.9,
):
    if not optimizer.param_groups:
        raise RuntimeError("Optimizer has no param_groups.")

    denoms = []
    for p in model_params_in_order:
        st = optimizer.state.get(p, None)
        if st is None or "exp_avg_sq" not in st:
            raise RuntimeError("Missing exp_avg_sq in optimizer state.")

        v = st["exp_avg_sq"].detach().to(device=device, dtype=torch.float32)

        if bias_correct and "step" in st:
            step = st["step"]
            step = int(step.item()) if torch.is_tensor(step) else int(step)
            bc2 = 1.0 - (beta2 ** step)
            if bc2 > 0:
                v = v / bc2

        denoms.append((v.sqrt() + eps).to(dtype=out_dtype))
    return denoms


@torch.no_grad()
def apply_frozen_denom_to_grad_inplace(grad_delta: TensorTree, denom_by_name: TensorTree) -> None:
    for name, g in grad_delta.items():
        d = denom_by_name[name]
        if g.shape != d.shape:
            raise RuntimeError(
                f"Shape mismatch for '{name}': grad {tuple(g.shape)} vs denom {tuple(d.shape)}"
            )
        g.div_(d.to(device=g.device, dtype=g.dtype))


class FrozenDenomPreconditioner:
    """Holds a {param_name: denom} map and applies it to grad trees."""

    def __init__(self, denom_by_name: TensorTree):
        self.denom_by_name = denom_by_name

    @classmethod
    @torch.no_grad()
    def from_adamw_state(
        cls,
        *,
        optimizer,
        model: torch.nn.Module,
        device: torch.device,
        eps: float,
        bias_correct: bool = True,
        beta2: float = 0.9,
        denom_dtype: torch.dtype = torch.float32,
    ) -> "FrozenDenomPreconditioner":
        named_params = list(model.named_parameters())
        names = [n for n, _ in named_params]
        params_in_order = [p for _, p in named_params]

        denom_list = build_frozen_denom_from_adamw(
            optimizer,
            model_params_in_order=params_in_order,
            device=device,
            out_dtype=denom_dtype,
            eps=eps,
            bias_correct=bias_correct,
            beta2=beta2,
        )
        denom_by_name = {n: d for n, d in zip(names, denom_list)}
        return cls(denom_by_name)

    @torch.no_grad()
    def apply_(self, grad_delta: TensorTree) -> None:
        apply_frozen_denom_to_grad_inplace(grad_delta, self.denom_by_name)


@torch.no_grad()
def sgd_momentum_step_inplace(
    delta: TensorTree,
    grad_delta: TensorTree,
    velocity: TensorTree,
    *,
    lr: float,
    momentum: float = 0.9,
) -> None:
    for k in delta.keys():
        velocity[k].mul_(momentum).add_(grad_delta[k])
        delta[k].add_(velocity[k], alpha=-lr)


@dataclass(frozen=True)
class CosineAnnealingSchedule:
    base_lr: float
    min_lr: float
    t_max: int  # number of steps to decay over

    def __call__(self, step: int) -> float:
        if self.t_max <= 0:
            return float(self.base_lr)
        s = min(max(step, 0), self.t_max)
        return float(
            self.min_lr
            + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * s / self.t_max))
        )

def linearized_loss_and_grad_microbatched(model, params0, buffers0, delta, batch, microbatch_size: int = 1):
    input_ids = batch["input_ids"]
    B = input_ids.shape[0]
    total_loss_sum = 0.0
    total_tokens = 0
    grad_accum = zeros_like_tree(delta)

    for s in range(0, B, microbatch_size):
        e = min(B, s + microbatch_size)
        ids = input_ids[s:e]
        x = ids[:, :-1]
        y = ids[:, 1:]
        tokens = y.numel()

        def f(p):
            out = functional_call(model, (p, buffers0), args=(), kwargs={"input_ids": x})
            return out.logits if hasattr(out, "logits") else out  # [b, T-1, V]

        logits0, v = jvp(f, (params0,), (delta,))
        logits_lin = (logits0 + v).detach().requires_grad_(True)

        loss_mb = F.cross_entropy(
            logits_lin.reshape(-1, logits_lin.size(-1)),
            y.reshape(-1),
            reduction="mean",
        )
        (g_logits,) = torch.autograd.grad(loss_mb, (logits_lin,))
        _, vjp_fn = vjp(f, params0)
        (grad_delta_mb,) = vjp_fn(g_logits.detach())
        total_loss_sum += loss_mb.detach() * tokens
        total_tokens += tokens
        for k in grad_accum.keys():
            grad_accum[k] += grad_delta_mb[k] * tokens

        del logits0, logits_lin, g_logits, v, grad_delta_mb
    loss = total_loss_sum / total_tokens
    grad_delta = tree_scale(grad_accum, 1.0 / total_tokens)
    return loss, grad_delta

def load_cfg(rootdir: Path, *, num_workers: int, max_duration: int) -> TrainConfig:
    cfg = TrainConfig.load(rootdir / "config.yaml")
    cfg.data.num_workers = num_workers
    cfg.max_duration = max_duration
    return cfg

def load_model(cfg: TrainConfig, weights_path: Path, *, device: str, dtype: torch.dtype) -> OLMo:
    model = OLMo(cfg.model)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device=device, dtype=dtype).eval()
    return model

def load_optimizer(cfg: TrainConfig, model: torch.nn.Module, optim_path: Path):
    optimizer = build_optimizer(cfg, model)
    optimizer.load_state_dict(torch.load(optim_path, map_location="cpu", weights_only=True))
    return optimizer

@dataclass
class LinearizedTrainingState:
    params0: TensorTree
    buffers0: TensorTree
    delta: TensorTree
    velocity: TensorTree

class LinearizedTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        state: LinearizedTrainingState,
        precond: FrozenDenomPreconditioner,
        lr_schedule,
        momentum: float,
        microbatch_size: int,
        device: str,
    ):
        self.model = model
        self.state = state
        self.precond = precond
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.microbatch_size = microbatch_size
        self.device = device
        self._step_idx = 0

    def step(self, batch: dict) -> Tuple[float, float]:
        batch = {"input_ids": batch["input_ids"].to(device=self.device, non_blocking=True)}

        loss, grad_delta = linearized_loss_and_grad_microbatched(
            self.model,
            self.state.params0,
            self.state.buffers0,
            self.state.delta,
            batch,
            microbatch_size=self.microbatch_size,
        )

        self.precond.apply_(grad_delta)

        lr_t = float(self.lr_schedule(self._step_idx))
        sgd_momentum_step_inplace(
            self.state.delta,
            grad_delta,
            self.state.velocity,
            lr=lr_t,
            momentum=self.momentum,
        )

        self._step_idx += 1
        return float(loss), lr_t
def main():
    rootdir = Path("/n/netscratch/sham_lab/Lab/pranavajitnair/continual_learning/45977048_5/step12499-unsharded")
    device = "cuda:0"
    dtype = torch.float32

    base_lr = 3e-5
    min_lr = base_lr / 10
    num_workers = 6
    microbatch_size = 14
    max_duration = 1000

    weights_path = rootdir / "model.pt"
    optim_path = rootdir / "optim.pt"

    cfg = load_cfg(rootdir, num_workers=num_workers, max_duration=max_duration)
    model = load_model(cfg, weights_path, device=device, dtype=dtype)
    train_loader = build_train_dataloader(cfg)

    params0, buffers0 = extract_params_buffers(model)
    delta = zeros_like_tree(params0, device=device, dtype=dtype)
    velocity = zeros_like_tree(delta)

    optimizer = load_optimizer(cfg, model, optim_path)
    precond = FrozenDenomPreconditioner.from_adamw_state(
        optimizer=optimizer,
        model=model,
        device=torch.device(device),
        eps=cfg.optimizer.eps,
        bias_correct=True,
        beta2=cfg.optimizer.beta_1,
        denom_dtype=torch.float32,
    )

    t_max = int(cfg.max_duration)
    lr_schedule = CosineAnnealingSchedule(base_lr=base_lr, min_lr=min_lr, t_max=t_max)

    trainer = LinearizedTrainer(
        model=model,
        state=LinearizedTrainingState(params0=params0, buffers0=buffers0, delta=delta, velocity=velocity),
        precond=precond,
        lr_schedule=lr_schedule,
        momentum=cfg.optimizer.beta_0,
        microbatch_size=microbatch_size,
        device=device,
    )

    for step, batch in enumerate(train_loader):
        loss_val, lr_t = trainer.step(batch)
        print(f"step={step} lr={lr_t:.6e} loss={loss_val:.4f}")

if __name__ == "__main__":
    main()
