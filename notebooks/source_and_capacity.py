import torch
import numpy as np
import os
from pathlib import Path
from olmo.config import TrainConfig
from olmo.model import OLMo
from olmo.data import build_train_dataloader
from tqdm import tqdm

# Load all the good stuff
rootdir = Path("/n/netscratch/kempner_sham_lab/Everyone/ameterez/continual_learning/48120231_27/latest-unsharded") # this is best cosine at 32x chinchilla
cfg = TrainConfig.load(rootdir / 'config.yaml')
weights = rootdir / 'model.pt'
olmo_model = OLMo(cfg.model)
# dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
dtype  = torch.float
device = 'cuda:0'
olmo_model.load_state_dict(torch.load(weights, weights_only=True))
olmo_model = olmo_model.to(device=device, dtype=dtype).eval()

cfg.global_train_batch_size = 1
cfg.data.num_workers = 1
train_loader = build_train_dataloader(cfg)

batch = None
for i, batch in enumerate(train_loader):
    if i == 4:
        break
del batch['index']
batch['input_ids'] = batch['input_ids'].cuda()

import math
import torch
from torch.func import functional_call, jvp

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import math
import torch
from torch.func import functional_call, jvp
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ------------------------- helpers -------------------------

def _harmonic_sum(D, s):
    """H_{D,s} = sum_{i=1}^D i^{-s}"""
    if D <= 200_000:
        i = torch.arange(1, D + 1, dtype=torch.float64)
        return float((i.pow(-s)).sum().item())
    # integral-ish approx (fine for solving)
    if abs(s - 1.0) < 1e-8:
        return 1.0 + math.log(D)
    return 1.0 + (D**(1.0 - s) - 1.0) / (1.0 - s)

def _solve_alpha_from_trace_ratio(rhat, D, iters=80):
    # solve H_{D,2a}/H_{D,a}^2 = rhat for a=alpha (monotone in a for a>0)
    def ratio(a):
        h1 = _harmonic_sum(D, a)
        h2 = _harmonic_sum(D, 2.0 * a)
        return h2 / (h1 * h1 + 1e-30)

    lo, hi = 0.0, 10.0
    while ratio(hi) < rhat:
        hi *= 2.0
        if hi > 1e6:
            break

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < rhat:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _solve_beta_from_ratio(r, alpha, D, beta_lo=1.0 + 1e-6, beta_hi=50.0, iters=80):
    # solve H_{D,alpha+beta}/H_{D,beta} = r, beta>1
    if r <= 0:
        return float("nan")
    if r >= 1.0:
        return 1e6  # ratio -> 1 as beta -> +inf

    def ratio(beta):
        return _harmonic_sum(D, alpha + beta) / (_harmonic_sum(D, beta) + 1e-30)

    lo, hi = beta_lo, beta_hi
    while ratio(hi) < r and hi < 1e6:
        hi *= 2.0

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _dot_tree(a, b, names):
    return sum((a[n] * b[n]).sum().item() for n in names)

def _norm2_tree(a, names):
    return sum((a[n] * a[n]).sum().item() for n in names)

# ------------------------- GGN Hv -------------------------

def ggn_hvp(model, params, buffers, names, plist, batch, v, microbatch_size=1):
    """
    u = H v where H is the GGN induced by CE in logit space.
    v/u are dicts matching params structure.
    """
    B = batch["input_ids"].shape[0]
    u = {n: torch.zeros_like(params[n]) for n in names}

    for s in tqdm(range(0, B, microbatch_size)):
        e = min(B, s + microbatch_size)

        bmb = {
            k: (vv[s:e] if (torch.is_tensor(vv) and vv.shape[:1] == (B,)) else vv)
            for k, vv in batch.items()
        }

        def f(pdict):
            out = functional_call(model, {**pdict, **buffers}, args=(), kwargs=bmb)
            return out.logits[:, :-1, :]  # [b, T-1, V]

        logits, Jv = jvp(f, (params,), (v,))

        # WJv = (Diag(p) - p p^T) Jv
        with torch.no_grad():
            p = torch.softmax(logits, dim=-1)
            WJv = p * Jv - (p * Jv).sum(dim=-1, keepdim=True) * p

        grads = torch.autograd.grad(
            outputs=logits,
            inputs=plist,
            grad_outputs=WJv,
            retain_graph=False,
            create_graph=False,
        )

        with torch.no_grad():
            for n, g in zip(names, grads):
                u[n].add_(g)

        del bmb, logits, Jv, p, WJv, grads

    return u

# ------------------------- main: alpha + beta (w = theta*) -------------------------

def estimate_alpha_beta_ggn(model, batch, D, M=4, microbatch_size=1):
    """
    - alpha from trace ratio tr(H^2)/tr(H)^2 (Hutchinson with M probes)
    - beta from w-moments with w = theta* (i.e., parameters of `model`)
      using 2 Hv products: m1w = w^T H w, m2w = w^T H^2 w
    Also returns a_hat (scale in lambda_i = a_hat i^{-alpha}) and b_hat (optional).
    """
    model.eval()

    # params / buffers
    params = {n: p.requires_grad_(True) for n, p in model.named_parameters()}
    names  = list(params.keys())
    plist  = [params[n] for n in names]
    pdtype = plist[0].dtype
    buffers = {n: (b.to(dtype=pdtype) if b.is_floating_point() else b) for n, b in model.named_buffers()}

    # ---- 1) estimate alpha via Hutchinson traces ----
    trH = 0.0
    trH2 = 0.0

    for _ in range(M):
        v = {
            n: (torch.randint(0, 2, params[n].shape, device=params[n].device, dtype=torch.int8) * 2 - 1
                ).to(params[n].dtype)
            for n in names
        }
        Hv = ggn_hvp(model, params, buffers, names, plist, batch, v, microbatch_size=microbatch_size)

        trH  += _dot_tree(v, Hv, names)      # v^T H v
        trH2 += _norm2_tree(Hv, names)       # ||H v||^2 = v^T H^2 v  (H psd)

        del v, Hv

    trH /= M
    trH2 /= M

    rhat = trH2 / (trH * trH + 1e-30)
    alpha = _solve_alpha_from_trace_ratio(rhat, D=D)

    # also estimate scale a in lambda_i = a i^{-alpha}
    H_a  = _harmonic_sum(D, alpha)
    H_2a = _harmonic_sum(D, 2.0 * alpha)
    a_hat_1 = trH / (H_a + 1e-30)
    a_hat_2 = math.sqrt(max(trH2, 0.0) / (H_2a + 1e-30))
    a_hat   = 0.5 * (a_hat_1 + a_hat_2)

    # ---- 2) estimate beta using w = theta* ----
    w = {n: params[n].detach() for n in names}   # w = theta*

    # u1 = H w, u2 = H u1
    u1 = ggn_hvp(model, params, buffers, names, plist, batch, w, microbatch_size=microbatch_size)
    m1w = _dot_tree(w, u1, names)                # w^T H w
    u2 = ggn_hvp(model, params, buffers, names, plist, batch, u1, microbatch_size=microbatch_size)
    m2w = _dot_tree(w, u2, names)                # w^T H^2 w

    # ratio r = m2w/(a*m1w) = H_{D,alpha+beta}/H_{D,beta}
    r = m2w / (a_hat * (m1w + 1e-30) + 1e-30)
    beta = _solve_beta_from_ratio(r, alpha=alpha, D=D)

    # optional: recover b from m1w = b * H_{D,beta}
    b_hat = m1w / (_harmonic_sum(D, beta) + 1e-30)

    info = {
        "trH": trH,
        "trH2": trH2,
        "rhat_trace": rhat,
        "alpha_check_ratio": _harmonic_sum(D, 2.0 * alpha) / (_harmonic_sum(D, alpha)**2 + 1e-30),
        "a_hat_from_trH": a_hat_1,
        "a_hat_from_trH2": a_hat_2,
        "a_hat": a_hat,
        "m1w": m1w,
        "m2w": m2w,
        "r_beta": r,
        "b_hat": b_hat,
    }
    return alpha, beta, info

alpha, beta, info = estimate_alpha_beta_ggn(olmo_model, batch, D=50_000, M=32, microbatch_size=8)
print(alpha, beta, info)

alpha, beta, info = estimate_alpha_beta_ggn(olmo_model, batch, D=300_000, M=32, microbatch_size=8)
print(alpha, beta, info)
