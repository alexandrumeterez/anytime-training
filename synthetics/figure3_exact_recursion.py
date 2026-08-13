#!/usr/bin/env python3
"""Reproduce Figure 3 by evaluating the exact Gaussian risk recursion in JAX.

There is no Monte Carlo and no spectral approximation.  The Hessian is the
finite diagonal matrix H = diag(i**(-a)) for i=1,...,d, and all d diagonal
second moments are updated at every step.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


LEARNING_RATES = np.array(
    [
        1e-4,
        2e-4,
        5e-4,
        7e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        3e-2,
        5e-2,
        7.5e-2,
        1e-1,
        2e-1,
        3e-1,
        5e-1,
        8e-1,
        1.0,
        2.0,
        3.0,
        5.0,
        10.0,
    ],
    dtype=np.float64,
)
CHECKPOINTS = np.array([1000, 2000, 3000, 5000, 8000, 10000, 20000, 50000])


def make_problem(d: int, a: float, b: float) -> tuple[jax.Array, jax.Array, float]:
    """Return diag(H), diag(E[(w_0-w*)(w_0-w*)^T]), and tr(H).

    We use lambda_i = i^-a, (w_i*)^2 = i^(-(b-a)), and w_0 = 0, so
    lambda_i (w_i*)^2 = i^-b exactly.
    """
    index = jnp.arange(1, d + 1, dtype=jnp.float64)
    eigenvalues = index ** (-a)
    initial_second_moment = index ** (-(b - a))
    trace = float(jax.device_get(jnp.sum(eigenvalues)))
    return eigenvalues, initial_second_moment, trace


def moment_step(
    state: jax.Array,
    eigenvalues: jax.Array,
    step_lr: jax.Array,
    sigma2: float,
) -> tuple[jax.Array, jax.Array]:
    """One exact second-moment update for x ~ N(0, H), diagonal H.

    If s_i = E[(w-w*)_i^2] and q = sum_i lambda_i s_i, then

      s_i^+ = (1 - 2 eta lambda_i + 2 eta^2 lambda_i^2) s_i
                + eta^2 lambda_i (q + sigma^2).
    """
    previous_twice_excess_risk = state @ eigenvalues
    scaled_eigenvalues = step_lr[:, None] * eigenvalues[None, :]
    next_state = (
        (1.0 - 2.0 * scaled_eigenvalues + 2.0 * scaled_eigenvalues**2) * state
        + step_lr[:, None] ** 2
        * (previous_twice_excess_risk + sigma2)[:, None]
        * eigenvalues[None, :]
    )
    return next_state, scaled_eigenvalues


@partial(jax.jit, static_argnames=("steps", "schedule"))
def run_averaged_sweep(
    eigenvalues: jax.Array,
    initial_second_moment: jax.Array,
    trace: float,
    lr_multipliers: jax.Array,
    sigma2: float,
    steps: int,
    schedule: str,
) -> jax.Array:
    """Run constant or 1/sqrt(t), returning excess risk at every step."""
    base_lr = lr_multipliers / trace
    count = len(lr_multipliers)
    initial = jnp.broadcast_to(initial_second_moment, (count, len(eigenvalues)))

    def step(carry, t):
        state, averaged_state, cross_moment = carry
        step_lr = base_lr if schedule == "constant" else base_lr / jnp.sqrt(t)
        state, scaled_eigenvalues = moment_step(state, eigenvalues, step_lr, sigma2)

        # bar w_t = ((t-1)/t) bar w_{t-1} + (1/t) w_t.  The cross moment
        # E[(bar w_{t-1}-w*)(w_t-w*)^T] contracts by I - eta_t H.
        rho = 1.0 / t
        old_average_to_new_iterate = (1.0 - scaled_eigenvalues) * cross_moment
        averaged_state = (
            (1.0 - rho) ** 2 * averaged_state
            + rho**2 * state
            + 2.0 * rho * (1.0 - rho) * old_average_to_new_iterate
        )
        cross_moment = (1.0 - rho) * old_average_to_new_iterate + rho * state
        excess_risk = 0.5 * (averaged_state @ eigenvalues)
        return (state, averaged_state, cross_moment), excess_risk

    _, risks = jax.lax.scan(
        step,
        (initial, initial, initial),
        jnp.arange(1, steps + 1, dtype=jnp.float64),
    )
    return risks.T


def reproduce(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    checkpoints = CHECKPOINTS[CHECKPOINTS <= args.steps]
    if len(checkpoints) == 0 or checkpoints[-1] != args.steps:
        checkpoints = np.unique(np.append(checkpoints, args.steps))

    results: dict = {"panels": []}
    rows: list[dict] = []
    for a in (1.1, 1.5, 1.9):
        for b in (a, 2.0 * a):
            print(f"a={a:g}, b={b:g}", flush=True)
            eigenvalues, initial_second_moment, trace = make_problem(args.dimension, a, b)

            constant_sweep = np.asarray(
                jax.device_get(
                    run_averaged_sweep(
                        eigenvalues,
                        initial_second_moment,
                        trace,
                        jnp.asarray(LEARNING_RATES),
                        args.sigma2,
                        args.steps,
                        "constant",
                    )
                )
            )
            sqrt_sweep = np.asarray(
                jax.device_get(
                    run_averaged_sweep(
                        eigenvalues,
                        initial_second_moment,
                        trace,
                        jnp.asarray(LEARNING_RATES),
                        args.sigma2,
                        args.steps,
                        "sqrt",
                    )
                )
            )
            checkpoint_indices = checkpoints - 1
            constant_scores = constant_sweep[:, checkpoint_indices].mean(axis=1)
            sqrt_scores = sqrt_sweep[:, checkpoint_indices].mean(axis=1)
            constant_scores[~np.isfinite(constant_scores)] = np.inf
            sqrt_scores[~np.isfinite(sqrt_scores)] = np.inf
            constant_best = int(np.argmin(constant_scores))
            sqrt_best = int(np.argmin(sqrt_scores))

            plot_steps = np.unique(
                np.concatenate(
                    (
                        np.rint(np.geomspace(max(10, checkpoints[0] // 10), args.steps, 100)).astype(int),
                        checkpoints,
                    )
                )
            )
            constant_curve = constant_sweep[constant_best, plot_steps - 1]
            sqrt_curve = sqrt_sweep[sqrt_best, plot_steps - 1]
            constant_lr = float(LEARNING_RATES[constant_best])
            sqrt_lr = float(LEARNING_RATES[sqrt_best])

            panel = {
                "a": a,
                "b": b,
                "constant_lr_multiplier": constant_lr,
                "sqrt_lr_multiplier": sqrt_lr,
                "plot_steps": plot_steps.tolist(),
                "constant": constant_curve.tolist(),
                "sqrt": sqrt_curve.tolist(),
                "checkpoints": checkpoints.tolist(),
            }
            results["panels"].append(panel)
            for method, steps_for_method, losses in (
                ("constant+average", plot_steps, constant_curve),
                ("1/sqrt(t)+average", plot_steps, sqrt_curve),
            ):
                for sample_count, loss in zip(steps_for_method, losses):
                    rows.append(
                        {
                            "a": a,
                            "b": b,
                            "method": method,
                            "samples": int(sample_count),
                            "excess_risk": float(loss),
                        }
                    )
            print(
                f"  constant eta={constant_lr:g}; sqrt eta={sqrt_lr:g}",
                flush=True,
            )

    results.update(
        {
            "dimension": args.dimension,
            "steps": args.steps,
            "sigma2": args.sigma2,
            "learning_rate_note": "actual peak step size is eta / tr(H)",
            "selection_objective": "arithmetic mean excess risk over checkpoints",
            "method": "full finite-dimensional diagonal Gaussian moment recursion",
        }
    )
    return results, rows


def plot_figure(results: dict, output: Path) -> None:
    panels = {(panel["a"], panel["b"]): panel for panel in results["panels"]}
    colors = {"constant": "#4472C4", "sqrt": "#E36B5D"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 6), sharex="row", sharey="row")
    for column, a in enumerate((1.1, 1.5, 1.9)):
        for row, b in enumerate((a, 2.0 * a)):
            panel = panels[(a, b)]
            ax = axes[row, column]
            plot_steps = np.asarray(panel["plot_steps"])
            checkpoints = np.asarray(panel["checkpoints"])
            constant = np.asarray(panel["constant"])
            sqrt = np.asarray(panel["sqrt"])

            ax.plot(
                plot_steps,
                constant,
                color=colors["constant"],
                linestyle=":",
                linewidth=2,
                label="Const. + Avg.",
            )
            ax.plot(
                plot_steps,
                sqrt,
                color=colors["sqrt"],
                linestyle="--",
                linewidth=2,
                label=r"$1/\sqrt{t}$ + Avg.",
            )
            checkpoint_indices = np.searchsorted(plot_steps, checkpoints)
            ax.scatter(
                checkpoints,
                constant[checkpoint_indices],
                color=colors["constant"],
                marker="s",
                s=22,
                zorder=3,
            )
            ax.scatter(
                checkpoints,
                sqrt[checkpoint_indices],
                color=colors["sqrt"],
                marker="D",
                s=22,
                zorder=3,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(rf"$a={a:g},\ b={b:g}$")
            if column == 0:
                ax.set_ylabel("Excess Risk")
            if row == 1:
                ax.set_xlabel("Samples")
            if row == 0:
                ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=500_000)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--sigma2", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "figure3"
        / "noalpha_synthetic_sigma2_0.01.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"JAX devices: {jax.devices()}", flush=True)
    results, rows = reproduce(args)
    plot_figure(results, args.output)
    with args.output.with_suffix(".json").open("w") as handle:
        json.dump(results, handle, indent=2)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
