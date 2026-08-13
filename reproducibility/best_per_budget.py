#!/usr/bin/env python3
"""Reproduce the appendix best-hyperparameter-at-each-budget comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb_history import exact_row, load_history, loss_from_perplexity

HERE = Path(__file__).resolve().parent
PPL = "eval/c4_val/Perplexity"
EMA_0 = "eval/c4_val/all-small-ppl-validation_ema_0.0/CrossEntropyLoss"
EMA_25 = "eval/c4_val/all-small-ppl-validation_ema_25.0/CrossEntropyLoss"
METHODS = {
    "cosine_with_warmup": ("cosine", PPL),
    "constant_with_warmup": ("constant_ema25", EMA_25),
    "inverse_sqrt_with_warmup": ("inverse_sqrt_ema25", EMA_25),
    "linear_decay": ("wsd_last_iterate", EMA_0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("main_150M", "main_300M", "large_batch_150M"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "best_per_budget")
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache" / "wandb")
    parser.add_argument("--history-source", choices=("auto", "artifact", "api"), default="auto")
    return parser.parse_args()


def experiment_spec(manifest: dict, name: str) -> dict:
    if name == "main_150M":
        base = manifest["figure_2"]["150M"]
        return {**base, "groups": manifest["sweep_groups"][name], "warmup": 5000}
    if name == "main_300M":
        base = manifest["figure_2"]["300M"]
        return {**base, "groups": manifest["sweep_groups"][name], "warmup": 5000}
    large = manifest["large_batch"]
    return {
        "batch_size": large["batch_size"],
        "durations": large["durations"],
        "groups": manifest["sweep_groups"][name],
        "warmup": large["warmup_steps_in_runs"],
    }


def collect(api: wandb.Api, entity: str, project: str, spec: dict, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    seen = set()
    for group in spec["groups"]:
        for run in api.runs(
            f"{entity}/{project}",
            filters={"group": group, "state": "finished"},
            per_page=500,
        ):
            if run.id in seen:
                continue
            seen.add(run.id)
            config = run.config
            scheduler = config.get("scheduler", {})
            scheduler_name = scheduler.get("name")
            if scheduler_name not in METHODS:
                continue
            if config.get("global_train_batch_size") != spec["batch_size"]:
                continue
            # The submitted main sweeps use 5k warmup; the actual large-batch
            # runs use 500 steps.
            if scheduler.get("t_warmup") != spec["warmup"]:
                continue
            method, metric = METHODS[scheduler_name]
            frame = load_history(
                api,
                entity=entity,
                project=project,
                run_id=run.id,
                keys=[metric],
                cache_dir=args.cache_dir,
                source=args.history_source,
            )
            for step in spec["durations"]:
                try:
                    value = exact_row(frame, step, run_id=run.id)[metric]
                except RuntimeError:
                    continue
                if pd.isna(value):
                    continue
                loss = loss_from_perplexity(float(value)) if metric == PPL else float(value)
                rows.append(
                    {
                        "method": method,
                        "step": step,
                        "loss": loss,
                        "run_id": run.id,
                        "group": group,
                        "learning_rate": config.get("optimizer", {}).get("learning_rate"),
                        "beta2": config.get("optimizer", {}).get("beta_1"),
                        "offset": scheduler.get("offset"),
                    }
                )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("No matching run/checkpoint candidates found")
    selected = (
        candidates.loc[candidates.groupby(["method", "step"])["loss"].idxmin()]
        .sort_values(["method", "step"])
        .reset_index(drop=True)
    )
    return selected


def make_plot(selected: pd.DataFrame, spec: dict, output: Path) -> None:
    labels = {
        "constant_ema25": "Const. + Avg.",
        "inverse_sqrt_ema25": r"$1/\sqrt{t}$ + Avg.",
        "wsd_last_iterate": "WSD",
    }
    colors = {"constant_ema25": "#5a81db", "inverse_sqrt_ema25": "#e67777", "wsd_last_iterate": "#e0a546"}
    markers = {"constant_ema25": "s", "inverse_sqrt_ema25": "D", "wsd_last_iterate": "v"}
    cosine = selected.loc[selected["method"] == "cosine"].set_index("step")["loss"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for method, label in labels.items():
        curve = selected.loc[selected["method"] == method].sort_values("step").copy()
        curve["tokens"] = curve["step"] * spec["batch_size"] * 1024
        curve["delta"] = curve["loss"] - curve["step"].map(cosine)
        axes[0].plot(curve["tokens"], curve["loss"], color=colors[method], marker=markers[method], label=label)
        axes[1].plot(curve["tokens"], curve["delta"], color=colors[method], marker=markers[method], label=label)
    cosine_frame = selected.loc[selected["method"] == "cosine"].sort_values("step").copy()
    cosine_frame["tokens"] = cosine_frame["step"] * spec["batch_size"] * 1024
    axes[0].plot(
        cosine_frame["tokens"],
        cosine_frame["loss"],
        color="#781202",
        marker="*",
        markersize=12,
        label="Cosine env.",
    )
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[0].set_ylabel("Validation loss")
    axes[1].set_ylabel(r"$\mathcal{L}_{sched}-\mathcal{L}_{cosine}\ \downarrow$")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Tokens")
        axis.grid(True, axis="y", alpha=0.4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "run_manifest.json").read_text())
    spec = experiment_spec(manifest, args.experiment)
    api = wandb.Api(timeout=120)
    selected = collect(api, manifest["wandb"]["entity"], manifest["wandb"]["project"], spec, args)
    selected.to_csv(args.output_dir / f"{args.experiment}.csv", index=False)
    make_plot(selected, spec, args.output_dir / args.experiment)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
