#!/usr/bin/env python3
"""Reproduce the cosine-envelope transfer plots from exact run histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb_history import load_history, loss_from_perplexity

HERE = Path(__file__).resolve().parent
PPL = "eval/c4_val/Perplexity"
SEQUENCE_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "cosine_transfer")
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache" / "wandb")
    parser.add_argument("--history-source", choices=("auto", "artifact", "api"), default="auto")
    return parser.parse_args()


def nearest_checkpoint(frame: pd.DataFrame, target: int) -> pd.Series:
    valid = frame.dropna(subset=[PPL]).copy()
    valid["distance"] = (valid["_step"] - target).abs()
    return valid.sort_values(["distance", "_step"]).iloc[0]


def build(
    api: wandb.Api,
    entity: str,
    project: str,
    model: str,
    spec: dict,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_scale = spec["batch_size"] * SEQUENCE_LENGTH
    envelope_rows, transfer_rows = [], []
    for tuned_duration, run_id in zip(spec["durations"], spec["cosine_envelope"]):
        frame = load_history(
            api,
            entity=entity,
            project=project,
            run_id=run_id,
            keys=[PPL],
            cache_dir=args.cache_dir,
            source=args.history_source,
        )
        endpoint = nearest_checkpoint(frame, tuned_duration)
        if int(endpoint["_step"]) != tuned_duration:
            raise RuntimeError(f"{run_id} has no exact terminal evaluation at {tuned_duration}")
        envelope_rows.append(
            {
                "model": model,
                "tuned_duration": tuned_duration,
                "run_id": run_id,
                "tokens": tuned_duration * token_scale,
                "loss": loss_from_perplexity(float(endpoint[PPL])),
            }
        )
        for target in (step for step in spec["durations"] if step <= tuned_duration):
            row = nearest_checkpoint(frame, target)
            transfer_rows.append(
                {
                    "model": model,
                    "run_id": run_id,
                    "tuned_duration": tuned_duration,
                    "target_step": target,
                    "logged_step": int(row["_step"]),
                    "tokens": target * token_scale,
                    "loss": loss_from_perplexity(float(row[PPL])),
                }
            )
    return pd.DataFrame(envelope_rows), pd.DataFrame(transfer_rows)


def make_plot(model: str, envelope: pd.DataFrame, transfer: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
        }
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    for tokens in envelope["tokens"]:
        axis.axvline(tokens, color="0.5", linestyle="--", linewidth=1, alpha=0.6)

    # The shortest-horizon run contributes only its terminal point, which is
    # already shown by the cosine envelope.  Plot the genuinely transferred
    # trajectories (2x and longer), matching the submitted figure.
    durations = sorted(envelope["tuned_duration"], reverse=True)[:-1]

    # Plot the envelope first so the legend order matches the paper, while
    # zorder keeps it visible above the gray transfer trajectories.
    axis.plot(
        envelope["tokens"],
        envelope["loss"],
        color="#781202",
        linewidth=3.5,
        label="Cosine env.",
        zorder=4,
    )
    axis.scatter(
        envelope["tokens"],
        envelope["loss"],
        color="#781202",
        marker="*",
        s=150,
        zorder=5,
    )

    shade_positions = [0.9, 0.75, 0.6, 0.45, 0.3][: len(durations)]
    shades = [
        tuple(value)
        for value in mpl.colormaps["Greys"](shade_positions)
    ]
    all_durations = envelope["tuned_duration"].sort_values().tolist()
    for shade, duration in zip(shades, durations, strict=True):
        curve = transfer.loc[transfer["tuned_duration"] == duration].sort_values("target_step")
        multiple = 2 ** all_durations.index(duration)
        axis.plot(
            curve["tokens"],
            curve["loss"],
            color=shade,
            linewidth=3,
            label=rf"${multiple}\times$",
            zorder=2,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Tokens")
    axis.set_ylabel("Validation Loss")
    axis.text(
        0.02,
        0.06,
        model,
        transform=axis.transAxes,
        fontsize=18,
        fontweight="bold",
    )
    axis.grid(True, axis="y", alpha=0.4)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        ncol=len(durations) + 1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=15,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "run_manifest.json").read_text())
    entity, project = manifest["wandb"]["entity"], manifest["wandb"]["project"]
    api = wandb.Api(timeout=120)
    for model, spec in manifest["figure_2"].items():
        envelope, transfer = build(api, entity, project, model, spec, args)
        envelope.to_csv(args.output_dir / f"{model}_envelope.csv", index=False)
        transfer.to_csv(args.output_dir / f"{model}_transfer.csv", index=False)
        make_plot(model, envelope, transfer, args.output_dir / f"{model}_cosine_transfer")
        print(transfer[["model", "run_id", "tuned_duration", "target_step", "logged_step"]].to_string(index=False))


if __name__ == "__main__":
    main()
