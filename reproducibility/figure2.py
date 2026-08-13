#!/usr/bin/env python3
"""Reproduce the main anytime-schedule figure from exact W&B histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb_history import exact_row, load_history, loss_from_perplexity

HERE = Path(__file__).resolve().parent
EMA_0 = "eval/c4_val/all-small-ppl-validation_ema_0.0/CrossEntropyLoss"
EMA_25 = "eval/c4_val/all-small-ppl-validation_ema_25.0/CrossEntropyLoss"
PPL = "eval/c4_val/Perplexity"
TOKENS_PER_EXAMPLE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "figure2")
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache" / "wandb")
    parser.add_argument("--history-source", choices=("auto", "artifact", "api"), default="auto")
    parser.add_argument("--verify-configs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_config(api: wandb.Api, path: str) -> dict:
    return api.run(path).config


def assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def verify_configs(api: wandb.Api, entity: str, project: str, spec: dict) -> None:
    expected = spec["expected"]
    all_ids = [
        *spec["cosine_envelope"],
        spec["constant"],
        spec["inverse_sqrt"],
        *spec["wsd"],
    ]
    configs = {run_id: run_config(api, f"{entity}/{project}/{run_id}") for run_id in all_ids}
    for run_id, duration in zip(spec["cosine_envelope"], spec["durations"]):
        config = configs[run_id]
        assert_equal(config["max_duration"], duration, f"{run_id} max_duration")
        assert_equal(config["global_train_batch_size"], spec["batch_size"], f"{run_id} batch size")
        assert_equal(config["scheduler"]["name"], "cosine_with_warmup", f"{run_id} scheduler")
        assert_equal(config["scheduler"]["t_warmup"], expected["warmup_steps"], f"{run_id} warmup")

    method_specs = (
        ("constant", "constant_with_warmup"),
        ("inverse_sqrt", "inverse_sqrt_with_warmup"),
    )
    for key, scheduler_name in method_specs:
        run_id = spec[key]
        config = configs[run_id]
        method_expected = expected[key]
        assert_equal(config["scheduler"]["name"], scheduler_name, f"{run_id} scheduler")
        assert_equal(config["scheduler"]["t_warmup"], expected["warmup_steps"], f"{run_id} warmup")
        assert_equal(config["optimizer"]["learning_rate"], method_expected["learning_rate"], f"{run_id} LR")
        assert_equal(config["optimizer"]["beta_1"], method_expected["beta2"], f"{run_id} beta2")
        if key == "inverse_sqrt":
            assert_equal(config["scheduler"]["offset"], method_expected["offset"], f"{run_id} offset")

    for run_id, duration in zip(spec["wsd"], spec["durations"]):
        config = configs[run_id]
        assert_equal(config["max_duration"], duration, f"{run_id} max_duration")
        assert_equal(config["scheduler"]["name"], "linear_decay", f"{run_id} scheduler")
        assert_equal(config["scheduler"]["alpha_f"], expected["wsd"]["final_lr_fraction"], f"{run_id} LR floor")
        assert_equal(config["optimizer"]["learning_rate"], expected["wsd"]["learning_rate"], f"{run_id} LR")
        assert_equal(config["optimizer"]["beta_1"], expected["wsd"]["beta2"], f"{run_id} beta2")


def history(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    keys: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    return load_history(
        api,
        entity=entity,
        project=project,
        run_id=run_id,
        keys=keys,
        cache_dir=args.cache_dir,
        source=args.history_source,
    )


def loss_trace(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    trace = frame[["_step", metric]].dropna().rename(columns={"_step": "step", metric: "loss"})
    if metric == PPL:
        trace["loss"] = trace["loss"].map(loss_from_perplexity)
    return trace


def build_model_data(
    api: wandb.Api,
    entity: str,
    project: str,
    model: str,
    spec: dict,
    args: argparse.Namespace,
) -> dict:
    if args.verify_configs:
        verify_configs(api, entity, project, spec)

    durations = spec["durations"]
    cosine_rows = []
    for run_id, duration in zip(spec["cosine_envelope"], durations):
        frame = history(api, entity, project, run_id, [PPL], args)
        row = exact_row(frame, duration, run_id=run_id)
        cosine_rows.append({"step": duration, "run_id": run_id, "loss": loss_from_perplexity(row[PPL])})
    cosine = pd.DataFrame(cosine_rows)

    constant = loss_trace(history(api, entity, project, spec["constant"], [EMA_25], args), EMA_25)
    sqrt = loss_trace(history(api, entity, project, spec["inverse_sqrt"], [EMA_25], args), EMA_25)
    long_cosine = loss_trace(history(api, entity, project, spec["long_cosine"], [PPL], args), PPL)

    wsd_rows = []
    for run_id, duration in zip(spec["wsd"], durations):
        frame = history(api, entity, project, run_id, [EMA_0, PPL], args)
        row = exact_row(frame, duration, run_id=run_id)
        corrected = float(row[EMA_0])
        overwritten_alias = loss_from_perplexity(float(row[PPL]))
        wsd_rows.append(
            {
                "step": duration,
                "run_id": run_id,
                "loss": corrected,
                "old_generic_ppl_loss": overwritten_alias,
                "correction": corrected - overwritten_alias,
            }
        )
    wsd = pd.DataFrame(wsd_rows)

    point = lambda frame: pd.DataFrame(
        [
            {
                "step": step,
                "loss": float(exact_row(frame.rename(columns={"step": "_step"}), step, run_id=model)["loss"]),
            }
            for step in durations
        ]
    )
    constant_points = point(constant)
    sqrt_points = point(sqrt)
    token_scale = spec["batch_size"] * TOKENS_PER_EXAMPLE
    for frame in (cosine, constant, sqrt, long_cosine, wsd, constant_points, sqrt_points):
        frame["tokens"] = frame["step"] * token_scale

    table = pd.DataFrame(
        {
            "model": model,
            "horizon": [f"{2**index}x" for index in range(len(durations))],
            "step": durations,
            "tokens": cosine["tokens"],
            "cosine": cosine["loss"],
            "constant_ema25": constant_points["loss"],
            "inverse_sqrt_ema25": sqrt_points["loss"],
            "wsd_last_iterate": wsd["loss"],
            "wsd_old_overwritten_alias": wsd["old_generic_ppl_loss"],
            "wsd_metric_correction": wsd["correction"],
        }
    )
    for column in ("constant_ema25", "inverse_sqrt_ema25", "wsd_last_iterate"):
        table[f"delta_{column}"] = table[column] - table["cosine"]
    return {
        "cosine": cosine,
        "constant": constant,
        "sqrt": sqrt,
        "long_cosine": long_cosine,
        "wsd": wsd,
        "constant_points": constant_points,
        "sqrt_points": sqrt_points,
        "table": table,
    }


def plot(all_data: list[tuple[str, dict]], output: Path) -> None:
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
        }
    )
    colors = {"cosine": "#781202", "constant": "#5a81db", "sqrt": "#e67777", "wsd": "#e0a546"}
    fig, axes = plt.subplots(len(all_data), 2, figsize=(14.5, 4.25 * len(all_data)), squeeze=False)

    for row_index, (model, data) in enumerate(all_data):
        left, right = axes[row_index]
        cosine = data["cosine"]
        for tokens in cosine["tokens"]:
            left.axvline(tokens, color="0.55", linestyle="--", linewidth=0.8, alpha=0.6)
            right.axvline(tokens, color="0.55", linestyle="--", linewidth=0.8, alpha=0.6)

        left.scatter(
            cosine["tokens"],
            cosine["loss"],
            color=colors["cosine"],
            marker="*",
            s=150,
            zorder=5,
            label="Cosine env.",
        )
        left.plot(
            data["long_cosine"]["tokens"],
            data["long_cosine"]["loss"],
            color=colors["cosine"],
            linewidth=2.6,
            label=f"Cosine {data['table']['horizon'].iloc[-1]}",
        )
        left.plot(
            data["constant"]["tokens"],
            data["constant"]["loss"],
            color=colors["constant"],
            linewidth=2.6,
            linestyle=":",
            label="Const. + Avg.",
        )
        left.scatter(
            data["wsd"]["tokens"],
            data["wsd"]["loss"],
            color=colors["wsd"],
            marker="v",
            s=90,
            zorder=5,
            label="WSD",
        )
        left.plot(
            data["sqrt"]["tokens"],
            data["sqrt"]["loss"],
            color=colors["sqrt"],
            linewidth=2.6,
            linestyle="--",
            label=r"$1/\sqrt{t}$ + Avg.",
        )
        left.set_xscale("log")
        left.set_xlabel("Tokens")
        left.set_ylabel("Validation loss")
        left.text(0.02, 0.08, model, transform=left.transAxes, fontsize=16, fontweight="bold")

        table = data["table"]
        right.scatter(
            table["tokens"],
            table["delta_constant_ema25"],
            color=colors["constant"],
            marker="s",
            s=70,
            label="Const. + Avg.",
        )
        right.scatter(
            table["tokens"], table["delta_wsd_last_iterate"], color=colors["wsd"], marker="v", s=80, label="WSD"
        )
        right.scatter(
            table["tokens"],
            table["delta_inverse_sqrt_ema25"],
            color=colors["sqrt"],
            marker="D",
            s=65,
            label=r"$1/\sqrt{t}$ + Avg.",
        )
        right.axhline(0, color="0.35", linewidth=1.0)
        right.set_xscale("log")
        right.set_xlabel("Tokens")
        right.set_ylabel(r"$\mathcal{L}_{sched}-\mathcal{L}_{cosine}\ \downarrow$")

        for axis in (left, right):
            axis.grid(True, axis="y", alpha=0.4)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9, frameon=False)

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "run_manifest.json").read_text())
    entity, project = manifest["wandb"]["entity"], manifest["wandb"]["project"]
    api = wandb.Api(timeout=120)
    all_data = [
        (model, build_model_data(api, entity, project, model, spec, args))
        for model, spec in manifest["figure_2"].items()
    ]
    table = pd.concat([data["table"] for _, data in all_data], ignore_index=True)
    table.to_csv(args.output_dir / "losses.csv", index=False)
    plot(all_data, args.output_dir / "figure2")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
