"""Exact W&B history loading shared by the paper reproduction scripts."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import wandb


def load_history(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    run_id: str,
    keys: Iterable[str],
    cache_dir: Path,
    source: str = "auto",
) -> pd.DataFrame:
    """Load an unsampled history, preferring W&B's parquet history artifact."""
    requested = list(dict.fromkeys(["_step", *keys]))
    frame = None
    if source in {"auto", "artifact"}:
        try:
            run_dir = cache_dir / run_id
            parquet_files = sorted(run_dir.rglob("*.parquet")) if run_dir.exists() else []
            if not parquet_files:
                artifact = api.artifact(f"{entity}/{project}/run-{run_id}-history:v0")
                artifact.download(root=str(run_dir))
                parquet_files = sorted(run_dir.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in history artifact for {run_id}")
            frame = pd.concat(
                (pd.read_parquet(path, columns=requested) for path in parquet_files),
                ignore_index=True,
            )
        except Exception:
            if source == "artifact":
                raise
            warnings.warn(
                f"Parquet history artifact unavailable for {run_id}; falling back to exact scan_history().",
                stacklevel=2,
            )

    if frame is None and source in {"auto", "api"}:
        run = api.run(f"{entity}/{project}/{run_id}")
        frame = pd.DataFrame(run.scan_history(keys=requested, page_size=1000))
    elif frame is None:
        raise ValueError(f"Unknown history source: {source}")

    if "_step" not in frame:
        raise KeyError(f"Run {run_id} has no _step column")
    frame = (
        frame.dropna(subset=["_step"])
        .assign(_step=lambda data: data["_step"].astype(int))
        .drop_duplicates("_step", keep="last")
    )
    # W&B histories are normally already ordered. Avoid an unnecessary sort
    # for very long runs (and a pandas/numpy argsort edge case on some builds).
    if not frame["_step"].is_monotonic_increasing:
        frame = frame.sort_values("_step")
    return frame.reset_index(drop=True)


def loss_from_perplexity(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid perplexity: {value}")
    return math.log(value)


def exact_row(frame: pd.DataFrame, step: int, *, run_id: str) -> pd.Series:
    rows = frame.loc[frame["_step"] == step]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for run {run_id} at step {step}; found {len(rows)}")
    return rows.iloc[0]
