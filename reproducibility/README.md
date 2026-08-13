# Figure reproduction

These scripts replace the exploratory notebooks with deterministic, reviewable
entry points. They use exact W&B histories rather than the sampled
`Run.history()` API. They first try the parquet artifacts named
`run-<run_id>-history:v0`, cached under `.cache/wandb`, and fall back to exact
`scan_history()` for historical runs whose artifact is unavailable.

Authenticate once with `wandb login` or set `WANDB_API_KEY`. No credential is
stored in this repository.

```bash
python -m pip install -r reproducibility/requirements.txt

# Main Figure 2 and its numerical table.
python reproducibility/figure2.py

# Figure 1 (150M) and the analogous 300M appendix plot.
python reproducibility/cosine_transfer.py

# Appendix best-at-each-budget plots (Figures 5 and 6).
python reproducibility/best_per_budget.py main_150M
python reproducibility/best_per_budget.py main_300M
python reproducibility/best_per_budget.py large_batch_150M
```

Pass `--history-source artifact` to require parquet or `--history-source api`
to require `scan_history()`. Both paths are unsampled. Every output directory
includes a CSV of the exact losses/run IDs used in the plot.

Figure 2 deliberately uses:

- one 32x constant trajectory evaluated with EMA-25;
- one 32x (150M) or 16x (300M) inverse-square-root trajectory evaluated with
  EMA-25;
- a separately tuned cosine endpoint at each budget, evaluated at the last
  iterate;
- one WSD branch per budget, evaluated at the last iterate.

The historical generic key `eval/c4_val/Perplexity` was overwritten by the
EMA-100 evaluator in EMA-enabled runs. The scripts therefore read the explicit
EMA-0 cross-entropy key for WSD. This changes WSD by approximately
0.0021–0.0025 nats and does not change the selected WSD run at any audited
budget.

The exact run IDs and expected configurations are in `run_manifest.json`.
Configuration checks are enabled by default in `figure2.py`.
`reference/figure2_losses.csv` records the independently verified output for
all 11 model/budget points.

## Synthetic figures

`synthetics/figure3_exact_recursion.py` reproduces Figure 3 without Monte
Carlo. It sets

```text
H = diag(1^{-a}, 2^{-a}, ..., d^{-a})
(w_i^*)^2 = i^{-(b-a)}
```

and updates all `d=500000` diagonal second moments using the exact Gaussian
recursion. The target is not normalized, so
`lambda_i (w_i^*)^2 = i^{-b}` exactly. It sweeps one fixed learning-rate
multiplier per scheduler and selects it by mean excess risk over the eight
reported checkpoints. Run the full experiment on a GPU with:

```console
sbatch synthetics/figure3_exact_recursion.sbatch
```

The script writes the PDF, a PNG preview, the plotted CSV, and JSON containing
the selected hyperparameters. `synthetics/sim_recursion.py` remains available
as a lower-level utility for individual synthetic configurations.
