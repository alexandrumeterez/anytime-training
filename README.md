# Anytime Pretraining

Code and figure-reproduction utilities for **“Anytime Pretraining:
Horizon-Free Learning-Rate Schedules with Weight Averaging.”**

The training stack is based on
[AI2 OLMo](https://github.com/allenai/OLMo). Paper-specific model definitions,
sweeps, scheduler changes, EMA evaluation, and synthetic experiments live in:

- `configs/kempner/` — 150M, 300M, large-batch, cosine, constant,
  inverse-square-root, and WSD configurations;
- `olmo/optim.py` — the paper schedulers;
- `olmo/train.py`, `olmo/eval/evaluator.py`, and `scripts/train.py` — online
  EMA maintenance and labeled evaluation metrics;
- `reproducibility/` — exact W&B run manifests and non-notebook plotting
  scripts;
- `synthetics/` — exact linear-regression moment recursion.

## Install

Create a Python environment with the CUDA/PyTorch version appropriate for the
training machine, then install the repository:

```bash
python -m pip install -e ".[all]"
python -m pip install -r reproducibility/requirements.txt
```

For plots backed by W&B, authenticate with `wandb login` or set
`WANDB_API_KEY`. Credentials are never read from the legacy notebooks.

## Reproduce paper figures

The central empirical figure can be regenerated independently of the original
notebooks:

```bash
python reproducibility/figure2.py
```

This downloads the exact run histories as parquet artifacts, validates the run
configuration, writes a loss table, and produces PDF and PNG versions of the
plot. See [reproducibility/README.md](reproducibility/README.md) for the cosine
transfer, best-per-budget, large-batch, and synthetic commands.

The fixed Figure 2 comparison uses one constant trajectory and one shifted
inverse-square-root trajectory per model size, both evaluated with EMA-25.
Cosine is the independently tuned last-iterate envelope, and WSD is evaluated
at its last iterate. All main runs use a 5,000-step warmup. The exact run IDs
and hyperparameters are committed in
`reproducibility/run_manifest.json`.

## Train

The paper configurations use the existing sweep launcher:

```bash
python scripts/kempner/run_sweep.py \
  configs/kempner/sweeps/32_chinchilla/cosine.yaml
```

For a single resolved configuration, use the normal OLMo entry point:

```bash
torchrun --nproc_per_node=8 scripts/train.py path/to/config.yaml
```

The checked-in configs contain cluster-specific data and checkpoint paths.
Change those paths for a new environment before launching.

## Scheduler definitions

- **Constant + warmup:** linear warmup followed by a constant learning rate.
- **Inverse square root + warmup:** after warmup,
  \(\eta_t=\eta\sqrt{\alpha/(t-t_{\mathrm{warmup}}+\alpha)}\).
- **Cosine + warmup:** independently horizon-tuned, with final LR equal to
  10% of the peak LR in the paper runs.
- **WSD branch:** starts from a constant-run checkpoint at 90% of the target
  duration and linearly decays to 10% of the peak LR.

For an EMA labeled `25`, the old-weight coefficient is
\(2^{-25/t}\), so its instantaneous half-life is approximately \(t/25\),
or 4% of the current training age. A label of `100` corresponds to a 1%
half-life.

## Tests

```bash
pytest tests/optim_test.py tests/eval/evaluator_test.py
```

The original OLMo licensing terms are retained in [LICENSE](LICENSE).
