#!/bin/bash
#SBATCH --job-name=figure3-exact
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Lab/ameterez/anytime_synth/figure3-exact-%j.out

set -euo pipefail

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

workdir=/n/holylfs06/LABS/sham_lab/Lab/ameterez/anytime_synth
cd "$workdir"

~/pax/bin/python figure3_exact_recursion.py \
  --dimension 500000 \
  --steps 50000 \
  --sigma2 0.01 \
  --output "$workdir/noalpha_synthetic_sigma2_0.01.pdf"
