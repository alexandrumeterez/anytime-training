#!/bin/bash
#SBATCH --job-name=sgd_sweep
#SBATCH --account=kempner_grads
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Lab/ameterez/continual-learning/synthetics/logs/%x_%A_%a.out
#SBATCH --gpus-per-node=1    
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=05:00:00
#SBATCH --partition=kempner_h100,kempner
#SBATCH --array=0-15
#SBATCH --exclude=holygpu8a19102

source ~/.bashrc
conda deactivate
conda activate lr_decay

python -u sim_recursion.py \
  --device cuda \
  --dtype float32 \
  --output_dir /n/netscratch/kempner_sham_lab/Everyone/ameterez/synthetics_recursion \
  --seed 0 \
  --num_jobs 16 \
  --job_idx ${SLURM_ARRAY_TASK_ID} \
  --log_every 10