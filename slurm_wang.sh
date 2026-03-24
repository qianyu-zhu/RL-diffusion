#!/bin/bash
#SBATCH --job-name=lasd-wang
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wang_%j.out
#SBATCH --error=logs/wang_%j.err

set -e
echo "=== Wang et al. Comparison ==="
echo "Start: $(date)"

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
python run_wang_comparison.py 2>&1 | tee logs/wang_run_${SLURM_JOB_ID}.log

echo "=== Done: $(date) ==="
