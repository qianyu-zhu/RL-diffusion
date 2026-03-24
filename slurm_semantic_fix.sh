#!/bin/bash
#SBATCH --job-name=lasd-semfix
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/semantic_fix_%j.out
#SBATCH --error=logs/semantic_fix_%j.err

set -e

echo "=== LASD Semantic Fix ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs vectors results/semantic_fix

python run_semantic_fix.py 2>&1 | tee logs/semantic_fix_run_${SLURM_JOB_ID}.log

echo ""
echo "=== Done: $(date) ==="
