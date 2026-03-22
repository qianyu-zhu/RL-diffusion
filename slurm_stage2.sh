#!/bin/bash
#SBATCH --job-name=lasd-s2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

# LASD Stage 2: Autonomous experiment runner
# Partition: mit_normal_gpu (max 6h, max 2 GPUs, 32 cores)
# Using 1 GPU — DiT-XL/2 fits in ~5GB fp16

set -e

echo "=== LASD Stage 2 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

# Setup environment
module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs vectors results

# Run the autonomous experiment script
python run_stage2.py 2>&1 | tee logs/stage2_run_${SLURM_JOB_ID}.log

echo ""
echo "=== Done: $(date) ==="
