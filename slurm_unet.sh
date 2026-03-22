#!/bin/bash
#SBATCH --job-name=lasd-unet
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

# LASD on U-Net (Stable Diffusion v1.5)
# Cross-architecture comparison with DiT experiments

set -e

echo "=== LASD U-Net Experiments ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs vectors/sd15 results

python run_unet.py 2>&1 | tee logs/unet_run_${SLURM_JOB_ID}.log

echo ""
echo "=== Done: $(date) ==="
