#!/bin/bash
#SBATCH --job-name=lasd-clv2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/clip_v2_%j.out
#SBATCH --error=logs/clip_v2_%j.err

set -e
echo "=== LASD CLIP v2 ==="
echo "Start: $(date)"

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
python run_clip_v2.py 2>&1 | tee logs/clip_v2_run_${SLURM_JOB_ID}.log

echo "=== Done: $(date) ==="
