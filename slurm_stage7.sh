#!/bin/bash
#SBATCH --job-name=stage7
#SBATCH --partition=mit_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage7_%j.out
#SBATCH --error=logs/stage7_%j.err

source ~/.bashrc
conda activate cali-conf

cd /orcd/home/002/qianyu_z/RL-diffusion

echo "Starting Stage 7 experiments at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python run_stage7.py 2>&1 | tee logs/stage7_run_${SLURM_JOB_ID}.log

echo "Finished at $(date)"
