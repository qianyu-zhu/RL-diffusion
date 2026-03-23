#!/bin/bash
#SBATCH --job-name=fk-baseline
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fk_%j.out
#SBATCH --error=logs/fk_%j.err

source ~/.bashrc
conda activate cali-conf

cd /orcd/home/002/qianyu_z/RL-diffusion

echo "Starting FK baseline experiments at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python run_fk_baseline.py 2>&1 | tee logs/fk_run_${SLURM_JOB_ID}.log

echo "Finished at $(date)"
