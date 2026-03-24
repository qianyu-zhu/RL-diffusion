#!/bin/bash
#SBATCH --job-name=lasd-teas
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/teaser_%j.out
#SBATCH --error=logs/teaser_%j.err

set -e
module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf
cd /home/qianyu_z/RL-diffusion
mkdir -p figures/teaser
python run_teaser.py
echo "=== Done: $(date) ==="
