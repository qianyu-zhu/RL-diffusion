#!/bin/bash
#SBATCH --job-name=lasd-tv2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/teaser_v2_%j.out
#SBATCH --error=logs/teaser_v2_%j.err

set -e
module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf
cd /home/qianyu_z/RL-diffusion
mkdir -p figures/teaser_v2
python run_teaser_v2.py
echo "=== Done: $(date) ==="
