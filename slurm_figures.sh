#!/bin/bash
#SBATCH --job-name=lasd-fig
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err

module load miniforge/24.3.0-0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p figures

python make_figures.py

echo "=== Done: $(date) ==="
