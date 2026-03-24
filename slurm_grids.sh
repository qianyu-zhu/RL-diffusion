#!/bin/bash
#SBATCH --job-name=lasd-grid
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/grids_%j.out
#SBATCH --error=logs/grids_%j.err

module load miniforge/24.3.0-0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
python make_image_grid.py
echo "=== Done ==="
