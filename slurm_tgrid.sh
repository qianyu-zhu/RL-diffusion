#!/bin/bash
#SBATCH --job-name=lasd-tgrd
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/tgrid_%j.out
#SBATCH --error=logs/tgrid_%j.err

module load miniforge/24.3.0-0
source activate cali-conf
cd /home/qianyu_z/RL-diffusion
python make_teaser_grid.py
echo "Done"
