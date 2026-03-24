#!/bin/bash
#SBATCH --job-name=lasd-neg
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/negative_%j.out
#SBATCH --error=logs/negative_%j.err

set -e
module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf
cd /home/qianyu_z/RL-diffusion
python run_negative_clip.py 2>&1 | tee logs/negative_run_${SLURM_JOB_ID}.log
echo "=== Done: $(date) ==="
