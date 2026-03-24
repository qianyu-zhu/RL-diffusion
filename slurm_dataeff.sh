#!/bin/bash
#SBATCH --job-name=lasd-deff
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dataeff_%j.out
#SBATCH --error=logs/dataeff_%j.err

set -e
module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf
cd /home/qianyu_z/RL-diffusion
python run_data_efficiency_clip.py 2>&1 | tee logs/dataeff_run_${SLURM_JOB_ID}.log
echo "=== Done: $(date) ==="
