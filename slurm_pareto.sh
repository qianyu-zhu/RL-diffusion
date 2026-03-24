#!/bin/bash
#SBATCH --job-name=lasd-pareto
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pareto_%j.out
#SBATCH --error=logs/pareto_%j.err

set -e
echo "=== LASD Pareto Frontier ==="
echo "Start: $(date)"

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs

python run_pareto.py 2>&1 | tee logs/pareto_run_${SLURM_JOB_ID}.log

echo "=== Done: $(date) ==="
