#!/bin/bash
#SBATCH --job-name=lasd-s4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage4_%j.out
#SBATCH --error=logs/stage4_%j.err
#SBATCH --dependency=afterok:10775637

# LASD Stage 4: Composition and Scale
# Depends on Stage 3 (which depends on Stage 2)

set -e

echo "=== LASD Stage 4: Composition & Scale ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs vectors results

python run_stage4.py 2>&1 | tee logs/stage4_run_${SLURM_JOB_ID}.log

# Run analysis after all stages complete
echo ""
echo "=== Running analysis ==="
python analyze_results.py 2>&1 | tee logs/analysis_${SLURM_JOB_ID}.log

echo ""
echo "=== Done: $(date) ==="
