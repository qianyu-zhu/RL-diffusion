#!/bin/bash
#SBATCH --job-name=lasd-thv2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/theory_v2_%j.out
#SBATCH --error=logs/theory_v2_%j.err

# LASD Theory v2: Fixed callback issues. Re-runs temporal stability + multi-step probing.

set -e

echo "=== LASD Theory v2 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs vectors

# Temporal stability (now with hook-based step tracking, no callback)
echo ""
echo "========================================"
echo "  Experiment 1: Temporal Stability"
echo "========================================"
python extract.py --mode stability --concepts brightness,colorfulness --n_samples 200 --layers 0,5,10,15,20,25,27 2>&1 | tee logs/stability_v2_${SLURM_JOB_ID}.log

# Multi-step probing (now with hook-based step tracking)
echo ""
echo "========================================"
echo "  Experiment 2: Multi-step Probing"
echo "========================================"
python extract.py --mode probe_multistep --concepts brightness,colorfulness --n_samples 200 --layers 0,7,14,21,27 2>&1 | tee logs/probe_multistep_v2_${SLURM_JOB_ID}.log

# Semantic concept extraction + steering at layer 20 (best for animal)
echo ""
echo "========================================"
echo "  Experiment 3: Semantic Concept Steering (layer 20)"
echo "========================================"
python extract.py --mode extract --concepts animal,natural --n_samples 300 --layers 0,10,20,25 2>&1 | tee logs/semantic_extract_${SLURM_JOB_ID}.log

echo ""
echo "=== Done: $(date) ==="
