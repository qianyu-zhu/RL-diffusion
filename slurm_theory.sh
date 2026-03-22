#!/bin/bash
#SBATCH --job-name=lasd-theory
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/theory_%j.out
#SBATCH --error=logs/theory_%j.err

# LASD Theory Validation: modified NFA + temporal stability + multi-step probing
# Independent of Stage 2 — can run in parallel

set -e

echo "=== LASD Theory Validation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

module load miniforge/24.3.0-0 cuda/12.4.0
source activate cali-conf

cd /home/qianyu_z/RL-diffusion
mkdir -p logs vectors

# ---------------------------------------------------------------
# Experiment 1: Modified NFA with adaLN-conditioned weights
# Tests theory prediction: W_eff(t)^T W_eff(t) ∝ AGOP
# Uses 200 samples (enough for meaningful AGOP in 1152-d with modulation)
# ---------------------------------------------------------------
echo ""
echo "========================================"
echo "  Experiment 1: Modified NFA (adaLN)"
echo "========================================"
python extract.py --mode nfa_adaln --n_samples 200 2>&1 | tee logs/nfa_adaln_${SLURM_JOB_ID}.log
echo ""

# ---------------------------------------------------------------
# Experiment 2: Temporal stability of concept vectors
# Extracts mean-diff vectors at 6 denoising steps, computes
# pairwise cosine stability. Also saves per-step vectors for Strategy B.
# Uses 200 samples, select layers
# ---------------------------------------------------------------
echo ""
echo "========================================"
echo "  Experiment 2: Temporal Stability"
echo "========================================"
python extract.py --mode stability --concepts brightness,colorfulness --n_samples 200 --layers 0,5,10,15,20,25,27 2>&1 | tee logs/stability_${SLURM_JOB_ID}.log
echo ""

# ---------------------------------------------------------------
# Experiment 3: Multi-timestep probing
# Probe accuracy at different denoising steps — reveals when
# concepts become linearly decodable during denoising.
# Uses 200 samples, select layers
# ---------------------------------------------------------------
echo ""
echo "========================================"
echo "  Experiment 3: Multi-step Probing"
echo "========================================"
python extract.py --mode probe_multistep --concepts brightness,colorfulness --n_samples 200 --layers 0,7,14,21,27 2>&1 | tee logs/probe_multistep_${SLURM_JOB_ID}.log
echo ""

# ---------------------------------------------------------------
# Experiment 4: Semantic concept probing (animal, natural)
# Addresses critic weakness #4: test non-trivial concepts
# Uses 300 samples for better class balance
# ---------------------------------------------------------------
echo ""
echo "========================================"
echo "  Experiment 4: Semantic Concept Probing"
echo "========================================"
python extract.py --mode probe --concepts animal,natural --n_samples 300 --layers 0,5,10,15,20,25,27 2>&1 | tee logs/semantic_probe_${SLURM_JOB_ID}.log
echo ""

echo "=== All theory experiments done: $(date) ==="
