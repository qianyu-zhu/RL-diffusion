# Environment Setup

## Local (macOS / Apple Silicon)

```bash
git clone https://github.com/qianyu-zhu/RL-diffusion.git
cd RL-diffusion
git checkout autoresearch/mar21

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download DiT-XL/2 model (~2.5GB)
uv run python setup.py

# Verify
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## HPC (Linux + CUDA)

```bash
git clone https://github.com/qianyu-zhu/RL-diffusion.git
cd RL-diffusion
git checkout autoresearch/mar21

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (will pick up CUDA torch automatically)
uv sync

# Download model
uv run python setup.py

# Verify
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}')"
```

### HPC Config Changes

Edit `config.py` for your GPU:

```python
# For A100 / H100:
DEVICE = "cuda"
DTYPE = torch.float16
DEFAULT_N_SAMPLES = 5000
NFA_N_SAMPLES = 1200
EVAL_N_IMAGES = 500
EVAL_BATCH_SIZE = 16
NUM_INFERENCE_STEPS = 50

# For smaller GPUs (V100, RTX 3090):
EVAL_BATCH_SIZE = 8
DEFAULT_N_SAMPLES = 2000
```

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=lasd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

module load python cuda
cd $HOME/RL-diffusion
uv run python extract.py --mode probe --concepts brightness,colorfulness,animal,natural --n_samples 5000 > run.log 2>&1
```

## Dependencies

All managed via `pyproject.toml`:
- torch >= 2.0
- diffusers >= 0.30
- transformers
- accelerate
- safetensors
- timm
- numpy, scipy, scikit-learn
- pillow, tqdm

## Project Structure

```
RL-diffusion/
├── config.py           # Model config, concepts, hyperparams (READ ONLY during experiments)
├── eval.py             # Labelers, metrics, summary printer (READ ONLY)
├── extract.py          # Activation hooks, RFM/AGOP, probing, NFA (MODIFY)
├── steer.py            # Steering hooks, evaluation, ablation (MODIFY)
├── setup.py            # One-time model download + smoke test
├── pyproject.toml      # Dependencies
├── program.md          # Autonomous experiment protocol
├── RESEARCH_LOG.md     # Experiment log with all results
├── SETUP.md            # This file
├── main.tex            # Literature review
├── proposal.tex        # LASD research proposal
├── .agents/skills/     # Claude agent skills for research automation
├── .claude/skills/     # Claude skills (paper writing, experiment dev, etc.)
├── vectors/            # Extracted concept vectors (gitignored, regenerate via extract.py)
└── results/            # Saved images from steering (gitignored)
```

## Quick Start: Resume Experiments

```bash
# 1. Re-extract vectors (needed since vectors/ is gitignored)
uv run python extract.py --mode extract --concepts brightness,colorfulness --n_samples 5000

# 2. Run Stage 2 layer sweep (the key experiment)
for layer in 0 5 10 15 20 25 27; do
    uv run python steer.py --mode evaluate --concept brightness --method mean_diff --epsilon 0.5 --n_images 500 --layers $layer > run_layer${layer}.log 2>&1
done

# 3. Check results
grep "lift:" run_layer*.log
```
