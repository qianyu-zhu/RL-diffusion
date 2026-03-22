"""
LASD experiment configuration. READ ONLY — do not modify during experiments.
"""
import torch

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "facebook/DiT-XL-2-256"       # 675M params, class-conditional ImageNet 256x256
IMAGE_SIZE = 256
NUM_CLASSES = 1000
LATENT_CHANNELS = 4
PATCH_SIZE = 2
HIDDEN_DIM = 1152                        # DiT-XL hidden dimension
NUM_LAYERS = 28                          # DiT-XL depth
NUM_HEADS = 16

# Device
DEVICE = "cuda"
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0                     # CFG scale for class-conditional generation
SCHEDULER = "ddpm"

# ---------------------------------------------------------------------------
# Concepts — defined as binary classification tasks on generated images
# For class-conditional DiT, we define meta-concepts that cut across classes.
# ---------------------------------------------------------------------------
CONCEPTS = {
    "brightness": {
        "description": "bright vs dark images",
        "labeler": "brightness_heuristic",    # mean pixel value > 0.5
    },
    "colorfulness": {
        "description": "colorful vs desaturated images",
        "labeler": "colorfulness_heuristic",  # std of color channels
    },
    "natural": {
        "description": "natural scenes (animals, plants, landscapes) vs artificial (vehicles, objects)",
        "labeler": "class_group",
        "positive_range": list(range(0, 398)),    # animals + plants in ImageNet
        "negative_range": list(range(398, 700)),   # vehicles, objects, etc.
    },
    "animal": {
        "description": "animal vs non-animal",
        "labeler": "class_group",
        "positive_range": list(range(0, 398)),
        "negative_range": list(range(398, 1000)),
    },
}

# ---------------------------------------------------------------------------
# RFM defaults
# ---------------------------------------------------------------------------
RFM_ITERATIONS = 3
RFM_KERNEL = "laplace"
RFM_RIDGE_LAMBDA = 0.01
DEFAULT_N_SAMPLES = 5000
NFA_N_SAMPLES = 1200

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_N_IMAGES = 500
EVAL_BATCH_SIZE = 16
FID_N_REAL = 1000                        # real images for FID reference

# ---------------------------------------------------------------------------
# Steering defaults
# ---------------------------------------------------------------------------
DEFAULT_EPSILON = 0.1
EPSILON_RANGE = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
TIMESTEP_BINS = [3, 4, 5]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = "~/.cache/lasd"
RESULTS_DIR = "./results"
VECTORS_DIR = "./vectors"                # saved concept vectors
