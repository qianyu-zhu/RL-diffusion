"""
LASD evaluation harness. READ ONLY — do not modify during experiments.
"""
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from config import DEVICE, EVAL_N_IMAGES


# ---------------------------------------------------------------------------
# Concept labelers
# ---------------------------------------------------------------------------

def label_brightness(images: list[Image.Image]) -> np.ndarray:
    """Label images as bright (1) or dark (-1) based on mean pixel value."""
    labels = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        labels.append(1 if arr.mean() > 0.5 else -1)
    return np.array(labels)


def label_colorfulness(images: list[Image.Image]) -> np.ndarray:
    """Label images as colorful (1) or desaturated (-1)."""
    labels = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Colorfulness = std across color channels, averaged over pixels
            color_std = arr.std(axis=2).mean()
            labels.append(1 if color_std > 0.08 else -1)
        else:
            labels.append(-1)
    return np.array(labels)


def label_class_group(images: list[Image.Image], class_ids: list[int],
                      positive_range: list[int], negative_range: list[int]) -> np.ndarray:
    """Label based on whether class_id falls in positive or negative range."""
    labels = []
    for cid in class_ids:
        if cid in positive_range:
            labels.append(1)
        elif cid in negative_range:
            labels.append(-1)
        else:
            labels.append(0)  # neither — will be filtered
    return np.array(labels)


LABELERS = {
    "brightness_heuristic": label_brightness,
    "colorfulness_heuristic": label_colorfulness,
    "class_group": label_class_group,
}


def get_concept_labels(concept_config: dict, images: list[Image.Image],
                       class_ids: list[int] | None = None) -> np.ndarray:
    """Get binary labels for a concept."""
    labeler_name = concept_config["labeler"]
    if labeler_name == "class_group":
        assert class_ids is not None
        return label_class_group(
            images, class_ids,
            concept_config["positive_range"],
            concept_config["negative_range"],
        )
    else:
        return LABELERS[labeler_name](images)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def concept_accuracy(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Fraction of images where steering moved the concept in the right direction."""
    mask = labels_true != 0
    if mask.sum() == 0:
        return 0.0
    return (labels_pred[mask] == labels_true[mask]).mean()


def mean_concept_score(scores: np.ndarray) -> float:
    """Mean continuous concept score (e.g., brightness value)."""
    return float(scores.mean())


def compute_diversity(images: list[Image.Image]) -> float:
    """Simple diversity metric: mean pairwise L2 distance in pixel space (normalized)."""
    if len(images) < 2:
        return 0.0
    arrays = [np.array(img).astype(np.float32).flatten() / 255.0 for img in images]
    n = min(len(arrays), 50)  # cap for speed
    arrays = arrays[:n]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(arrays[i] - arrays[j]))
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Summary printer (matches program.md output format)
# ---------------------------------------------------------------------------

def print_summary(**kwargs):
    """Print experiment summary in the standard format."""
    print("---")
    for k, v in kwargs.items():
        if isinstance(v, float):
            print(f"{k}:{' ' * max(1, 18 - len(k))}{v:.6f}")
        else:
            print(f"{k}:{' ' * max(1, 18 - len(k))}{v}")


class Timer:
    """Simple wall-clock timer."""
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time


def get_peak_memory_mb() -> float:
    """Get peak memory usage. On MPS, use process RSS as approximation."""
    try:
        import resource
        # maxrss is in bytes on macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return 0.0
