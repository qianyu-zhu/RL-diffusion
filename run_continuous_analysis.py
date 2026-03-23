"""
Continuous metric analysis: does steering shift CONTINUOUS concept scores
even when binary labels don't change? Verifies that the lift metric
isn't hiding real effects.

Also tests the hypothesis that class-conditional concepts (animal/natural)
can't be steered because class identity enters via adaLN, not the residual
stream — by measuring continuous brightness scores for both concepts.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, DTYPE, NUM_CLASSES, NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, CONCEPTS, VECTORS_DIR,
)
from eval import get_concept_labels, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer, ActivationCollector,
    generate_and_collect, extract_concept_vector_mean_diff,
    extract_concept_vector_pca,
)
from steer import load_concept_vectors, generate_steered, generate_baseline

EVAL_N = 200


def compute_continuous_brightness(images):
    """Return mean pixel value for each image (continuous, not thresholded)."""
    return np.array([np.array(img).astype(np.float32).mean() / 255.0 for img in images])


def compute_continuous_colorfulness(images):
    """Return color channel std for each image (continuous)."""
    scores = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            scores.append(arr.std(axis=2).mean())
        else:
            scores.append(0.0)
    return np.array(scores)


def phase(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")


def main():
    timer = Timer()
    print(f"LASD Continuous Analysis")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipe = load_dit_pipeline()

    class_ids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
    print(f"Generating {EVAL_N} baseline images...")
    baseline, _ = generate_baseline(pipe, EVAL_N, class_ids=class_ids)

    base_brightness = compute_continuous_brightness(baseline)
    base_colorfulness = compute_continuous_colorfulness(baseline)

    print(f"Baseline brightness: mean={base_brightness.mean():.4f}, std={base_brightness.std():.4f}")
    print(f"Baseline colorfulness: mean={base_colorfulness.mean():.4f}, std={base_colorfulness.std():.4f}")

    # =====================================================================
    phase("Phase 1: Continuous brightness shift")

    configs = [
        ("all-layer mean_diff eps=-0.5", "mean_diff", None, -0.5),
        ("all-layer pca eps=-0.5", "pca", None, -0.5),
        ("layer0 mean_diff eps=-0.5", "mean_diff", [0], -0.5),
        ("all-layer mean_diff eps=+0.5", "mean_diff", None, 0.5),
    ]

    for desc, method, layers, eps in configs:
        vectors = load_concept_vectors("brightness", method=method, layers=layers)
        if not vectors:
            continue

        steered, _ = generate_steered(pipe, EVAL_N, vectors, epsilon=eps, class_ids=class_ids)
        s_brightness = compute_continuous_brightness(steered)
        s_colorfulness = compute_continuous_colorfulness(steered)

        b_shift = s_brightness.mean() - base_brightness.mean()
        c_shift = s_colorfulness.mean() - base_colorfulness.mean()
        b_tstat = b_shift / (np.sqrt(s_brightness.var()/EVAL_N + base_brightness.var()/EVAL_N) + 1e-8)

        print(f"\n  {desc}:")
        print(f"    brightness: {base_brightness.mean():.4f} → {s_brightness.mean():.4f} "
              f"(Δ={b_shift:+.4f}, t={b_tstat:+.2f})")
        print(f"    colorfulness: {base_colorfulness.mean():.4f} → {s_colorfulness.mean():.4f} "
              f"(Δ={c_shift:+.4f})")

    # =====================================================================
    phase("Phase 2: Colorfulness continuous shift")

    for desc, method, layers, eps in [
        ("all-layer mean_diff eps=-0.5", "mean_diff", None, -0.5),
        ("all-layer pca eps=-0.5", "pca", None, -0.5),
    ]:
        vectors = load_concept_vectors("colorfulness", method=method, layers=layers)
        if not vectors:
            continue

        steered, _ = generate_steered(pipe, EVAL_N, vectors, epsilon=eps, class_ids=class_ids)
        s_brightness = compute_continuous_brightness(steered)
        s_colorfulness = compute_continuous_colorfulness(steered)

        b_shift = s_brightness.mean() - base_brightness.mean()
        c_shift = s_colorfulness.mean() - base_colorfulness.mean()

        print(f"\n  colorfulness {desc}:")
        print(f"    brightness: Δ={b_shift:+.4f} (cross-concept leakage)")
        print(f"    colorfulness: Δ={c_shift:+.4f}")

    # =====================================================================
    phase("Phase 3: Why semantic steering fails")
    print("  Hypothesis: class-conditional concepts (animal/natural) can't be")
    print("  steered because class identity enters via adaLN, not residual stream.")
    print("  Test: measure continuous brightness shift when steering with 'animal' vector.")

    # Extract animal vector at layer 20 (best probe layer)
    transformer = get_dit_transformer(pipe)
    collector = ActivationCollector(transformer, layers=[0, 10, 20])
    imgs, acts, cids = generate_and_collect(pipe, 300, collector)
    collector.remove_hooks()

    labels = get_concept_labels(CONCEPTS["animal"], imgs, cids)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    mask = labels_t != 0

    for layer_idx in [0, 10, 20]:
        if layer_idx not in acts:
            continue
        acts_valid = acts[layer_idx][mask]
        labels_valid = labels_t[mask]
        vec = extract_concept_vector_mean_diff(acts_valid, labels_valid)
        vecs = {layer_idx: vec}

        steered, _ = generate_steered(pipe, EVAL_N, vecs, epsilon=-1.0, class_ids=class_ids)
        s_brightness = compute_continuous_brightness(steered)
        b_shift = s_brightness.mean() - base_brightness.mean()

        # Check animal rate
        s_labels = get_concept_labels(CONCEPTS["animal"], steered, class_ids)
        animal_rate_base = (get_concept_labels(CONCEPTS["animal"], baseline, class_ids) == 1).mean()
        animal_rate_steered = (s_labels == 1).mean()

        print(f"\n  animal vector at layer {layer_idx}, eps=-1.0:")
        print(f"    animal rate: {animal_rate_base:.3f} → {animal_rate_steered:.3f} "
              f"(Δ={animal_rate_steered - animal_rate_base:+.3f})")
        print(f"    brightness shift: Δ={b_shift:+.4f}")
        print(f"    Note: animal rate CAN'T change because it's determined by class_id,")
        print(f"    not by image content. Class conditioning enters via adaLN.")

    # =====================================================================
    phase("DONE")
    print(f"Total wall time: {timer.elapsed():.0f}s")


if __name__ == "__main__":
    main()
