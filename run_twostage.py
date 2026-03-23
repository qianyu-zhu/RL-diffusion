"""
Two-stage steering comparison (per Wang et al. 2026).

Tests:
1. Wang-style norm scaling: h + eps * ||h|| * v  vs our h + eps * v
2. Two-stage: PCA early + RFM/mean-diff late (mimic Wang's approach)
3. All-layer as implicit two-stage (our proposed simplification)
4. Layer-specific epsilon (stronger at early layers, weaker at late)

Hypothesis: all-layer steering works because it provides guidance at ALL
noise levels simultaneously, while single-layer can only help at one
noise level — making it a simpler alternative to Wang's two-stage.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, DTYPE, NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE,
    CONCEPTS, VECTORS_DIR,
)
from eval import get_concept_labels, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer, ActivationCollector,
    generate_and_collect, extract_concept_vector_mean_diff,
    extract_concept_vector_pca,
)
from steer import (
    load_concept_vectors, generate_steered, generate_baseline,
    SCHEDULES,
)

RESULTS_FILE = "results.tsv"
EVAL_N = 200


def log_result(commit, stage, metric, value, memory_gb, status, description):
    header = "commit\tstage\tmetric\tvalue\tmemory_gb\tstatus\tdescription\n"
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(header)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def get_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def phase(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")


def main():
    overall_timer = Timer()
    commit = get_commit()

    print(f"LASD Two-Stage Steering Comparison")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipe = load_dit_pipeline()

    class_ids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
    print(f"Generating {EVAL_N} baseline images...")
    baseline, _ = generate_baseline(pipe, EVAL_N, class_ids=class_ids)
    baseline_labels = get_concept_labels(CONCEPTS["brightness"], baseline, class_ids)
    baseline_rate = (baseline_labels == 1).mean()
    print(f"Baseline brightness rate: {baseline_rate:.3f}")

    # =====================================================================
    phase("Phase 1: Norm scaling comparison (Wang et al. vs ours)")
    # =====================================================================

    for layers_desc, layers in [("all", None), ("layer0", [0]), ("first5", list(range(5)))]:
        vectors = load_concept_vectors("brightness", method="mean_diff", layers=layers)
        if not vectors:
            continue

        for eps in [-0.5, -0.1]:
            for norm_scale in [False, True]:
                label = f"norm_scale={norm_scale}"
                steered, _ = generate_steered(
                    pipe, EVAL_N, vectors, epsilon=eps,
                    class_ids=class_ids, norm_scale=norm_scale,
                )
                labels = get_concept_labels(CONCEPTS["brightness"], steered, class_ids)
                rate = (labels == 1).mean()
                lift = rate - baseline_rate
                print(f"  {layers_desc} eps={eps} {label}: lift={lift:+.3f}")

                log_result(commit, "S6", "lift", lift, get_gpu_mem_gb(),
                           "keep" if abs(lift) > 0.02 else "discard",
                           f"twostage {layers_desc} {label} eps={eps}")

    # =====================================================================
    phase("Phase 2: Layer-weighted epsilon (stronger at early layers)")
    # =====================================================================

    vectors = load_concept_vectors("brightness", method="mean_diff", layers=None)
    if vectors:
        # Exponentially decaying epsilon per layer
        for decay in [0.9, 0.95, 0.97]:
            layer_eps = {}
            for l in vectors:
                layer_eps[l] = -0.5 * (decay ** l)

            steered, _ = generate_steered(
                pipe, EVAL_N, vectors,
                epsilon=layer_eps,
                class_ids=class_ids,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, class_ids)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"  decay={decay}: eps[0]={layer_eps[0]:.3f}, eps[27]={layer_eps.get(27, 0):.3f}, lift={lift:+.3f}")

            log_result(commit, "S6", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"twostage layer_decay={decay} all-layer")

    # =====================================================================
    phase("Phase 3: Two-stage — early PCA + late mean-diff")
    # =====================================================================

    # Simulate two-stage by using different vectors at different layers:
    # Early layers (0-9): PCA vectors (captures variance = noise alignment analog)
    # Late layers (18-27): mean-diff vectors (discriminative steering)
    pca_vectors = load_concept_vectors("brightness", method="pca", layers=list(range(10)))
    md_vectors = load_concept_vectors("brightness", method="mean_diff", layers=list(range(18, 28)))

    if pca_vectors and md_vectors:
        combined = {**pca_vectors, **md_vectors}

        for eps in [-0.5, -0.3, -0.1]:
            steered, _ = generate_steered(
                pipe, EVAL_N, combined, epsilon=eps, class_ids=class_ids,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, class_ids)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"  early_PCA+late_MD eps={eps}: lift={lift:+.3f}")

            log_result(commit, "S6", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"twostage earlyPCA+lateMD eps={eps}")

    # Also test: early mean-diff + late PCA (inverted)
    md_early = load_concept_vectors("brightness", method="mean_diff", layers=list(range(10)))
    pca_late = load_concept_vectors("brightness", method="pca", layers=list(range(18, 28)))

    if md_early and pca_late:
        combined = {**md_early, **pca_late}
        for eps in [-0.5, -0.3]:
            steered, _ = generate_steered(
                pipe, EVAL_N, combined, epsilon=eps, class_ids=class_ids,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, class_ids)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"  early_MD+late_PCA eps={eps}: lift={lift:+.3f}")

            log_result(commit, "S6", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"twostage earlyMD+latePCA eps={eps}")

    # =====================================================================
    phase("Phase 4: All-layer with different methods")
    # =====================================================================

    for method in ["mean_diff", "pca"]:
        vectors = load_concept_vectors("brightness", method=method, layers=None)
        if not vectors:
            continue
        for eps in [-0.5, -0.3, -0.1]:
            steered, _ = generate_steered(
                pipe, EVAL_N, vectors, epsilon=eps, class_ids=class_ids,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, class_ids)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"  all-layer {method} eps={eps}: lift={lift:+.3f}")

            log_result(commit, "S6", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"twostage all-layer {method} eps={eps}")

    # =====================================================================
    phase("DONE — Two-Stage Comparison")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    print("\n--- results.tsv (S6 entries) ---")
    with open(RESULTS_FILE) as f:
        for line in f:
            if "S6" in line or line.startswith("commit"):
                print(line.rstrip())


if __name__ == "__main__":
    main()
