"""
Predictive-Causal Gap Analysis — THE KEY FIGURE.

For each of 28 DiT layers, compute:
1. Predictive accuracy (linear probe accuracy)
2. Causal effectiveness (steering lift with eps=-0.5)

The gap between these tells us where information is ENCODED vs
where perturbations are CAUSALLY EFFECTIVE.

Also analyzes WHY all-layer steering works:
- Is it additive (sum of individual effects)?
- Or synergistic (layers interact)?
- Compare: all-layer lift vs sum-of-individual-lifts
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
    extract_concept_vector_pca, linear_probe,
)
from steer import load_concept_vectors, generate_steered, generate_baseline

RESULTS_FILE = "results.tsv"
EVAL_N = 200  # moderate for balance of speed and significance


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

    print(f"LASD Predictive-Causal Gap Analysis")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipe = load_dit_pipeline()
    transformer = get_dit_transformer(pipe)

    # =====================================================================
    # Phase 1: Comprehensive probe accuracy at every layer
    # =====================================================================
    phase("Phase 1: Probe accuracy (all 28 layers)")

    all_layers = list(range(28))
    collector = ActivationCollector(transformer, layers=all_layers)
    print(f"Generating 500 images for probing...")
    images, activations, class_ids = generate_and_collect(pipe, 500, collector)
    collector.remove_hooks()

    labels_b = get_concept_labels(CONCEPTS["brightness"], images, class_ids)
    labels_t = torch.tensor(labels_b)

    probe_accs = {}
    for layer_idx in all_layers:
        if layer_idx not in activations:
            continue
        acc, _ = linear_probe(activations[layer_idx], labels_t)
        probe_accs[layer_idx] = acc

    print(f"\n  Layer | Probe Acc")
    print(f"  {'-'*5}-+-{'-'*9}")
    for l in sorted(probe_accs):
        print(f"  {l:5d} | {probe_accs[l]:.3f}")

    # =====================================================================
    # Phase 2: Single-layer steering lift at every layer
    # =====================================================================
    phase("Phase 2: Single-layer steering (all 28 layers)")

    # Shared baseline
    eval_cids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
    print(f"Generating {EVAL_N} baseline images...")
    baseline_images, _ = generate_baseline(pipe, EVAL_N, class_ids=eval_cids)
    baseline_labels = get_concept_labels(CONCEPTS["brightness"], baseline_images, eval_cids)
    baseline_rate = (baseline_labels == 1).mean()
    print(f"Baseline brightness rate: {baseline_rate:.3f}")

    # Extract mean-diff vectors at all layers (using existing or freshly extracted)
    labels_float = torch.tensor(labels_b, dtype=torch.float32)
    mask = labels_float != 0

    single_lifts = {}
    for layer_idx in all_layers:
        if layer_idx not in activations:
            continue

        acts_valid = activations[layer_idx][mask]
        labels_valid = labels_float[mask]

        vec = extract_concept_vector_mean_diff(acts_valid, labels_valid)
        vecs = {layer_idx: vec}

        # Try both signs
        best_lift = 0.0
        for eps in [-0.5, 0.5]:
            steered, _ = generate_steered(pipe, EVAL_N, vecs, epsilon=eps, class_ids=eval_cids)
            labels_s = get_concept_labels(CONCEPTS["brightness"], steered, eval_cids)
            rate = (labels_s == 1).mean()
            lift = rate - baseline_rate
            if lift > best_lift:
                best_lift = lift

        single_lifts[layer_idx] = best_lift
        print(f"  Layer {layer_idx:2d}: probe={probe_accs.get(layer_idx, 0):.3f}, "
              f"lift={best_lift:+.3f}")

        log_result(commit, "GAP", "lift", best_lift, get_gpu_mem_gb(),
                   "keep", f"gap_analysis brightness layer{layer_idx} best_lift")

    # =====================================================================
    # Phase 3: Print the gap table
    # =====================================================================
    phase("Phase 3: Predictive-Causal Gap Table")

    print(f"  {'Layer':>5s} | {'Probe':>6s} | {'Lift':>6s} | {'Gap':>6s}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for l in sorted(probe_accs):
        p = probe_accs.get(l, 0)
        c = single_lifts.get(l, 0)
        gap = p - c
        print(f"  {l:5d} | {p:.3f}  | {c:+.3f} | {gap:+.3f}")

    # Correlation
    layers_both = sorted(set(probe_accs.keys()) & set(single_lifts.keys()))
    p_vals = np.array([probe_accs[l] for l in layers_both])
    c_vals = np.array([single_lifts[l] for l in layers_both])
    corr = np.corrcoef(p_vals, c_vals)[0, 1]
    print(f"\n  Pearson correlation (probe acc vs lift): {corr:.3f}")

    # =====================================================================
    # Phase 4: Why all-layer works — additivity analysis
    # =====================================================================
    phase("Phase 4: Additivity analysis")

    sum_individual = sum(single_lifts.values())
    print(f"  Sum of individual lifts: {sum_individual:+.3f}")

    # All-layer steering
    all_vecs = {}
    for layer_idx in all_layers:
        if layer_idx not in activations:
            continue
        acts_valid = activations[layer_idx][mask]
        labels_valid = labels_float[mask]
        all_vecs[layer_idx] = extract_concept_vector_mean_diff(acts_valid, labels_valid)

    for eps in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
        steered, _ = generate_steered(pipe, EVAL_N, all_vecs, epsilon=eps, class_ids=eval_cids)
        labels_s = get_concept_labels(CONCEPTS["brightness"], steered, eval_cids)
        rate = (labels_s == 1).mean()
        lift = rate - baseline_rate
        print(f"  All-layer eps={eps:+.1f}: lift={lift:+.3f}")

        log_result(commit, "GAP", "lift", lift, get_gpu_mem_gb(),
                   "keep", f"gap_analysis brightness all-layer eps={eps}")

    # Cumulative layer analysis: add one layer at a time
    print("\n  Cumulative (adding layers 0, 1, 2, ...):")
    cumulative_vecs = {}
    for layer_idx in sorted(all_vecs.keys()):
        cumulative_vecs[layer_idx] = all_vecs[layer_idx]
        if layer_idx % 4 == 0 or layer_idx == 27:
            steered, _ = generate_steered(
                pipe, EVAL_N, cumulative_vecs, epsilon=-0.5, class_ids=eval_cids
            )
            labels_s = get_concept_labels(CONCEPTS["brightness"], steered, eval_cids)
            rate = (labels_s == 1).mean()
            lift = rate - baseline_rate
            print(f"    Layers 0-{layer_idx:2d} ({len(cumulative_vecs)} layers): lift={lift:+.3f}")

            log_result(commit, "GAP", "lift", lift, get_gpu_mem_gb(),
                       "keep", f"gap_analysis cumulative layers0-{layer_idx}")

    # =====================================================================
    # Phase 5: PCA all-layer (the star result)
    # =====================================================================
    phase("Phase 5: PCA all-layer steering")

    pca_vecs = {}
    for layer_idx in all_layers:
        if layer_idx not in activations:
            continue
        acts_valid = activations[layer_idx][mask]
        labels_valid = labels_float[mask]
        pca_vecs[layer_idx] = extract_concept_vector_pca(acts_valid, labels_valid)

    for eps in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
        steered, _ = generate_steered(pipe, EVAL_N, pca_vecs, epsilon=eps, class_ids=eval_cids)
        labels_s = get_concept_labels(CONCEPTS["brightness"], steered, eval_cids)
        rate = (labels_s == 1).mean()
        lift = rate - baseline_rate
        print(f"  PCA all-layer eps={eps:+.1f}: lift={lift:+.3f}")

        log_result(commit, "GAP", "lift", lift, get_gpu_mem_gb(),
                   "keep", f"gap_analysis brightness PCA all-layer eps={eps}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE — Gap Analysis")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")


if __name__ == "__main__":
    main()
