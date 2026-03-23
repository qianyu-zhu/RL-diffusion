"""
Follow-up experiments on key findings from Stages 2-4.
Focuses on the biggest discoveries that need deeper investigation:

1. All-layer PCA steering (lift=+0.560 in S3, the star result)
2. Layer 0 eps=-0.5 mean-diff (lift=+0.250, the original finding)
3. Semantic concept steering at CORRECT layers with PCA
4. Full-scale evaluations with 500+ images for statistical significance
5. PCA vs mean-diff at the all-layer configuration
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
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer, ActivationCollector,
    generate_and_collect, extract_concept_vector_mean_diff,
    extract_concept_vector_pca, linear_probe,
)
from steer import (
    load_concept_vectors, generate_steered, generate_baseline,
)

RESULTS_FILE = "results.tsv"


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

    print(f"LASD Follow-up Experiments")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipe = load_dit_pipeline()
    transformer = get_dit_transformer(pipe)

    # =====================================================================
    # Phase 1: Full eval of best configs with 500 images
    # =====================================================================
    phase("Phase 1: Full-scale evaluation (500 images)")

    EVAL_N = 500
    class_ids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()

    print(f"Generating {EVAL_N} baseline images...")
    baseline_images, _ = generate_baseline(pipe, EVAL_N, class_ids=class_ids)
    baseline_labels_b = get_concept_labels(CONCEPTS["brightness"], baseline_images, class_ids)
    baseline_labels_c = get_concept_labels(CONCEPTS["colorfulness"], baseline_images, class_ids)
    baseline_rate_b = (baseline_labels_b == 1).mean()
    baseline_rate_c = (baseline_labels_c == 1).mean()
    print(f"Baseline brightness: {baseline_rate_b:.3f}")
    print(f"Baseline colorfulness: {baseline_rate_c:.3f}")

    # Save sample baseline images
    os.makedirs("results/followup", exist_ok=True)
    for i in range(20):
        baseline_images[i].save(f"results/followup/baseline_{i:03d}.png")

    configs = [
        # (description, method, layers, eps)
        ("all-layer mean_diff eps=-0.5", "mean_diff", None, -0.5),
        ("all-layer pca eps=-0.5", "pca", None, -0.5),
        ("all-layer pca eps=-0.3", "pca", None, -0.3),
        ("all-layer pca eps=-0.1", "pca", None, -0.1),
        ("all-layer pca eps=+0.5", "pca", None, 0.5),
        ("layer0 mean_diff eps=-0.5", "mean_diff", [0], -0.5),
        ("layer0 pca eps=-0.5", "pca", [0], -0.5),
        ("first5 pca eps=-0.5", "pca", list(range(5)), -0.5),
        ("first5 mean_diff eps=-0.5", "mean_diff", list(range(5)), -0.5),
    ]

    for desc, method, layers, eps in configs:
        vectors = load_concept_vectors("brightness", method=method, layers=layers)
        if not vectors:
            print(f"  {desc}: no vectors, skipping")
            continue

        print(f"\n  {desc} ({len(vectors)} layers):")
        steered, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=eps, class_ids=class_ids
        )
        labels_b = get_concept_labels(CONCEPTS["brightness"], steered, class_ids)
        labels_c = get_concept_labels(CONCEPTS["colorfulness"], steered, class_ids)
        rate_b = (labels_b == 1).mean()
        rate_c = (labels_c == 1).mean()
        lift_b = rate_b - baseline_rate_b
        lift_c = rate_c - baseline_rate_c
        print(f"    brightness: rate={rate_b:.3f}, lift={lift_b:+.3f}")
        print(f"    colorfulness: rate={rate_c:.3f}, lift={lift_c:+.3f}")

        # Save sample steered images for best configs
        if abs(lift_b) > 0.1:
            save_dir = f"results/followup/{desc.replace(' ', '_')}"
            os.makedirs(save_dir, exist_ok=True)
            for i in range(20):
                steered[i].save(f"{save_dir}/steered_{i:03d}.png")

        log_result(commit, "S5", "lift", lift_b, get_gpu_mem_gb(),
                   "keep" if abs(lift_b) > 0.02 else "discard",
                   f"FULL500 brightness {desc}")
        log_result(commit, "S5", "lift", lift_c, get_gpu_mem_gb(),
                   "keep" if abs(lift_c) > 0.02 else "discard",
                   f"FULL500 colorfulness {desc}")

    # =====================================================================
    # Phase 2: Colorfulness-specific steering
    # =====================================================================
    phase("Phase 2: Colorfulness all-layer steering")

    for method in ["pca", "mean_diff"]:
        for eps in [-0.5, -0.3, 0.5]:
            vectors = load_concept_vectors("colorfulness", method=method, layers=None)
            if not vectors:
                continue
            print(f"\n  colorfulness all-layer {method} eps={eps} ({len(vectors)} layers):")
            steered, _ = generate_steered(
                pipe, EVAL_N, vectors, epsilon=eps, class_ids=class_ids
            )
            labels = get_concept_labels(CONCEPTS["colorfulness"], steered, class_ids)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate_c
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S5", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"FULL500 colorfulness all-layer {method} eps={eps}")

    # =====================================================================
    # Phase 3: Semantic concepts at correct layers
    # =====================================================================
    phase("Phase 3: Semantic concept steering (correct layers)")

    # Animal best at layer 20, natural best at layer 10 (from theory v2)
    # Need to extract vectors at those layers first
    print("Extracting semantic concept vectors (500 samples)...")
    # Use select layers: 0, 10, 15, 20, 25 for semantic concepts
    sem_layers = [0, 10, 15, 20, 25]
    collector = ActivationCollector(transformer, layers=sem_layers)
    sem_images, sem_activations, sem_cids = generate_and_collect(pipe, 500, collector)
    collector.remove_hooks()

    sem_eval_cids = torch.randint(0, NUM_CLASSES, (200,)).tolist()
    print("Generating 200 semantic baseline images...")
    sem_baseline, _ = generate_baseline(pipe, 200, class_ids=sem_eval_cids)

    for concept_name, best_layers in [("animal", [0, 10, 20]), ("natural", [10, 15, 20])]:
        concept_cfg = CONCEPTS[concept_name]
        labels = get_concept_labels(concept_cfg, sem_images, sem_cids)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask = labels_t != 0

        n_pos = (labels == 1).sum()
        n_neg = (labels == -1).sum()
        print(f"\n  {concept_name}: {n_pos} positive, {n_neg} negative")

        if n_pos < 10 or n_neg < 10:
            print(f"    Too few samples, skipping")
            continue

        sem_base_labels = get_concept_labels(concept_cfg, sem_baseline, sem_eval_cids)
        sem_base_rate = (sem_base_labels == 1).mean()
        print(f"  Baseline rate: {sem_base_rate:.3f}")

        for layer_idx in best_layers:
            if layer_idx not in sem_activations:
                continue

            acts_valid = sem_activations[layer_idx][mask]
            labels_valid = labels_t[mask]

            # Probe accuracy
            acc, _ = linear_probe(sem_activations[layer_idx], labels_t.long())

            for method_name, extract_fn in [
                ("mean_diff", extract_concept_vector_mean_diff),
                ("pca", extract_concept_vector_pca),
            ]:
                vec = extract_fn(acts_valid, labels_valid)
                vecs = {layer_idx: vec}

                for eps in [-1.0, -0.5, 0.5, 1.0]:
                    steered, _ = generate_steered(
                        pipe, 200, vecs, epsilon=eps, class_ids=sem_eval_cids
                    )
                    steer_labels = get_concept_labels(concept_cfg, steered, sem_eval_cids)
                    steer_rate = (steer_labels == 1).mean()
                    lift = steer_rate - sem_base_rate

                    print(f"    {concept_name} {method_name} layer{layer_idx} eps={eps}: "
                          f"probe={acc:.3f}, lift={lift:+.3f}")

                    log_result(commit, "S5", "lift", lift, get_gpu_mem_gb(),
                               "keep" if abs(lift) > 0.02 else "discard",
                               f"semantic {concept_name} {method_name} layer{layer_idx} eps={eps}")

        # Also try all-layer steering for semantic concepts
        print(f"\n  {concept_name} all-layer steering:")
        all_vecs = {}
        for layer_idx in sem_layers:
            if layer_idx not in sem_activations:
                continue
            acts_valid = sem_activations[layer_idx][mask]
            labels_valid = labels_t[mask]
            all_vecs[layer_idx] = extract_concept_vector_mean_diff(acts_valid, labels_valid)

        if all_vecs:
            for eps in [-1.0, -0.5, 0.5, 1.0]:
                steered, _ = generate_steered(
                    pipe, 200, all_vecs, epsilon=eps, class_ids=sem_eval_cids
                )
                steer_labels = get_concept_labels(concept_cfg, steered, sem_eval_cids)
                steer_rate = (steer_labels == 1).mean()
                lift = steer_rate - sem_base_rate
                print(f"    all-layer mean_diff eps={eps}: lift={lift:+.3f}")

                log_result(commit, "S5", "lift", lift, get_gpu_mem_gb(),
                           "keep" if abs(lift) > 0.02 else "discard",
                           f"semantic {concept_name} all-layer mean_diff eps={eps}")

    # =====================================================================
    # Phase 4: Predictive vs Causal gap analysis
    # =====================================================================
    phase("Phase 4: Predictive vs Causal gap table")

    # Compile probe accuracy vs steering lift for each layer
    # Re-use the already-extracted vectors
    print("  Layer | Probe Acc | Lift (eps=-0.5) | Lift (eps=+0.5)")
    print("  " + "-" * 55)

    for layer_idx in [0, 5, 10, 15, 20, 25, 27]:
        # Probe accuracy from the original extraction
        if layer_idx in sem_activations:
            b_labels = get_concept_labels(CONCEPTS["brightness"], sem_images, sem_cids)
            b_labels_t = torch.tensor(b_labels)
            acc, _ = linear_probe(sem_activations[layer_idx], b_labels_t)
        else:
            acc = float('nan')

        # Load steering lifts from results.tsv
        lift_neg = "N/A"
        lift_pos = "N/A"
        with open(RESULTS_FILE) as f:
            for line in f:
                if f"layer{layer_idx} eps=-0.5" in line and "brightness mean_diff" in line and "S2\t" in line:
                    lift_neg = line.split("\t")[3]
                if f"layer{layer_idx} eps=0.5" in line and "brightness mean_diff" in line and "S2\t" in line:
                    lift_pos = line.split("\t")[3]

        print(f"  {layer_idx:5d} | {acc:9.3f} | {lift_neg:>15s} | {lift_pos:>15s}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE — Follow-up Experiments")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")

    # Print all S5 results
    print("\n--- results.tsv (S5 entries) ---")
    with open(RESULTS_FILE) as f:
        for line in f:
            if "S5" in line or line.startswith("commit"):
                print(line.rstrip())

    # Run analysis
    print("\n--- Running full analysis ---")
    os.system("python analyze_results.py")


if __name__ == "__main__":
    main()
