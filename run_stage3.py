"""
Autonomous Stage 3 ablation runner for HPC.
Designed to run AFTER Stage 2 completes — reads Stage 2 results to determine
best layer/method/epsilon, then systematically ablates.

Ablations:
  1. Layer selection strategies: single-best, last-5, middle-5, all layers
  2. Data efficiency: M = 50, 100, 200, 500 (re-extract vectors with fewer samples)
  3. RFM iterations: R = 1, 2, 3, 5
  4. Epsilon schedule: constant, linear_decay, linear_ramp, cosine
  5. Norm clip threshold: 1.2, 1.5, 2.0, inf (no clip)
  6. Strategy B vs A comparison (if temporal stability vectors available)
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE,
    CONCEPTS, VECTORS_DIR, EPSILON_RANGE,
)
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer, ActivationCollector,
    generate_and_collect, extract_concept_vector_mean_diff,
    extract_concept_vector_pca, extract_concept_vector_rfm,
    linear_probe,
)
from steer import (
    load_concept_vectors, load_binned_vectors, generate_steered, generate_baseline,
    SCHEDULES,
)

RESULTS_FILE = "results.tsv"
EVAL_N = 100


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


def read_stage2_best():
    """Read results.tsv to determine best config from Stage 2."""
    if not os.path.exists(RESULTS_FILE):
        return {"layer": 25, "eps": 0.5, "method": "mean_diff"}

    best_layer = 25
    best_eps = 0.5
    best_lift = -999
    best_method = "mean_diff"

    with open(RESULTS_FILE) as f:
        for line in f:
            if line.startswith("commit"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            stage, metric, value, status, desc = parts[1], parts[2], float(parts[3]), parts[5], parts[6]
            if stage == "S2" and metric == "lift" and status == "keep":
                if value > best_lift:
                    best_lift = value
                    # Parse layer and eps from description
                    for token in desc.split():
                        if token.startswith("layer"):
                            try:
                                best_layer = int(token.replace("layer", "").strip("[]"))
                            except ValueError:
                                pass
                        if token.startswith("eps="):
                            try:
                                best_eps = float(token.replace("eps=", ""))
                            except ValueError:
                                pass
                    for m in ["mean_diff", "rfm", "pca", "logreg"]:
                        if m in desc:
                            best_method = m
                            break

    return {"layer": best_layer, "eps": best_eps, "method": best_method, "lift": best_lift}


def main():
    overall_timer = Timer()
    commit = get_commit()
    concept = "brightness"
    concept_cfg = CONCEPTS[concept]

    print(f"LASD Stage 3 — Ablation Studies")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Read best config from Stage 2
    best = read_stage2_best()
    print(f"\nStage 2 best config:")
    print(f"  Layer: {best['layer']}, Eps: {best['eps']}, Method: {best.get('method', 'mean_diff')}")
    print(f"  Lift: {best.get('lift', 'unknown')}")

    pipe = load_dit_pipeline()
    transformer = get_dit_transformer(pipe)

    # Shared baseline
    baseline_cids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
    print(f"\nGenerating {EVAL_N} baseline images...")
    baseline_images, _ = generate_baseline(pipe, EVAL_N, class_ids=baseline_cids)
    baseline_labels = get_concept_labels(concept_cfg, baseline_images, baseline_cids)
    baseline_rate = (baseline_labels == 1).mean()
    print(f"Baseline brightness positive rate: {baseline_rate:.3f}")

    best_layer = best["layer"]
    best_eps = best["eps"]

    # =====================================================================
    # Ablation 1: Layer selection strategies
    # =====================================================================
    phase("Ablation 1: Layer Selection Strategies")

    layer_strategies = {
        "single_best": [best_layer],
        "last_5": list(range(23, 28)),
        "last_10": list(range(18, 28)),
        "middle_5": list(range(11, 16)),
        "first_5": list(range(0, 5)),
        "all": list(range(28)),
        "even": list(range(0, 28, 2)),
    }

    for strategy_name, layers in layer_strategies.items():
        vectors = load_concept_vectors(concept, method="mean_diff", layers=layers)
        if not vectors:
            print(f"  {strategy_name}: no vectors, skipping")
            continue

        print(f"\n  {strategy_name} (layers={layers}):")
        steered, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=best_eps, class_ids=baseline_cids
        )
        labels = get_concept_labels(concept_cfg, steered, baseline_cids)
        rate = (labels == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S3", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"layer_selection {strategy_name} eps={best_eps}")

    # =====================================================================
    # Ablation 2: Epsilon schedule
    # =====================================================================
    phase("Ablation 2: Epsilon Schedule")

    vectors = load_concept_vectors(concept, method="mean_diff", layers=[best_layer])

    for sched_name, sched_fn in SCHEDULES.items():
        print(f"\n  Schedule: {sched_name}")
        steered, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=best_eps, class_ids=baseline_cids,
            epsilon_schedule=sched_fn,
        )
        labels = get_concept_labels(concept_cfg, steered, baseline_cids)
        rate = (labels == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S3", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"schedule {sched_name} layer{best_layer} eps={best_eps}")

    # =====================================================================
    # Ablation 3: Norm clip threshold
    # =====================================================================
    phase("Ablation 3: Norm Clip Threshold")

    for clip in [1.2, 1.5, 2.0, 5.0, 0]:  # 0 = no clipping
        label = "no_clip" if clip == 0 else f"clip_{clip}"
        print(f"\n  {label}:")
        steered, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=best_eps, class_ids=baseline_cids,
            norm_clip=clip if clip > 0 else -1,
        )
        labels = get_concept_labels(concept_cfg, steered, baseline_cids)
        rate = (labels == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S3", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"norm_clip {label} layer{best_layer} eps={best_eps}")

    # =====================================================================
    # Ablation 4: Data efficiency (re-extract vectors with fewer samples)
    # =====================================================================
    phase("Ablation 4: Data Efficiency")

    for M in [50, 100, 200, 500]:
        print(f"\n  M={M}: generating and extracting...")
        collector = ActivationCollector(transformer, layers=[best_layer])
        imgs, acts, cids = generate_and_collect(pipe, M, collector)
        collector.remove_hooks()

        labels = get_concept_labels(concept_cfg, imgs, cids)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask = labels_t != 0

        if best_layer not in acts:
            continue

        acts_valid = acts[best_layer][mask]
        labels_valid = labels_t[mask]

        vec_md = extract_concept_vector_mean_diff(acts_valid, labels_valid)

        # Steer with this vector
        vecs = {best_layer: vec_md}
        steered, _ = generate_steered(
            pipe, EVAL_N, vecs, epsilon=best_eps, class_ids=baseline_cids
        )
        labels_s = get_concept_labels(concept_cfg, steered, baseline_cids)
        rate = (labels_s == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S3", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"data_efficiency M={M} mean_diff layer{best_layer} eps={best_eps}")

    # =====================================================================
    # Ablation 5: Strategy B (time-binned) vs Strategy A
    # =====================================================================
    phase("Ablation 5: Strategy B vs A")

    for n_bins in [3, 4, 5]:
        binned = load_binned_vectors(concept, method="mean_diff",
                                     n_bins=n_bins, layers=[best_layer])
        if not binned:
            print(f"  n_bins={n_bins}: no step-specific vectors found, skipping")
            print("  (Run extract.py --mode stability first to generate per-step vectors)")
            continue

        print(f"\n  Strategy B, {n_bins} bins:")
        steered, _ = generate_steered(
            pipe, EVAL_N, {}, epsilon=best_eps, class_ids=baseline_cids,
            strategy="B", binned_vectors=binned,
        )
        labels = get_concept_labels(concept_cfg, steered, baseline_cids)
        rate = (labels == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S3", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"strategy_B bins={n_bins} layer{best_layer} eps={best_eps}")

    # =====================================================================
    # Ablation 6: RFM iterations
    # =====================================================================
    phase("Ablation 6: RFM Iterations")

    # Re-use the M=500 activations from earlier if available, else generate
    print("  Generating 500 samples for RFM ablation...")
    collector = ActivationCollector(transformer, layers=[best_layer])
    imgs, acts, cids = generate_and_collect(pipe, 500, collector)
    collector.remove_hooks()

    labels = get_concept_labels(concept_cfg, imgs, cids)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    mask = labels_t != 0

    if best_layer in acts:
        acts_valid = acts[best_layer][mask]
        labels_valid = labels_t[mask]

        for R in [1, 2, 3, 5]:
            print(f"\n  RFM iterations R={R}:")
            try:
                vec_rfm, _ = extract_concept_vector_rfm(acts_valid, labels_valid, n_iter=R)
                vecs = {best_layer: vec_rfm}
                steered, _ = generate_steered(
                    pipe, EVAL_N, vecs, epsilon=best_eps, class_ids=baseline_cids
                )
                labels_s = get_concept_labels(concept_cfg, steered, baseline_cids)
                rate = (labels_s == 1).mean()
                lift = rate - baseline_rate
                print(f"    rate={rate:.3f}, lift={lift:+.3f}")

                log_result(commit, "S3", "lift", lift, get_gpu_mem_gb(),
                           "keep" if abs(lift) > 0.02 else "discard",
                           f"rfm_iter R={R} layer{best_layer} eps={best_eps}")
            except Exception as e:
                print(f"    FAILED: {e}")
                log_result(commit, "S3", "lift", 0.0, get_gpu_mem_gb(),
                           "crash", f"rfm_iter R={R} crashed: {e}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE — Stage 3 Ablations")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")

    print("\n--- results.tsv (S3 entries) ---")
    with open(RESULTS_FILE) as f:
        for line in f:
            if "S3" in line or line.startswith("commit"):
                print(line.rstrip())


if __name__ == "__main__":
    main()
