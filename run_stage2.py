"""
Autonomous Stage 2 experiment runner for HPC.
Extracts vectors, then runs layer sweep, epsilon sweep, method comparison.
Logs all results to results.tsv.
"""
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Ensure we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE,
    DEFAULT_N_SAMPLES, CONCEPTS, VECTORS_DIR, EPSILON_RANGE,
)
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer, ActivationCollector,
    generate_and_collect, extract_concept_vector_rfm,
    extract_concept_vector_mean_diff, extract_concept_vector_pca,
    linear_probe, mlp_probe,
)
from steer import (
    load_concept_vectors, generate_steered, generate_baseline,
    evaluate_steering,
)

RESULTS_FILE = "results.tsv"
SWEEP_LAYERS = [0, 5, 10, 15, 20, 25, 27]
# Use moderate sample sizes to fit within 6hr budget
EXTRACT_N_SAMPLES = 500   # 5x what M1 used, enough for decent vectors
EVAL_N = 100              # images per steering eval (quick pass)
EVAL_N_FULL = 250         # images for final best-config eval


def log_result(commit, stage, metric, value, memory_gb, status, description):
    """Append a row to results.tsv."""
    header = "commit\tstage\tmetric\tvalue\tmemory_gb\tstatus\tdescription\n"
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(header)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def get_commit():
    """Get current short git hash."""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_gpu_mem_gb():
    """Get peak GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def phase(name):
    """Print a phase header."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")


def main():
    overall_timer = Timer()
    commit = get_commit()

    print(f"LASD Stage 2 — Autonomous HPC Run")
    print(f"Commit: {commit}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Extract samples: {EXTRACT_N_SAMPLES}")
    print(f"Eval images: {EVAL_N}")
    print()

    # =====================================================================
    # Phase 1: Load model
    # =====================================================================
    phase("Phase 1: Loading DiT-XL/2-256")
    pipe = load_dit_pipeline()
    transformer = get_dit_transformer(pipe)
    print(f"Model loaded. Time: {overall_timer.elapsed():.0f}s")

    # =====================================================================
    # Phase 2: Extract concept vectors (brightness + colorfulness)
    # =====================================================================
    phase("Phase 2: Extracting concept vectors")
    os.makedirs(VECTORS_DIR, exist_ok=True)

    collector = ActivationCollector(transformer)
    print(f"Generating {EXTRACT_N_SAMPLES} images and collecting activations...")
    timer = Timer()
    images, activations, class_ids = generate_and_collect(
        pipe, EXTRACT_N_SAMPLES, collector
    )
    collector.remove_hooks()
    print(f"Generation done. Time: {timer.elapsed():.0f}s")

    for concept_name in ["brightness", "colorfulness"]:
        concept_cfg = CONCEPTS[concept_name]
        labels = get_concept_labels(concept_cfg, images, class_ids)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask = labels_t != 0

        n_pos = (labels == 1).sum()
        n_neg = (labels == -1).sum()
        print(f"\n  {concept_name}: {n_pos} positive, {n_neg} negative")

        # Probe accuracy at each layer (Stage 1 re-run with more data)
        best_acc = 0.0
        best_layer = -1
        for layer_idx in sorted(activations.keys()):
            acts = activations[layer_idx]
            lin_acc, vec_lr = linear_probe(acts, labels_t.long())
            mlp_acc = mlp_probe(acts, labels_t.long())
            print(f"    Layer {layer_idx:2d}: linear={lin_acc:.3f}  mlp={mlp_acc:.3f}  gap={mlp_acc-lin_acc:+.3f}")
            if lin_acc > best_acc:
                best_acc = lin_acc
                best_layer = layer_idx

            # Extract and save all vector types
            acts_valid = acts[mask]
            labels_valid = labels_t[mask]

            # Mean-diff
            vec_md = extract_concept_vector_mean_diff(acts_valid, labels_valid)
            torch.save({"vector": vec_md, "method": "mean_diff", "layer": layer_idx},
                       Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_meandiff.pt")

            # PCA
            vec_pca = extract_concept_vector_pca(acts_valid, labels_valid)
            torch.save({"vector": vec_pca, "method": "pca", "layer": layer_idx},
                       Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_pca.pt")

            # Logreg
            if vec_lr is not None:
                torch.save({"vector": vec_lr, "method": "logreg", "layer": layer_idx},
                           Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_logreg.pt")

        print(f"  Best layer: {best_layer} (acc={best_acc:.3f})")
        log_result(commit, "S1", "probe_acc", best_acc, get_gpu_mem_gb(),
                   "gate_pass" if best_acc > 0.8 else "gate_fail",
                   f"{concept_name} probing with {EXTRACT_N_SAMPLES} samples, best layer {best_layer}")

    # RFM extraction (only for select layers to save time)
    print("\n  Extracting RFM vectors (select layers)...")
    for concept_name in ["brightness", "colorfulness"]:
        concept_cfg = CONCEPTS[concept_name]
        labels = get_concept_labels(concept_cfg, images, class_ids)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask = labels_t != 0

        for layer_idx in SWEEP_LAYERS:
            if layer_idx not in activations:
                continue
            acts_valid = activations[layer_idx][mask]
            labels_valid = labels_t[mask]
            try:
                vec_rfm, agop = extract_concept_vector_rfm(acts_valid, labels_valid)
                torch.save({
                    "vector": vec_rfm, "agop": agop, "method": "rfm",
                    "concept": concept_name, "layer": layer_idx,
                    "n_samples": int(mask.sum()),
                }, Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_rfm.pt")
                print(f"    RFM {concept_name} layer {layer_idx}: done")
            except Exception as e:
                print(f"    RFM {concept_name} layer {layer_idx}: FAILED ({e})")

    extraction_time = overall_timer.elapsed()
    print(f"\nExtraction complete. Total time: {extraction_time:.0f}s")

    # =====================================================================
    # Phase 3: Stage 2 — Layer sweep (brightness, mean_diff)
    # =====================================================================
    phase("Phase 3: Layer sweep — brightness × mean_diff")

    # Generate shared baseline (reuse class_ids for fair comparison)
    baseline_class_ids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
    print(f"Generating {EVAL_N} baseline images...")
    baseline_images, _ = generate_baseline(pipe, EVAL_N, class_ids=baseline_class_ids)
    baseline_labels = get_concept_labels(CONCEPTS["brightness"], baseline_images, baseline_class_ids)
    baseline_rate = (baseline_labels == 1).mean()
    print(f"Baseline brightness positive rate: {baseline_rate:.3f}")

    layer_results = {}
    for layer_idx in SWEEP_LAYERS:
        vectors = load_concept_vectors("brightness", method="mean_diff", layers=[layer_idx])
        if not vectors:
            print(f"  Layer {layer_idx}: no vector, skipping")
            continue

        for eps in [0.5, -0.5]:
            print(f"\n  Layer {layer_idx}, eps={eps}:")
            steered_images, _ = generate_steered(
                pipe, EVAL_N, vectors, epsilon=eps, class_ids=baseline_class_ids
            )
            steered_labels = get_concept_labels(CONCEPTS["brightness"], steered_images, baseline_class_ids)
            steered_rate = (steered_labels == 1).mean()
            lift = steered_rate - baseline_rate
            print(f"    steered_rate={steered_rate:.3f}, lift={lift:+.3f}")

            key = (layer_idx, eps)
            layer_results[key] = {
                "concept_acc": float(steered_rate),
                "baseline_acc": float(baseline_rate),
                "lift": float(lift),
            }
            log_result(commit, "S2", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"brightness mean_diff layer{layer_idx} eps={eps}")

    # Find best layer (highest absolute lift with positive sign for eps>0 or negative for eps<0)
    best_layer = None
    best_lift = 0.0
    best_eps_sign = 1.0
    for (layer_idx, eps), res in layer_results.items():
        # We want positive lift for positive eps (steering toward concept)
        effective_lift = res["lift"] if eps > 0 else -res["lift"]
        if effective_lift > best_lift:
            best_lift = effective_lift
            best_layer = layer_idx
            best_eps_sign = 1.0 if eps > 0 else -1.0

    print(f"\n  BEST: layer {best_layer}, effective lift={best_lift:+.3f}, "
          f"eps sign={'positive' if best_eps_sign > 0 else 'negative'}")

    if best_layer is None:
        print("  WARNING: No layer showed positive steering effect. Using layer 25 as default.")
        best_layer = 25
        best_eps_sign = 1.0

    # =====================================================================
    # Phase 4: Epsilon sweep at best layer
    # =====================================================================
    phase(f"Phase 4: Epsilon sweep at layer {best_layer}")

    eps_results = {}
    vectors = load_concept_vectors("brightness", method="mean_diff", layers=[best_layer])

    for eps_mag in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        eps = eps_mag * best_eps_sign
        print(f"\n  eps={eps}:")
        steered_images, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=eps, class_ids=baseline_class_ids
        )
        steered_labels = get_concept_labels(CONCEPTS["brightness"], steered_images, baseline_class_ids)
        steered_rate = (steered_labels == 1).mean()
        lift = steered_rate - baseline_rate
        print(f"    steered_rate={steered_rate:.3f}, lift={lift:+.3f}")

        eps_results[eps] = {"lift": float(lift), "concept_acc": float(steered_rate)}
        log_result(commit, "S2", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"brightness mean_diff layer{best_layer} eps={eps}")

    # Find best epsilon
    best_eps = max(eps_results, key=lambda e: eps_results[e]["lift"])
    print(f"\n  BEST eps={best_eps}, lift={eps_results[best_eps]['lift']:+.3f}")

    # =====================================================================
    # Phase 5: Method comparison at best layer + best eps
    # =====================================================================
    phase(f"Phase 5: Method comparison at layer {best_layer}, eps={best_eps}")

    for method in ["mean_diff", "pca", "logreg", "rfm"]:
        vectors = load_concept_vectors("brightness", method=method, layers=[best_layer])
        if not vectors:
            print(f"  {method}: no vector found, skipping")
            log_result(commit, "S2", "lift", 0.0, get_gpu_mem_gb(), "crash",
                       f"brightness {method} layer{best_layer} eps={best_eps} — no vector")
            continue

        print(f"\n  Method: {method}")
        steered_images, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=best_eps, class_ids=baseline_class_ids
        )
        steered_labels = get_concept_labels(CONCEPTS["brightness"], steered_images, baseline_class_ids)
        steered_rate = (steered_labels == 1).mean()
        lift = steered_rate - baseline_rate
        print(f"    steered_rate={steered_rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S2", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"brightness {method} layer{best_layer} eps={best_eps}")

    # =====================================================================
    # Phase 6: Repeat best config for colorfulness
    # =====================================================================
    phase(f"Phase 6: Colorfulness steering at layer {best_layer}")

    color_baseline_labels = get_concept_labels(CONCEPTS["colorfulness"], baseline_images, baseline_class_ids)
    color_baseline_rate = (color_baseline_labels == 1).mean()
    print(f"Baseline colorfulness positive rate: {color_baseline_rate:.3f}")

    for method in ["mean_diff", "pca", "logreg"]:
        vectors = load_concept_vectors("colorfulness", method=method, layers=[best_layer])
        if not vectors:
            print(f"  {method}: no vector, skipping")
            continue

        for eps in [best_eps, -best_eps]:
            print(f"\n  colorfulness {method} eps={eps}:")
            steered_images, _ = generate_steered(
                pipe, EVAL_N, vectors, epsilon=eps, class_ids=baseline_class_ids
            )
            steered_labels = get_concept_labels(CONCEPTS["colorfulness"], steered_images, baseline_class_ids)
            steered_rate = (steered_labels == 1).mean()
            lift = steered_rate - color_baseline_rate
            print(f"    steered_rate={steered_rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S2", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"colorfulness {method} layer{best_layer} eps={eps}")

    # =====================================================================
    # Phase 7: Multi-layer steering (top 3 layers)
    # =====================================================================
    phase("Phase 7: Multi-layer steering")

    # Sort layers by absolute lift from Phase 3
    layer_lifts = {}
    for (layer_idx, eps), res in layer_results.items():
        effective = res["lift"] if eps > 0 else -res["lift"]
        if layer_idx not in layer_lifts or effective > layer_lifts[layer_idx]:
            layer_lifts[layer_idx] = effective

    top3_layers = sorted(layer_lifts, key=lambda l: layer_lifts[l], reverse=True)[:3]
    print(f"Top 3 layers by lift: {top3_layers}")

    vectors = load_concept_vectors("brightness", method="mean_diff", layers=top3_layers)
    if vectors:
        print(f"\n  Multi-layer ({top3_layers}) mean_diff eps={best_eps}:")
        steered_images, _ = generate_steered(
            pipe, EVAL_N, vectors, epsilon=best_eps, class_ids=baseline_class_ids
        )
        steered_labels = get_concept_labels(CONCEPTS["brightness"], steered_images, baseline_class_ids)
        steered_rate = (steered_labels == 1).mean()
        lift = steered_rate - baseline_rate
        print(f"    steered_rate={steered_rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S2", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"brightness mean_diff layers{top3_layers} eps={best_eps}")

    # =====================================================================
    # Phase 8: Full evaluation of best config with more images
    # =====================================================================
    phase("Phase 8: Full eval of best config")

    full_class_ids = torch.randint(0, NUM_CLASSES, (EVAL_N_FULL,)).tolist()
    print(f"Generating {EVAL_N_FULL} baseline images...")
    full_baseline, _ = generate_baseline(pipe, EVAL_N_FULL, class_ids=full_class_ids)
    full_baseline_labels = get_concept_labels(CONCEPTS["brightness"], full_baseline, full_class_ids)
    full_baseline_rate = (full_baseline_labels == 1).mean()

    vectors = load_concept_vectors("brightness", method="mean_diff", layers=[best_layer])
    print(f"Generating {EVAL_N_FULL} steered images (layer {best_layer}, eps={best_eps})...")
    full_steered, _ = generate_steered(
        pipe, EVAL_N_FULL, vectors, epsilon=best_eps, class_ids=full_class_ids
    )
    full_steered_labels = get_concept_labels(CONCEPTS["brightness"], full_steered, full_class_ids)
    full_steered_rate = (full_steered_labels == 1).mean()
    full_lift = full_steered_rate - full_baseline_rate

    # Save sample images
    os.makedirs("results/best_config", exist_ok=True)
    for i in range(min(20, EVAL_N_FULL)):
        full_baseline[i].save(f"results/best_config/baseline_{i:03d}.png")
        full_steered[i].save(f"results/best_config/steered_{i:03d}.png")

    print(f"\n  Full eval results:")
    print(f"    Baseline rate: {full_baseline_rate:.3f}")
    print(f"    Steered rate:  {full_steered_rate:.3f}")
    print(f"    Lift:          {full_lift:+.3f}")

    log_result(commit, "S2", "lift", full_lift, get_gpu_mem_gb(),
               "keep",
               f"FULL EVAL brightness mean_diff layer{best_layer} eps={best_eps} n={EVAL_N_FULL}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")
    print(f"\nResults saved to {RESULTS_FILE}")

    # Print results.tsv
    print("\n--- results.tsv ---")
    with open(RESULTS_FILE) as f:
        print(f.read())

    print_summary(
        wall_seconds=total_time,
        peak_vram_mb=get_gpu_mem_gb() * 1000,
    )


if __name__ == "__main__":
    main()
