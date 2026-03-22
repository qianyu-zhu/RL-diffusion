"""
LASD experiments on U-Net (Stable Diffusion v1.5).
Mirrors the DiT Stage 2 experiments for cross-architecture comparison.

Experiments:
  1. Linearity probing at each U-Net block
  2. Block sweep: steering brightness at each hook point
  3. Epsilon sweep at best block
  4. Method comparison (mean-diff, PCA, logreg)
  5. h-space (mid block) comparison — the most studied location in literature
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONCEPTS, VECTORS_DIR
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb
from extract import (
    extract_concept_vector_mean_diff, extract_concept_vector_pca,
    linear_probe, mlp_probe,
)
from unet_adapter import (
    load_sd_pipeline, get_unet, UNetActivationCollector,
    generate_sd_and_collect, generate_sd_steered, generate_sd_baseline,
    UNET_HOOK_POINTS,
)

RESULTS_FILE = "results.tsv"
EXTRACT_N = 300
EVAL_N = 100
SD_VECTORS_DIR = os.path.join(VECTORS_DIR, "sd15")


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

    print(f"LASD U-Net Experiments (Stable Diffusion v1.5)")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(SD_VECTORS_DIR, exist_ok=True)

    pipe = load_sd_pipeline()
    unet = get_unet(pipe)

    # =====================================================================
    # Phase 1: Extract and probe
    # =====================================================================
    phase("Phase 1: Activation collection and probing")

    collector = UNetActivationCollector(unet)
    print(f"Generating {EXTRACT_N} images and collecting activations...")
    images, activations, prompts = generate_sd_and_collect(pipe, EXTRACT_N, collector)
    collector.remove_hooks()

    print(f"Generated {len(images)} images")
    print(f"Hook points with activations: {list(activations.keys())}")

    for concept_name in ["brightness", "colorfulness"]:
        concept_cfg = CONCEPTS[concept_name]
        # SD doesn't have class_ids — use None for heuristic labelers
        labels = get_concept_labels(concept_cfg, images, None)
        labels_t = torch.tensor(labels)
        mask = labels_t != 0

        n_pos = (labels == 1).sum()
        n_neg = (labels == -1).sum()
        print(f"\n  {concept_name}: {n_pos} positive, {n_neg} negative")

        best_acc = 0.0
        best_point = None

        for point_name, acts in sorted(activations.items()):
            lin_acc, vec_lr = linear_probe(acts, labels_t)
            mlp_acc = mlp_probe(acts, labels_t)
            dim = acts.shape[1]
            print(f"    {point_name:8s} (d={dim:4d}): linear={lin_acc:.3f}  mlp={mlp_acc:.3f}  gap={mlp_acc-lin_acc:+.3f}")

            if lin_acc > best_acc:
                best_acc = lin_acc
                best_point = point_name

            # Extract and save vectors
            labels_valid = labels_t[mask].float()
            acts_valid = acts[mask]

            vec_md = extract_concept_vector_mean_diff(acts_valid, labels_valid)
            torch.save({"vector": vec_md, "method": "mean_diff", "point": point_name},
                       Path(SD_VECTORS_DIR) / f"{concept_name}_{point_name}_meandiff.pt")

            vec_pca = extract_concept_vector_pca(acts_valid, labels_valid)
            torch.save({"vector": vec_pca, "method": "pca", "point": point_name},
                       Path(SD_VECTORS_DIR) / f"{concept_name}_{point_name}_pca.pt")

            if vec_lr is not None:
                torch.save({"vector": vec_lr, "method": "logreg", "point": point_name},
                           Path(SD_VECTORS_DIR) / f"{concept_name}_{point_name}_logreg.pt")

        print(f"  Best: {best_point} (acc={best_acc:.3f})")

        log_result(commit, "S1-SD", "probe_acc", best_acc, get_gpu_mem_gb(),
                   "gate_pass" if best_acc > 0.8 else "gate_fail",
                   f"SD15 {concept_name} probing, best={best_point}")

    # =====================================================================
    # Phase 2: Block sweep for brightness
    # =====================================================================
    phase("Phase 2: Block sweep — brightness steering")

    # Use fixed prompts for fair comparison
    eval_prompts = ["a photograph"] * EVAL_N

    print(f"Generating {EVAL_N} baseline images...")
    baseline_images, _ = generate_sd_baseline(pipe, EVAL_N, prompts=eval_prompts)
    baseline_labels = get_concept_labels(CONCEPTS["brightness"], baseline_images, None)
    baseline_rate = (baseline_labels == 1).mean()
    print(f"Baseline brightness rate: {baseline_rate:.3f}")

    block_results = {}
    for point_name in UNET_HOOK_POINTS:
        vec_path = Path(SD_VECTORS_DIR) / f"brightness_{point_name}_meandiff.pt"
        if not vec_path.exists():
            continue

        data = torch.load(vec_path, map_location="cpu", weights_only=True)
        vectors = {point_name: data["vector"]}

        for eps in [0.5, -0.5]:
            print(f"\n  {point_name} eps={eps}:")
            steered, _ = generate_sd_steered(
                pipe, EVAL_N, vectors, epsilon=eps, prompts=eval_prompts
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            block_results[(point_name, eps)] = lift
            log_result(commit, "S2-SD", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"SD15 brightness mean_diff {point_name} eps={eps}")

    # Find best block
    best_block = None
    best_lift = 0.0
    best_eps_sign = 1.0
    for (point_name, eps), lift in block_results.items():
        effective = lift if eps > 0 else -lift
        if effective > best_lift:
            best_lift = effective
            best_block = point_name
            best_eps_sign = 1.0 if eps > 0 else -1.0

    print(f"\n  BEST: {best_block}, effective lift={best_lift:+.3f}, sign={'positive' if best_eps_sign > 0 else 'negative'}")

    if best_block is None:
        best_block = "mid"
        best_eps_sign = 1.0

    # =====================================================================
    # Phase 3: Epsilon sweep at best block
    # =====================================================================
    phase(f"Phase 3: Epsilon sweep at {best_block}")

    vec_path = Path(SD_VECTORS_DIR) / f"brightness_{best_block}_meandiff.pt"
    data = torch.load(vec_path, map_location="cpu", weights_only=True)
    vectors = {best_block: data["vector"]}

    for eps_mag in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        eps = eps_mag * best_eps_sign
        print(f"\n  eps={eps}:")
        steered, _ = generate_sd_steered(
            pipe, EVAL_N, vectors, epsilon=eps, prompts=eval_prompts
        )
        labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
        rate = (labels == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S2-SD", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"SD15 brightness mean_diff {best_block} eps={eps}")

    # =====================================================================
    # Phase 4: Method comparison at best block
    # =====================================================================
    phase(f"Phase 4: Method comparison at {best_block}")

    best_eps = 0.1 * best_eps_sign  # use moderate eps

    for method_name, suffix in [("mean_diff", "meandiff"), ("pca", "pca"), ("logreg", "logreg")]:
        vec_path = Path(SD_VECTORS_DIR) / f"brightness_{best_block}_{suffix}.pt"
        if not vec_path.exists():
            print(f"  {method_name}: no vector, skipping")
            continue

        data = torch.load(vec_path, map_location="cpu", weights_only=True)
        vectors = {best_block: data["vector"]}

        print(f"\n  {method_name}:")
        steered, _ = generate_sd_steered(
            pipe, EVAL_N, vectors, epsilon=best_eps, prompts=eval_prompts
        )
        labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
        rate = (labels == 1).mean()
        lift = rate - baseline_rate
        print(f"    rate={rate:.3f}, lift={lift:+.3f}")

        log_result(commit, "S2-SD", "lift", lift, get_gpu_mem_gb(),
                   "keep" if abs(lift) > 0.02 else "discard",
                   f"SD15 brightness {method_name} {best_block} eps={best_eps}")

    # =====================================================================
    # Phase 5: h-space comparison
    # =====================================================================
    phase("Phase 5: h-space (mid block) — literature baseline")

    vec_path = Path(SD_VECTORS_DIR) / f"brightness_mid_meandiff.pt"
    if vec_path.exists():
        data = torch.load(vec_path, map_location="cpu", weights_only=True)
        vectors = {"mid": data["vector"]}

        for eps in [0.1, 0.5, 1.0]:
            print(f"\n  h-space eps={eps}:")
            steered, _ = generate_sd_steered(
                pipe, EVAL_N, vectors, epsilon=eps, prompts=eval_prompts
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S2-SD", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"SD15 brightness h-space (mid) eps={eps}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE — U-Net Experiments")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")

    print("\n--- results.tsv (SD entries) ---")
    with open(RESULTS_FILE) as f:
        for line in f:
            if "SD" in line or line.startswith("commit"):
                print(line.rstrip())


if __name__ == "__main__":
    main()
