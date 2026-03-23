"""
U-Net v2 experiments: Fix the three issues from v1:
1. Steer only in semantic window (first 30% of steps, per Kwon et al.)
2. Use larger epsilons (SD activations have different scale than DiT)
3. Test steer_fraction ablation (0.3, 0.5, 1.0)

Also: test h-space (mid block) more thoroughly since that's the
most-studied location in the literature.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONCEPTS, VECTORS_DIR
from eval import get_concept_labels, Timer, get_peak_memory_mb
from extract import (
    extract_concept_vector_mean_diff, extract_concept_vector_pca,
    linear_probe,
)
from unet_adapter import (
    load_sd_pipeline, get_unet, UNetActivationCollector,
    generate_sd_and_collect, generate_sd_steered, generate_sd_baseline,
    UNET_HOOK_POINTS,
)

RESULTS_FILE = "results.tsv"
SD_VECTORS_DIR = os.path.join(VECTORS_DIR, "sd15")
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


def main():
    overall_timer = Timer()
    commit = get_commit()

    print(f"LASD U-Net v2 — Fixed Experiments")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(SD_VECTORS_DIR, exist_ok=True)

    pipe = load_sd_pipeline()
    unet = get_unet(pipe)

    # Check if we already have vectors from v1
    has_vectors = os.path.exists(os.path.join(SD_VECTORS_DIR, "brightness_mid_meandiff.pt"))
    if not has_vectors:
        phase("Phase 0: Extract vectors (re-run from v1)")
        collector = UNetActivationCollector(unet)
        images, activations, prompts = generate_sd_and_collect(pipe, 300, collector)
        collector.remove_hooks()

        for concept_name in ["brightness", "colorfulness"]:
            concept_cfg = CONCEPTS[concept_name]
            labels = get_concept_labels(concept_cfg, images, None)
            labels_t = torch.tensor(labels, dtype=torch.float32)
            mask = labels_t != 0

            for point_name, acts in activations.items():
                acts_valid = acts[mask]
                labels_valid = labels_t[mask]
                vec_md = extract_concept_vector_mean_diff(acts_valid, labels_valid)
                torch.save({"vector": vec_md, "method": "mean_diff", "point": point_name},
                           Path(SD_VECTORS_DIR) / f"{concept_name}_{point_name}_meandiff.pt")
                vec_pca = extract_concept_vector_pca(acts_valid, labels_valid)
                torch.save({"vector": vec_pca, "method": "pca", "point": point_name},
                           Path(SD_VECTORS_DIR) / f"{concept_name}_{point_name}_pca.pt")

    # Shared baseline
    eval_prompts = ["a photograph"] * EVAL_N
    print(f"Generating {EVAL_N} baseline images...")
    baseline_images, _ = generate_sd_baseline(pipe, EVAL_N, prompts=eval_prompts)
    baseline_labels = get_concept_labels(CONCEPTS["brightness"], baseline_images, None)
    baseline_rate = (baseline_labels == 1).mean()
    print(f"Baseline brightness rate: {baseline_rate:.3f}")

    # =====================================================================
    # Phase 1: Steer fraction ablation at h-space (mid block)
    # =====================================================================
    phase("Phase 1: Steer fraction ablation at h-space")

    mid_path = Path(SD_VECTORS_DIR) / "brightness_mid_meandiff.pt"
    if mid_path.exists():
        data = torch.load(mid_path, map_location="cpu", weights_only=True)
        vectors = {"mid": data["vector"]}

        for steer_frac in [0.2, 0.3, 0.5, 0.7, 1.0]:
            for eps in [1.0, 2.0, 5.0, 10.0]:
                print(f"\n  mid mean_diff frac={steer_frac} eps={eps}:")
                steered, _ = generate_sd_steered(
                    pipe, EVAL_N, vectors, epsilon=eps,
                    prompts=eval_prompts, steer_fraction=steer_frac,
                )
                labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
                rate = (labels == 1).mean()
                lift = rate - baseline_rate
                print(f"    rate={rate:.3f}, lift={lift:+.3f}")

                log_result(commit, "S2-SDv2", "lift", lift, get_gpu_mem_gb(),
                           "keep" if abs(lift) > 0.02 else "discard",
                           f"SD15v2 brightness mid mean_diff frac={steer_frac} eps={eps}")

    # =====================================================================
    # Phase 2: Block sweep with semantic window (frac=0.3)
    # =====================================================================
    phase("Phase 2: Block sweep with semantic window (frac=0.3)")

    for point_name in ["down_0", "down_1", "down_2", "mid", "up_1", "up_2", "up_3"]:
        vec_path = Path(SD_VECTORS_DIR) / f"brightness_{point_name}_meandiff.pt"
        if not vec_path.exists():
            continue

        data = torch.load(vec_path, map_location="cpu", weights_only=True)
        vectors = {point_name: data["vector"]}

        for eps in [2.0, 5.0, -2.0, -5.0]:
            print(f"\n  {point_name} eps={eps} frac=0.3:")
            steered, _ = generate_sd_steered(
                pipe, EVAL_N, vectors, epsilon=eps,
                prompts=eval_prompts, steer_fraction=0.3,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S2-SDv2", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"SD15v2 brightness {point_name} frac=0.3 eps={eps}")

    # =====================================================================
    # Phase 3: PCA at best block
    # =====================================================================
    phase("Phase 3: PCA at mid block with semantic window")

    pca_path = Path(SD_VECTORS_DIR) / "brightness_mid_pca.pt"
    if pca_path.exists():
        data = torch.load(pca_path, map_location="cpu", weights_only=True)
        vectors = {"mid": data["vector"]}

        for eps in [2.0, 5.0, 10.0, -2.0, -5.0, -10.0]:
            print(f"\n  mid PCA eps={eps} frac=0.3:")
            steered, _ = generate_sd_steered(
                pipe, EVAL_N, vectors, epsilon=eps,
                prompts=eval_prompts, steer_fraction=0.3,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S2-SDv2", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"SD15v2 brightness mid pca frac=0.3 eps={eps}")

    # =====================================================================
    # Phase 4: Multi-block steering with semantic window
    # =====================================================================
    phase("Phase 4: Multi-block steering")

    multi_configs = {
        "down_all": ["down_0", "down_1", "down_2"],
        "h_space_expanded": ["down_2", "mid", "up_0"],
        "up_all": ["up_1", "up_2", "up_3"],
        "all_blocks": list(UNET_HOOK_POINTS.keys()),
    }

    for config_name, points in multi_configs.items():
        vectors = {}
        for pt in points:
            vec_path = Path(SD_VECTORS_DIR) / f"brightness_{pt}_meandiff.pt"
            if vec_path.exists():
                data = torch.load(vec_path, map_location="cpu", weights_only=True)
                vectors[pt] = data["vector"]

        if not vectors:
            continue

        for eps in [2.0, 5.0, -2.0, -5.0]:
            print(f"\n  {config_name} ({len(vectors)} blocks) eps={eps} frac=0.3:")
            steered, _ = generate_sd_steered(
                pipe, EVAL_N, vectors, epsilon=eps,
                prompts=eval_prompts, steer_fraction=0.3,
            )
            labels = get_concept_labels(CONCEPTS["brightness"], steered, None)
            rate = (labels == 1).mean()
            lift = rate - baseline_rate
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S2-SDv2", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"SD15v2 brightness {config_name} frac=0.3 eps={eps}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE — U-Net v2 Experiments")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")

    print("\n--- results.tsv (SDv2 entries) ---")
    with open(RESULTS_FILE) as f:
        for line in f:
            if "SDv2" in line or line.startswith("commit"):
                print(line.rstrip())


if __name__ == "__main__":
    main()
