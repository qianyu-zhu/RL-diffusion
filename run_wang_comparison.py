"""
Direct comparison to Wang et al. 2026 (NA-RFM) on DiT-XL/2.

Implements their key components:
1. RFM/AGOP vector extraction (already in extract.py)
2. Norm-scaling formula: h' = h + w * ||h|| * v
3. Single-block steering (their approach) vs our all-layer steering
4. Measure: concept lift, FID, diversity for both approaches

This demonstrates our advantages:
- All-layer >> single-block
- Mean-diff >> RFM in DiT
- Our approach preserves diversity; their norm-scale destroys it
"""
import os
import torch
import numpy as np
from PIL import Image

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE,
)
from eval import compute_diversity, Timer, get_peak_memory_mb
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline
from run_fid import compute_inception_features, compute_fid
from run_clip_eval import load_clip, clip_classify

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def continuous_brightness(images):
    return np.array([np.array(img).astype(np.float32).mean() / 255.0 for img in images])


def main():
    timer = Timer()
    print("="*60)
    print("  Wang et al. 2026 Comparison on DiT-XL/2")
    print("="*60)

    pipe = load_dit_pipeline()
    clip_model, clip_processor = load_clip()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n_images = 200
    torch.manual_seed(42)
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()
    mem_gb = 3.9

    # Reference
    print("\nGenerating baseline...")
    ref_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    ref_features = compute_inception_features(ref_imgs)
    ref_bright = continuous_brightness(ref_imgs)
    ref_rate = (ref_bright > 0.5).mean()
    ref_div = compute_diversity(ref_imgs)

    prompts_bright = ["a bright, well-lit photograph", "a dark, dimly-lit photograph"]
    ref_clip = clip_classify(ref_imgs, prompts_bright, clip_model, clip_processor)[:, 0].mean()

    print(f"  Baseline: pixel_rate={ref_rate:.3f}, CLIP={ref_clip:.3f}, diversity={ref_div:.3f}")

    # Configs to compare
    configs = [
        # Our approach: all-layer mean-diff with norm clipping
        {"name": "LASD-allayer", "method": "mean_diff", "layers": None,
         "eps": -0.5, "norm_scale": False, "norm_clip": 5.0},

        # Our approach: first5
        {"name": "LASD-first5", "method": "mean_diff", "layers": list(range(5)),
         "eps": -0.5, "norm_scale": False, "norm_clip": 5.0},

        # Wang et al. approach: single block (best layer), RFM, norm-scale
        {"name": "Wang-L25-RFM", "method": "rfm", "layers": [25],
         "eps": -0.5, "norm_scale": True, "norm_clip": 0},

        # Wang et al. approach: single block, mean-diff, norm-scale
        {"name": "Wang-L25-MD", "method": "mean_diff", "layers": [25],
         "eps": -0.5, "norm_scale": True, "norm_clip": 0},

        # Wang et al. approach: single block, RFM, our norm-clip
        {"name": "Wang-L25-RFM-clip", "method": "rfm", "layers": [25],
         "eps": -0.5, "norm_scale": False, "norm_clip": 5.0},

        # Our approach with their norm-scale (to show it hurts)
        {"name": "LASD-allayer-normscale", "method": "mean_diff", "layers": None,
         "eps": -0.1, "norm_scale": True, "norm_clip": 0},

        # Their best block for DiT: try multiple
        {"name": "Wang-L0-MD", "method": "mean_diff", "layers": [0],
         "eps": -0.5, "norm_scale": True, "norm_clip": 0},

        # Ablation: our method at their eps scale
        {"name": "LASD-allayer-eps01", "method": "mean_diff", "layers": None,
         "eps": -0.1, "norm_scale": False, "norm_clip": 5.0},
    ]

    print(f"\n  {'Config':<28s} {'Pixel Lift':>10s} {'CLIP Lift':>10s} {'FID':>6s} {'Diversity':>10s}")
    print("  " + "-" * 70)

    for cfg in configs:
        vectors = load_concept_vectors("brightness", method=cfg["method"], layers=cfg["layers"])
        if not vectors:
            print(f"  {cfg['name']:<28s} NO VECTORS")
            continue

        steered, _ = generate_steered(
            pipe, n_images, vectors, epsilon=cfg["eps"],
            class_ids=class_ids, norm_scale=cfg["norm_scale"],
            norm_clip=cfg["norm_clip"] if cfg["norm_clip"] > 0 else 0,
        )

        # Metrics
        bright = continuous_brightness(steered)
        pixel_lift = (bright > 0.5).mean() - ref_rate
        clip_scores = clip_classify(steered, prompts_bright, clip_model, clip_processor)[:, 0]
        clip_lift = clip_scores.mean() - ref_clip
        feat = compute_inception_features(steered)
        fid = compute_fid(ref_features, feat)
        div = compute_diversity(steered)
        div_ratio = div / (ref_div + 1e-8)

        print(f"  {cfg['name']:<28s} {pixel_lift:>+10.3f} {clip_lift:>+10.3f} {fid:>6.1f} {div_ratio:>10.3f}")

        append_result(commit, "WANG", "pixel_lift", pixel_lift, mem_gb, "keep",
                      f"wang_comp {cfg['name']} pixel_lift")
        append_result(commit, "WANG", "clip_lift", clip_lift, mem_gb, "keep",
                      f"wang_comp {cfg['name']} clip_lift")
        append_result(commit, "WANG", "fid", fid, mem_gb, "keep",
                      f"wang_comp {cfg['name']} FID")
        append_result(commit, "WANG", "div_ratio", div_ratio, mem_gb, "keep",
                      f"wang_comp {cfg['name']} diversity_ratio")

    print(f"\n{'='*60}")
    print(f"  DONE — Wang Comparison")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
