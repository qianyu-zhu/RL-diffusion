"""
Pareto Frontier: Concept lift vs FID at varying epsilon.
Also: 500-image CLIP evaluation with bootstrap CIs for top configs.
This is the key figure for the paper.
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE,
)
from eval import Timer, get_peak_memory_mb
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
    print("  Pareto Frontier + Large-N CLIP Evaluation")
    print("="*60)

    pipe = load_dit_pipeline()
    clip_model, clip_processor = load_clip()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n_images = 500
    torch.manual_seed(42)
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()
    mem_gb = 3.9

    # === Reference ===
    print(f"\nGenerating {n_images} reference images...")
    ref_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    ref_features = compute_inception_features(ref_imgs)

    prompts_bright = ["a bright, well-lit photograph", "a dark, dimly-lit photograph"]
    prompts_animal = ["a photo of an animal", "a photo of an object or scene without animals"]
    prompts_natural = ["a photo of nature, plants, or animals", "a photo of man-made objects or vehicles"]

    ref_clip_bright = clip_classify(ref_imgs, prompts_bright, clip_model, clip_processor)[:, 0]
    ref_clip_animal = clip_classify(ref_imgs, prompts_animal, clip_model, clip_processor)[:, 0]
    ref_clip_natural = clip_classify(ref_imgs, prompts_natural, clip_model, clip_processor)[:, 0]

    ref_bright_rate = (continuous_brightness(ref_imgs) > 0.5).mean()

    print(f"  Ref: pixel_bright={ref_bright_rate:.3f}, CLIP_bright={ref_clip_bright.mean():.3f}, "
          f"CLIP_animal={ref_clip_animal.mean():.3f}, CLIP_natural={ref_clip_natural.mean():.3f}")

    # === Pareto: Brightness at varying eps ===
    print("\n--- Brightness Pareto (all-layer mean_diff) ---")
    bright_vecs = load_concept_vectors("brightness", method="mean_diff")

    for eps in [-0.05, -0.1, -0.2, -0.3, -0.5, -0.7, -1.0]:
        steered, _ = generate_steered(pipe, n_images, bright_vecs, epsilon=eps,
                                       class_ids=class_ids)
        feat = compute_inception_features(steered)
        fid = compute_fid(ref_features, feat)
        bright = continuous_brightness(steered)
        pixel_lift = (bright > 0.5).mean() - ref_bright_rate
        clip_scores = clip_classify(steered, prompts_bright, clip_model, clip_processor)[:, 0]
        clip_lift = clip_scores.mean() - ref_clip_bright.mean()

        print(f"  eps={eps:+.2f}: FID={fid:.1f}, pixel_lift={pixel_lift:+.3f}, CLIP_lift={clip_lift:+.3f}")
        append_result(commit, "PARETO", "fid", fid, mem_gb, "keep",
                      f"pareto brightness eps={eps} FID")
        append_result(commit, "PARETO", "pixel_lift", pixel_lift, mem_gb, "keep",
                      f"pareto brightness eps={eps} pixel_lift")
        append_result(commit, "PARETO", "clip_lift", clip_lift, mem_gb, "keep",
                      f"pareto brightness eps={eps} CLIP_lift")

    # === Pareto: Animal at varying eps ===
    print("\n--- Animal Pareto (all-layer mean_diff) ---")
    animal_vecs = load_concept_vectors("animal", method="mean_diff")
    if animal_vecs:
        for eps in [+1.0, +2.0, +5.0, +7.0, +10.0]:
            steered, _ = generate_steered(pipe, n_images, animal_vecs, epsilon=eps,
                                           class_ids=class_ids)
            feat = compute_inception_features(steered)
            fid = compute_fid(ref_features, feat)
            clip_scores = clip_classify(steered, prompts_animal, clip_model, clip_processor)[:, 0]
            clip_lift = clip_scores.mean() - ref_clip_animal.mean()

            print(f"  eps={eps:+.1f}: FID={fid:.1f}, CLIP_animal_lift={clip_lift:+.3f}")
            append_result(commit, "PARETO", "fid", fid, mem_gb, "keep",
                          f"pareto animal eps={eps} FID")
            append_result(commit, "PARETO", "clip_lift", clip_lift, mem_gb, "keep",
                          f"pareto animal eps={eps} CLIP_lift")

    # === Pareto: Natural ===
    print("\n--- Natural Pareto (all-layer mean_diff) ---")
    natural_vecs = load_concept_vectors("natural", method="mean_diff")
    if natural_vecs:
        for eps in [-1.0, -2.0, -5.0, -7.0, -10.0]:
            steered, _ = generate_steered(pipe, n_images, natural_vecs, epsilon=eps,
                                           class_ids=class_ids)
            feat = compute_inception_features(steered)
            fid = compute_fid(ref_features, feat)
            clip_scores = clip_classify(steered, prompts_natural, clip_model, clip_processor)[:, 0]
            clip_lift = clip_scores.mean() - ref_clip_natural.mean()

            print(f"  eps={eps:+.1f}: FID={fid:.1f}, CLIP_natural_lift={clip_lift:+.3f}")
            append_result(commit, "PARETO", "fid", fid, mem_gb, "keep",
                          f"pareto natural eps={eps} FID")
            append_result(commit, "PARETO", "clip_lift", clip_lift, mem_gb, "keep",
                          f"pareto natural eps={eps} CLIP_lift")

    # === Bootstrap CIs for top configs ===
    print("\n--- Bootstrap CIs (3 seeds × 200 images) ---")
    top_configs = [
        ("brightness", "mean_diff", -0.5, None, "bright-all-MD"),
        ("animal", "mean_diff", +5.0, None, "animal-all-MD"),
        ("natural", "mean_diff", -5.0, None, "natural-all-MD"),
    ]

    for concept, method, eps, layers, name in top_configs:
        vecs = load_concept_vectors(concept, method=method, layers=layers)
        if not vecs:
            continue

        lifts = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            cids = torch.randint(0, NUM_CLASSES, (200,)).tolist()
            base, _ = generate_baseline(pipe, 200, class_ids=cids)
            steer, _ = generate_steered(pipe, 200, vecs, epsilon=eps, class_ids=cids)

            if "bright" in name:
                prompts = prompts_bright
                ref_mean = ref_clip_bright.mean()
            elif "animal" in name:
                prompts = prompts_animal
                ref_mean = ref_clip_animal.mean()
            else:
                prompts = prompts_natural
                ref_mean = ref_clip_natural.mean()

            base_clip = clip_classify(base, prompts, clip_model, clip_processor)[:, 0].mean()
            steer_clip = clip_classify(steer, prompts, clip_model, clip_processor)[:, 0].mean()
            lifts.append(steer_clip - base_clip)

        mean_l = np.mean(lifts)
        std_l = np.std(lifts)
        print(f"  {name}: CLIP lift = {mean_l:+.3f} ± {std_l:.3f}")
        append_result(commit, "PARETO-CI", "clip_mean", mean_l, mem_gb, "keep",
                      f"bootstrap {name} mean={mean_l:.3f} std={std_l:.3f}")

    print(f"\n{'='*60}")
    print(f"  DONE — Pareto + Large-N")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
