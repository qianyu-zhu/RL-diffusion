"""
CLIP v2: Test sign flips and more epsilon values for semantic steering.
Previous results showed animal mean-diff at eps=-5 gives -0.028 (wrong sign).
Test both positive and negative eps to find the right direction.
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
from eval import Timer, get_peak_memory_mb
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline
from run_clip_eval import load_clip, clip_classify, append_result

RESULTS_FILE = "results.tsv"


def main():
    timer = Timer()

    print("="*60)
    print("  CLIP v2: Sign Flips + More Eps")
    print("="*60)

    pipe = load_dit_pipeline()
    clip_model, clip_processor = load_clip()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n_images = 100
    torch.manual_seed(42)
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()
    mem_gb = 3.9

    # Baseline CLIP scores
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)

    prompts_animal = ["a photo of an animal", "a photo of an object or scene without animals"]
    prompts_natural = ["a photo of nature, plants, or animals", "a photo of man-made objects or vehicles"]
    prompts_bright = ["a bright, well-lit photograph", "a dark, dimly-lit photograph"]

    baseline_animal = clip_classify(baseline_imgs, prompts_animal, clip_model, clip_processor)[:, 0].mean()
    baseline_natural = clip_classify(baseline_imgs, prompts_natural, clip_model, clip_processor)[:, 0].mean()
    baseline_bright = clip_classify(baseline_imgs, prompts_bright, clip_model, clip_processor)[:, 0].mean()

    print(f"  Baseline: animal={baseline_animal:.3f}, natural={baseline_natural:.3f}, bright={baseline_bright:.3f}")

    # Test configurations: both eps signs
    configs = [
        # Animal
        ("animal", "mean_diff", -1.0, None, "animal MD eps=-1"),
        ("animal", "mean_diff", +1.0, None, "animal MD eps=+1"),
        ("animal", "mean_diff", -5.0, None, "animal MD eps=-5"),
        ("animal", "mean_diff", +5.0, None, "animal MD eps=+5"),
        ("animal", "mean_diff", -10.0, None, "animal MD eps=-10"),
        ("animal", "mean_diff", +10.0, None, "animal MD eps=+10"),
        ("animal", "pca", -5.0, None, "animal PCA eps=-5"),
        ("animal", "pca", +5.0, None, "animal PCA eps=+5"),
        # Natural
        ("natural", "mean_diff", -5.0, None, "natural MD eps=-5"),
        ("natural", "mean_diff", +5.0, None, "natural MD eps=+5"),
        ("natural", "mean_diff", -1.0, None, "natural MD eps=-1"),
        ("natural", "mean_diff", +1.0, None, "natural MD eps=+1"),
        # Brightness with CLIP (compare to pixel-based)
        ("brightness", "mean_diff", -0.5, None, "bright MD eps=-0.5"),
        ("brightness", "mean_diff", +0.5, None, "bright MD eps=+0.5"),
        ("brightness", "mean_diff", -0.1, None, "bright MD eps=-0.1"),
    ]

    for concept, method, eps, layers, desc in configs:
        vectors = load_concept_vectors(concept, method=method, layers=layers)
        if not vectors:
            continue

        steered_imgs, _ = generate_steered(pipe, n_images, vectors, epsilon=eps,
                                            class_ids=class_ids)

        # Evaluate with relevant CLIP prompts
        if "animal" in concept:
            scores = clip_classify(steered_imgs, prompts_animal, clip_model, clip_processor)
            lift = scores[:, 0].mean() - baseline_animal
            print(f"  {desc}: CLIP animal={scores[:, 0].mean():.3f}, lift={lift:+.3f}")
            append_result(commit, "CLIPv2", "clip_animal", lift, mem_gb, "keep", f"CLIPv2 {desc}")
        elif "natural" in concept:
            scores = clip_classify(steered_imgs, prompts_natural, clip_model, clip_processor)
            lift = scores[:, 0].mean() - baseline_natural
            print(f"  {desc}: CLIP natural={scores[:, 0].mean():.3f}, lift={lift:+.3f}")
            append_result(commit, "CLIPv2", "clip_natural", lift, mem_gb, "keep", f"CLIPv2 {desc}")
        elif "bright" in concept:
            scores = clip_classify(steered_imgs, prompts_bright, clip_model, clip_processor)
            lift = scores[:, 0].mean() - baseline_bright
            print(f"  {desc}: CLIP bright={scores[:, 0].mean():.3f}, lift={lift:+.3f}")
            append_result(commit, "CLIPv2", "clip_bright", lift, mem_gb, "keep", f"CLIPv2 {desc}")

    print(f"\n{'='*60}")
    print(f"  DONE — CLIP v2")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
