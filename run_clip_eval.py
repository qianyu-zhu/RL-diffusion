"""
CLIP-based semantic evaluation of steering.

Uses CLIP zero-shot classification to measure whether activation steering
changes the semantic content of generated images (not just pixel statistics).

This resolves the measurement artifact: label_class_group used class_ids,
not image content. CLIP evaluates the ACTUAL image.
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, VECTORS_DIR,
)
from eval import print_summary, Timer, get_peak_memory_mb
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def load_clip():
    """Load CLIP model for zero-shot classification."""
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    return clip_model, clip_processor


def clip_classify(images, text_prompts, clip_model, clip_processor):
    """Zero-shot CLIP classification. Returns softmax scores per image."""
    all_scores = []
    batch_size = 16

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        inputs = clip_processor(text=text_prompts, images=batch_imgs,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image  # (batch, n_prompts)
            probs = logits.softmax(dim=-1).cpu().numpy()

        all_scores.append(probs)

    return np.concatenate(all_scores, axis=0)  # (n_images, n_prompts)


def main():
    timer = Timer()

    print("="*60)
    print("  CLIP-Based Semantic Evaluation")
    print("="*60)

    pipe = load_dit_pipeline()

    print("Loading CLIP...")
    clip_model, clip_processor = load_clip()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n_images = 100
    torch.manual_seed(42)
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()
    mem_gb = 3.9

    # Define CLIP text prompts for concepts
    concept_prompts = {
        "animal": ["a photo of an animal", "a photo of an object or scene without animals"],
        "natural": ["a photo of nature, plants, or animals", "a photo of man-made objects or vehicles"],
        "bright": ["a bright, well-lit photograph", "a dark, dimly-lit photograph"],
        "colorful": ["a colorful, vibrant photograph", "a grey, desaturated photograph"],
        "warm": ["a photograph with warm red and orange tones", "a photograph with cool blue tones"],
    }

    # 1. Baseline
    print("\n--- Baseline ---")
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)

    baseline_scores = {}
    for concept, prompts in concept_prompts.items():
        scores = clip_classify(baseline_imgs, prompts, clip_model, clip_processor)
        baseline_scores[concept] = scores[:, 0].mean()  # mean P(positive prompt)
        print(f"  {concept}: CLIP P(positive) = {baseline_scores[concept]:.3f}")

    # 2. Steering experiments
    configs = [
        ("brightness", "mean_diff", -0.5, None, "bright_all-layer"),
        ("brightness", "mean_diff", -0.5, [0], "bright_L0"),
        ("animal", "mean_diff", -5.0, None, "animal_all-layer_eps5"),
        ("animal", "mean_diff", -1.0, None, "animal_all-layer_eps1"),
        ("animal", "pca", -5.0, None, "animal_pca_eps5"),
        ("natural", "mean_diff", -5.0, None, "natural_all-layer_eps5"),
        ("colorfulness", "mean_diff", -0.5, None, "color_all-layer"),
        ("warmth", "mean_diff", -0.5, None, "warmth_all-layer"),
    ]

    for concept, method, eps, layers, desc in configs:
        print(f"\n--- {desc} ---")
        vectors = load_concept_vectors(concept, method=method, layers=layers)
        if not vectors:
            print(f"  No vectors for {concept}/{method}")
            continue

        steered_imgs, _ = generate_steered(pipe, n_images, vectors, epsilon=eps,
                                            class_ids=class_ids)

        for eval_concept, prompts in concept_prompts.items():
            scores = clip_classify(steered_imgs, prompts, clip_model, clip_processor)
            steered_score = scores[:, 0].mean()
            lift = steered_score - baseline_scores[eval_concept]

            if eval_concept == concept or (concept in ["brightness"] and eval_concept == "bright"):
                # Primary metric
                status = "keep"
                print(f"  CLIP {eval_concept}: {steered_score:.3f} (baseline {baseline_scores[eval_concept]:.3f}, "
                      f"lift {lift:+.3f})")
                append_result(commit, "CLIP", "clip_lift", lift, mem_gb, status,
                              f"CLIP {eval_concept} lift from {desc}")
            elif abs(lift) > 0.02:
                # Side effect
                print(f"  [side] CLIP {eval_concept}: lift {lift:+.3f}")

    print(f"\n{'='*60}")
    print(f"  DONE — CLIP Evaluation")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
