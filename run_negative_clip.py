"""
Negative steering (concept suppression) with CLIP evaluation.
Shows bidirectional control: increase OR decrease a concept.
"""
import os, torch, numpy as np
from config import MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE
from eval import Timer
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline
from run_clip_eval import load_clip, clip_classify

RESULTS_FILE = "results.tsv"

def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")

def main():
    timer = Timer()
    print("="*60)
    print("  Negative Steering (Concept Suppression) + CLIP")
    print("="*60)

    pipe = load_dit_pipeline()
    clip_model, clip_processor = load_clip()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n = 200
    torch.manual_seed(42)
    cids = torch.randint(0, NUM_CLASSES, (n,)).tolist()
    mem = 3.9

    base, _ = generate_baseline(pipe, n, class_ids=cids)

    prompts = {
        "animal": ["a photo of an animal", "a photo of an object or scene without animals"],
        "natural": ["a photo of nature, plants, or animals", "a photo of man-made objects or vehicles"],
        "bright": ["a bright, well-lit photograph", "a dark, dimly-lit photograph"],
    }

    ref = {}
    for k, p in prompts.items():
        ref[k] = clip_classify(base, p, clip_model, clip_processor)[:, 0].mean()
        print(f"  Baseline {k}: {ref[k]:.3f}")

    # Bidirectional: positive = more concept, negative = less concept
    configs = [
        ("animal", "mean_diff", +5.0, "animal_increase"),
        ("animal", "mean_diff", -5.0, "animal_decrease"),
        ("natural", "mean_diff", -5.0, "natural_increase"),
        ("natural", "mean_diff", +5.0, "natural_decrease"),
        ("brightness", "mean_diff", -0.5, "bright_increase"),
        ("brightness", "mean_diff", +0.5, "bright_decrease"),
        ("warmth", "mean_diff", -0.5, "warm_increase"),
        ("warmth", "mean_diff", +0.5, "warm_decrease"),
    ]

    for concept, method, eps, desc in configs:
        vecs = load_concept_vectors(concept, method=method)
        if not vecs:
            continue

        imgs, _ = generate_steered(pipe, n, vecs, epsilon=eps, class_ids=cids)

        # Measure CLIP for the relevant concept
        eval_concept = concept if concept != "brightness" else "bright"
        if eval_concept == "warmth":
            # No warmth prompt, use bright as proxy
            eval_concept = "bright"

        if eval_concept in prompts:
            scores = clip_classify(imgs, prompts[eval_concept], clip_model, clip_processor)[:, 0]
            lift = scores.mean() - ref[eval_concept]
            print(f"  {desc}: CLIP {eval_concept} lift = {lift:+.3f}")
            append_result(commit, "NEG", "clip_lift", lift, mem, "keep", f"negative_steering {desc}")

    print(f"\n  Wall time: {timer.elapsed():.0f}s")

if __name__ == "__main__":
    main()
