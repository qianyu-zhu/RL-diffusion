"""
Data efficiency with CLIP evaluation.
How many extraction samples M are needed for effective steering?
Tests M = 10, 25, 50, 100, 200, 500 with CLIP metric.
"""
import os, torch, numpy as np
from config import MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE, CONCEPTS, VECTORS_DIR
from eval import get_concept_labels, Timer, get_peak_memory_mb
from extract import load_dit_pipeline, get_dit_transformer, ActivationCollector, generate_and_collect, extract_concept_vector_mean_diff
from steer import generate_steered, generate_baseline, SteeringHook
from run_clip_eval import load_clip, clip_classify
from pathlib import Path

RESULTS_FILE = "results.tsv"

def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")

def main():
    timer = Timer()
    print("="*60)
    print("  Data Efficiency with CLIP")
    print("="*60)

    pipe = load_dit_pipeline()
    clip_model, clip_processor = load_clip()
    transformer = get_dit_transformer(pipe)

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    # Generate a large pool of images + activations for extraction
    print("\nGenerating 500 images for vector extraction pool...")
    all_layers = list(range(NUM_LAYERS))
    collector = ActivationCollector(transformer, layers=all_layers)
    images, activations, class_ids = generate_and_collect(pipe, 500, collector)
    collector.remove_hooks()

    labels = np.array([1 if np.array(img).mean() / 255.0 > 0.5 else -1 for img in images])
    labels_t = torch.tensor(labels, dtype=torch.float32)
    mask = labels_t != 0

    # Evaluation set
    n_eval = 200
    torch.manual_seed(42)
    eval_cids = torch.randint(0, NUM_CLASSES, (n_eval,)).tolist()
    baseline_imgs, _ = generate_baseline(pipe, n_eval, class_ids=eval_cids)

    prompts = ["a bright, well-lit photograph", "a dark, dimly-lit photograph"]
    baseline_clip = clip_classify(baseline_imgs, prompts, clip_model, clip_processor)[:, 0].mean()
    baseline_bright = np.array([np.array(img).mean() / 255.0 for img in baseline_imgs])
    baseline_rate = (baseline_bright > 0.5).mean()

    print(f"  Baseline: pixel_rate={baseline_rate:.3f}, CLIP={baseline_clip:.3f}")

    mem_gb = 3.9

    # Test each M
    for M in [10, 25, 50, 100, 200, 500]:
        print(f"\n--- M={M} ---")
        # Subsample
        idx = torch.randperm(500)[:M]
        sub_labels = labels_t[idx]
        sub_mask = sub_labels != 0

        # Extract vectors from subsample
        vectors = {}
        for layer_idx in all_layers:
            acts = activations[layer_idx][idx]
            acts_valid = acts[sub_mask]
            labels_valid = sub_labels[sub_mask]
            if (labels_valid == 1).sum() < 3 or (labels_valid == -1).sum() < 3:
                continue
            v = extract_concept_vector_mean_diff(acts_valid, labels_valid)
            vectors[layer_idx] = v

        if not vectors:
            print(f"  Not enough samples, skip")
            continue

        # Steer
        steered, _ = generate_steered(pipe, n_eval, vectors, epsilon=-0.5, class_ids=eval_cids)
        bright = np.array([np.array(img).mean() / 255.0 for img in steered])
        pixel_lift = (bright > 0.5).mean() - baseline_rate
        clip_scores = clip_classify(steered, prompts, clip_model, clip_processor)[:, 0]
        clip_lift = clip_scores.mean() - baseline_clip

        print(f"  M={M}: pixel_lift={pixel_lift:+.3f}, CLIP_lift={clip_lift:+.3f}")
        append_result(commit, "DATA-EFF", "pixel_lift", pixel_lift, mem_gb, "keep",
                      f"data_efficiency M={M} pixel_lift")
        append_result(commit, "DATA-EFF", "clip_lift", clip_lift, mem_gb, "keep",
                      f"data_efficiency M={M} CLIP_lift")

    print(f"\n  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")

if __name__ == "__main__":
    main()
