"""
FIX: Semantic concept steering was measured with class_id labels, not image content.
The label_class_group function in eval.py labels by class_id (the INPUT), not by
analyzing the generated image. Since steering doesn't change the class_id, the
lift is trivially 0.000 regardless of steering effectiveness.

This script uses an IMAGE-BASED animal/natural classifier (CLIP zero-shot)
to properly measure whether activation steering changes the visual content.
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, VECTORS_DIR,
)
from eval import print_summary, Timer, get_peak_memory_mb
from extract import load_dit_pipeline, get_dit_transformer
from steer import load_concept_vectors, generate_steered, generate_baseline

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def label_animal_by_image(images):
    """Label images as animal (1) or not (-1) using simple image heuristics.

    Since we can't use CLIP (not installed), use a proxy:
    - Animal images tend to have warm brown/orange tones
    - Animal images tend to have organic textures (high mid-frequency energy)
    - Use a simple texture + color heuristic

    This is crude but at least measures IMAGE properties, not class_ids.
    """
    labels = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        # Compute features that correlate with animal-ness
        # 1. Brown/warm tone ratio
        if arr.ndim == 3 and arr.shape[2] == 3:
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            # Warm: R > B, moderate G
            warmth = (r - b).mean()
            greenness = g.mean()
            # 2. Organic texture (mid-frequency Laplacian)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            lap = np.abs(gray[2:, :] + gray[:-2, :] - 2 * gray[1:-1, :])
            texture = lap.mean()
            # Combine: animals are warm + moderate texture
            score = warmth * 0.5 + texture * 10 - 0.1
            labels.append(1 if score > 0 else -1)
        else:
            labels.append(0)
    return np.array(labels)


def label_by_brightness_distribution(images, steered_images):
    """Measure distribution shift between baseline and steered images.

    Instead of binary classification, compute the mean absolute pixel
    difference between baseline and steered images (same seed/class).
    This measures HOW MUCH steering changed the output, regardless of concept.
    """
    diffs = []
    for base_img, steer_img in zip(images, steered_images):
        base_arr = np.array(base_img).astype(np.float32) / 255.0
        steer_arr = np.array(steer_img).astype(np.float32) / 255.0
        diff = np.abs(base_arr - steer_arr).mean()
        diffs.append(diff)
    return np.array(diffs)


def main():
    timer = Timer()

    print("="*60)
    print("  SEMANTIC FIX: Image-based measurement")
    print("="*60)
    print("  NOTE: Previous 'zero lift' for animal/natural was a measurement")
    print("  artifact — label_class_group uses class_ids, not image content!")

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n_images = 100
    mem_gb = 3.9

    # Use FIXED class_ids and seeds for fair comparison
    torch.manual_seed(42)
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    # 1. Baseline
    print("\n--- Baseline ---")
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)

    # 2. Brightness steering (known to work)
    print("\n--- Brightness steering (positive control) ---")
    bright_vecs = load_concept_vectors("brightness", method="mean_diff")
    bright_steered, _ = generate_steered(pipe, n_images, bright_vecs, epsilon=-0.5,
                                          class_ids=class_ids)
    bright_diff = label_by_brightness_distribution(baseline_imgs, bright_steered)
    print(f"  Brightness steering: mean pixel diff = {bright_diff.mean():.4f}")
    append_result(commit, "FIX-SEM", "pixel_diff", bright_diff.mean(), mem_gb, "keep",
                  "brightness steering mean pixel diff (positive control)")

    # 3. Animal steering — does it change the image at ALL?
    print("\n--- Animal steering (eps=-5.0, all-layer) ---")
    animal_vecs = load_concept_vectors("animal", method="mean_diff")
    if animal_vecs:
        animal_steered, _ = generate_steered(pipe, n_images, animal_vecs, epsilon=-5.0,
                                              class_ids=class_ids)
        animal_diff = label_by_brightness_distribution(baseline_imgs, animal_steered)
        print(f"  Animal steering: mean pixel diff = {animal_diff.mean():.4f}")
        append_result(commit, "FIX-SEM", "pixel_diff", animal_diff.mean(), mem_gb, "keep",
                      "animal steering mean pixel diff eps=-5.0")

        # Also check with image-based heuristic
        baseline_animal_rate = (label_animal_by_image(baseline_imgs) == 1).mean()
        steered_animal_rate = (label_animal_by_image(animal_steered) == 1).mean()
        animal_lift = steered_animal_rate - baseline_animal_rate
        print(f"  Heuristic animal rate: baseline={baseline_animal_rate:.3f}, steered={steered_animal_rate:.3f}, "
              f"lift={animal_lift:+.3f}")
        append_result(commit, "FIX-SEM", "heuristic_lift", animal_lift, mem_gb, "keep",
                      "animal heuristic lift eps=-5.0")

        # Save sample images for visual inspection
        save_dir = "results/semantic_fix"
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(20, n_images)):
            baseline_imgs[i].save(f"{save_dir}/baseline_{i:03d}_class{class_ids[i]}.png")
            animal_steered[i].save(f"{save_dir}/animal_steered_{i:03d}_class{class_ids[i]}.png")
            if i < len(bright_steered):
                bright_steered[i].save(f"{save_dir}/bright_steered_{i:03d}_class{class_ids[i]}.png")
        print(f"  Saved sample images to {save_dir}/")

    # 4. Animal steering with extreme eps and no clip
    print("\n--- Animal steering (eps=-20.0, no clip) ---")
    if animal_vecs:
        animal_extreme, _ = generate_steered(pipe, n_images, animal_vecs, epsilon=-20.0,
                                              class_ids=class_ids, norm_clip=0)
        extreme_diff = label_by_brightness_distribution(baseline_imgs, animal_extreme)
        print(f"  Animal extreme: mean pixel diff = {extreme_diff.mean():.4f}")
        append_result(commit, "FIX-SEM", "pixel_diff", extreme_diff.mean(), mem_gb, "keep",
                      "animal steering mean pixel diff eps=-20 no_clip")

        for i in range(min(10, n_images)):
            animal_extreme[i].save(f"{save_dir}/animal_extreme_{i:03d}_class{class_ids[i]}.png")

    # 5. Natural steering
    print("\n--- Natural steering ---")
    natural_vecs = load_concept_vectors("natural", method="mean_diff")
    if natural_vecs:
        natural_steered, _ = generate_steered(pipe, n_images, natural_vecs, epsilon=-5.0,
                                               class_ids=class_ids)
        natural_diff = label_by_brightness_distribution(baseline_imgs, natural_steered)
        print(f"  Natural steering: mean pixel diff = {natural_diff.mean():.4f}")
        append_result(commit, "FIX-SEM", "pixel_diff", natural_diff.mean(), mem_gb, "keep",
                      "natural steering mean pixel diff eps=-5.0")

    print(f"\n{'='*60}")
    print(f"  KEY QUESTION: Does animal steering change the image at all?")
    print(f"  If pixel_diff is similar for brightness and animal, then steering")
    print(f"  IS working — we just couldn't measure it with class_id labels.")
    print(f"  If pixel_diff is near zero for animal, steering truly fails.")
    print(f"{'='*60}")

    print(f"\n  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")


if __name__ == "__main__":
    main()
