"""
Generate teaser figure images: baseline vs steered pairs for paper Figure 1.
Uses correct eps signs and moderate steering for visual quality.
"""
import os
import torch
import numpy as np
from PIL import Image

from config import MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline

def main():
    os.makedirs('figures/teaser', exist_ok=True)

    pipe = load_dit_pipeline()

    # Fixed seeds for reproducibility
    n = 8
    torch.manual_seed(123)
    class_ids = torch.randint(0, NUM_CLASSES, (n,)).tolist()

    # Baseline
    print("Generating baseline...")
    base_imgs, _ = generate_baseline(pipe, n, class_ids=class_ids)
    for i, img in enumerate(base_imgs):
        img.save(f'figures/teaser/baseline_{i:02d}_c{class_ids[i]}.png')

    # Brightness (eps=-0.3, moderate)
    print("Brightness steering...")
    vecs = load_concept_vectors("brightness", method="mean_diff")
    imgs, _ = generate_steered(pipe, n, vecs, epsilon=-0.3, class_ids=class_ids)
    for i, img in enumerate(imgs):
        img.save(f'figures/teaser/bright_{i:02d}.png')

    # Animal (eps=+5, correct sign)
    print("Animal steering...")
    vecs = load_concept_vectors("animal", method="mean_diff")
    if vecs:
        imgs, _ = generate_steered(pipe, n, vecs, epsilon=+5.0, class_ids=class_ids)
        for i, img in enumerate(imgs):
            img.save(f'figures/teaser/animal_{i:02d}.png')

    # Natural (eps=-3, moderate)
    print("Natural steering...")
    vecs = load_concept_vectors("natural", method="mean_diff")
    if vecs:
        imgs, _ = generate_steered(pipe, n, vecs, epsilon=-3.0, class_ids=class_ids)
        for i, img in enumerate(imgs):
            img.save(f'figures/teaser/natural_{i:02d}.png')

    # Warmth (eps=-0.3)
    print("Warmth steering...")
    vecs = load_concept_vectors("warmth", method="mean_diff")
    if vecs:
        imgs, _ = generate_steered(pipe, n, vecs, epsilon=-0.3, class_ids=class_ids)
        for i, img in enumerate(imgs):
            img.save(f'figures/teaser/warm_{i:02d}.png')

    # Composition: brightness + warmth
    print("Composition steering...")
    bright_vecs = load_concept_vectors("brightness", method="mean_diff")
    warm_vecs = load_concept_vectors("warmth", method="mean_diff")
    if bright_vecs and warm_vecs:
        combined = {}
        for l in bright_vecs:
            combined[l] = bright_vecs[l]
            if l in warm_vecs:
                combined[l] = (bright_vecs[l] + warm_vecs[l]) / 2
                combined[l] = combined[l] / combined[l].norm()
        imgs, _ = generate_steered(pipe, n, combined, epsilon=-0.3, class_ids=class_ids)
        for i, img in enumerate(imgs):
            img.save(f'figures/teaser/compose_{i:02d}.png')

    print(f"Saved {6*n} images to figures/teaser/")
    print("Done!")


if __name__ == "__main__":
    main()
