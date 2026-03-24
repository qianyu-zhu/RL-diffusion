"""Teaser v2: lower eps for semantic concepts, more seeds for variety."""
import os, torch, numpy as np
from config import MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline

def main():
    os.makedirs('figures/teaser_v2', exist_ok=True)
    pipe = load_dit_pipeline()

    n = 10
    torch.manual_seed(777)
    class_ids = torch.randint(0, NUM_CLASSES, (n,)).tolist()

    configs = [
        ("baseline", None, None, 0),
        ("bright", "brightness", "mean_diff", -0.3),
        ("warm", "warmth", "mean_diff", -0.3),
        ("compose_bw", None, None, -0.3),  # brightness + warmth
        ("animal_mild", "animal", "mean_diff", +2.0),
        ("animal_mod", "animal", "mean_diff", +3.0),
        ("natural_mild", "natural", "mean_diff", -2.0),
        ("natural_mod", "natural", "mean_diff", -3.0),
    ]

    print("Baseline...")
    base, _ = generate_baseline(pipe, n, class_ids=class_ids)
    for i, img in enumerate(base):
        img.save(f'figures/teaser_v2/baseline_{i:02d}_c{class_ids[i]}.png')

    for name, concept, method, eps in configs:
        if name == "baseline":
            continue

        if name.startswith("compose"):
            bv = load_concept_vectors("brightness", method="mean_diff")
            wv = load_concept_vectors("warmth", method="mean_diff")
            vecs = {}
            for l in bv:
                v = bv[l].clone()
                if l in wv:
                    v = v + wv[l]
                    v = v / v.norm()
                vecs[l] = v
        else:
            vecs = load_concept_vectors(concept, method=method)

        if not vecs:
            print(f"  {name}: no vectors, skip")
            continue

        print(f"{name} eps={eps}...")
        imgs, _ = generate_steered(pipe, n, vecs, epsilon=eps, class_ids=class_ids)
        for i, img in enumerate(imgs):
            img.save(f'figures/teaser_v2/{name}_{i:02d}.png')

    print(f"Saved to figures/teaser_v2/")

if __name__ == "__main__":
    main()
