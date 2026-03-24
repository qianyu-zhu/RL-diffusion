"""
adaLN Modulation Survival Factor Measurement.

Measures how much of a concept perturbation survives the adaLN modulation
at each layer, for different concepts and class conditions.

The survival factor rho_l(v, c) = v^T diag(1+gamma_l(c)) v / ||v||^2
measures whether the adaLN scale parameters compress or preserve the
concept direction v at layer l, given conditioning c.

Theory prediction:
- brightness vectors: rho ≈ 1 (adaLN doesn't control brightness)
- animal vectors: rho << 1 (adaLN actively shapes class-relevant dims)
"""
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, VECTORS_DIR,
)
from eval import print_summary, Timer, get_peak_memory_mb
from extract import load_dit_pipeline, get_dit_transformer
from steer import load_concept_vectors

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def measure_adaln_survival(pipe, commit):
    """Measure adaLN survival factor for concept vectors under different class conditions."""
    print("\n" + "="*60)
    print("  adaLN Modulation Survival Factor")
    print("="*60)

    transformer = get_dit_transformer(pipe)
    blocks = transformer.transformer_blocks

    # Hook to capture adaLN scale parameters (gamma)
    adaln_scales = {}  # {layer_idx: (batch, hidden_dim)}

    def make_scale_hook(layer_idx):
        """Hook into the adaLN modulation to capture scale_mlp (gamma for MLP sub-block).

        DiT block structure:
            norm1 produces: (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 5:
                # output = (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)
                scale_mlp = output[3]  # (batch, hidden_dim)
                # Take conditional half if CFG
                bs = scale_mlp.shape[0]
                if bs > 1 and bs % 2 == 0:
                    scale_mlp = scale_mlp[:bs // 2]
                adaln_scales[layer_idx] = scale_mlp.detach().cpu().float()
        return hook_fn

    # Register hooks on norm1 of each block
    hooks = []
    for idx, block in enumerate(blocks):
        h = block.norm1.register_forward_hook(make_scale_hook(idx))
        hooks.append(h)

    # Generate images to capture adaLN parameters
    n_images = 50
    # Use a mix of animal and non-animal classes
    animal_classes = [1, 130, 207, 281, 388]  # goldfish, flamingo, golden_retriever, tabby_cat, panda
    object_classes = [417, 561, 928, 985, 971]  # balloon, forklift, ice_cream, daisy, bubble

    concepts_to_test = ["brightness", "warmth", "colorfulness", "animal", "natural"]
    mem_gb = 3.9

    # Collect adaLN parameters for animal vs non-animal classes
    all_scales = defaultdict(list)  # layer -> list of (hidden_dim,) tensors

    for class_set, set_name in [(animal_classes, "animal"), (object_classes, "object")]:
        for cid in class_set:
            adaln_scales.clear()
            cids = [cid] * 10
            with torch.no_grad():
                pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                     guidance_scale=GUIDANCE_SCALE, output_type="pil")

            for layer_idx in range(NUM_LAYERS):
                if layer_idx in adaln_scales:
                    # Average over batch and use the last denoising step's values
                    scale = adaln_scales[layer_idx].mean(dim=0)  # (hidden_dim,)
                    all_scales[layer_idx].append(scale)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute survival factor for each concept vector
    print(f"\n  {'Concept':<15s} {'Layer':>5s} {'Mean γ':>8s} {'γ std':>8s} {'ρ(v,γ)':>8s} {'Interpretation'}")
    print("  " + "-" * 70)

    for concept in concepts_to_test:
        for layer_idx in [0, 5, 10, 15, 20, 25]:
            vectors = load_concept_vectors(concept, method="mean_diff", layers=[layer_idx])
            if layer_idx not in vectors or layer_idx not in all_scales:
                continue

            v = vectors[layer_idx].float()
            v = v / (v.norm() + 1e-8)

            # Average scale across all collected samples
            scales_list = all_scales[layer_idx]
            if not scales_list:
                continue
            avg_scale = torch.stack(scales_list).mean(dim=0)  # (hidden_dim,)

            # Survival factor: how much of v survives modulation by (1+gamma)
            modulation = 1.0 + avg_scale  # (hidden_dim,)
            v_modulated = v * modulation
            rho = (v_modulated.norm() / (v.norm() * modulation.norm() + 1e-8)).item()

            # Also compute: how much does (1+gamma) compress/expand in the v direction
            # vs orthogonal directions
            proj_scale = torch.dot(v, (modulation * v))  # scalar
            mean_gamma = avg_scale.mean().item()
            std_gamma = avg_scale.std().item()

            interpretation = "preserved" if rho > 0.5 else "compressed"

            print(f"  {concept:<15s} L{layer_idx:>3d} {mean_gamma:>+8.4f} {std_gamma:>8.4f} {rho:>8.4f}  {interpretation}")

            append_result(commit, "FIX-ADALN", "survival_rho", rho, mem_gb, "keep",
                          f"adaln_survival {concept} mean_diff layer{layer_idx}")

    # Compute the key comparison: survival for brightness vs animal at each layer
    print("\n  --- Brightness vs Animal survival comparison ---")
    for layer_idx in [0, 5, 10, 15, 20, 25]:
        bright_vecs = load_concept_vectors("brightness", method="mean_diff", layers=[layer_idx])
        animal_vecs = load_concept_vectors("animal", method="mean_diff", layers=[layer_idx])

        if layer_idx not in bright_vecs or layer_idx not in animal_vecs or layer_idx not in all_scales:
            continue

        scales_list = all_scales[layer_idx]
        avg_scale = torch.stack(scales_list).mean(dim=0)
        modulation = 1.0 + avg_scale

        v_bright = bright_vecs[layer_idx].float()
        v_bright = v_bright / (v_bright.norm() + 1e-8)
        v_animal = animal_vecs[layer_idx].float()
        v_animal = v_animal / (v_animal.norm() + 1e-8)

        rho_bright = (v_bright * modulation).norm().item() / (v_bright.norm() * modulation.norm() + 1e-8).item()
        rho_animal = (v_animal * modulation).norm().item() / (v_animal.norm() * modulation.norm() + 1e-8).item()

        print(f"  L{layer_idx}: ρ(brightness)={rho_bright:.4f}, ρ(animal)={rho_animal:.4f}, "
              f"ratio={rho_bright/(rho_animal+1e-8):.2f}x")


def measure_cfg_contribution(pipe, commit):
    """Test if CFG is responsible for suppressing class steering.

    Run brightness and animal steering with CFG=1.0 (no guidance amplification).
    """
    print("\n" + "="*60)
    print("  CFG Contribution Test (CFG=1.0 vs 4.0)")
    print("="*60)

    from steer import generate_steered, generate_baseline
    from run_stage7 import continuous_brightness

    n_images = 100
    mem_gb = 3.9

    for cfg_scale in [1.0, 4.0]:
        class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

        # Temporarily override guidance scale
        import config
        original_cfg = config.GUIDANCE_SCALE
        config.GUIDANCE_SCALE = cfg_scale

        # Baseline
        baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
        baseline_bright = continuous_brightness(baseline_imgs)
        baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()

        # Brightness steering
        vectors = load_concept_vectors("brightness", method="mean_diff")
        steered_imgs, _ = generate_steered(pipe, n_images, vectors, epsilon=-0.5,
                                            class_ids=class_ids)
        steered_bright = continuous_brightness(steered_imgs)
        steered_rate = (np.array([1 if b > 0.5 else -1 for b in steered_bright]) == 1).mean()
        bright_lift = steered_rate - baseline_rate

        print(f"  CFG={cfg_scale}: brightness lift={bright_lift:+.3f}")
        append_result(commit, "FIX-CFG", "bright_lift", bright_lift, mem_gb, "keep",
                      f"cfg_test brightness CFG={cfg_scale}")

        # Animal steering
        animal_vecs = load_concept_vectors("animal", method="mean_diff")
        if animal_vecs:
            from eval import get_concept_labels, CONCEPTS
            animal_steered, _ = generate_steered(pipe, n_images, animal_vecs, epsilon=-5.0,
                                                  class_ids=class_ids)
            animal_labels_base = get_concept_labels(CONCEPTS["animal"], baseline_imgs, class_ids)
            animal_labels_steer = get_concept_labels(CONCEPTS["animal"], animal_steered, class_ids)
            base_rate = (animal_labels_base == 1).mean()
            steer_rate = (animal_labels_steer == 1).mean()
            animal_lift = steer_rate - base_rate

            print(f"  CFG={cfg_scale}: animal lift={animal_lift:+.3f}")
            append_result(commit, "FIX-CFG", "animal_lift", animal_lift, mem_gb, "keep",
                          f"cfg_test animal CFG={cfg_scale} eps=-5.0")

        # Restore
        config.GUIDANCE_SCALE = original_cfg


def main():
    timer = Timer()

    print("="*60)
    print("  adaLN Survival + CFG Contribution Tests")
    print("="*60)

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    try:
        measure_adaln_survival(pipe, commit)
        measure_cfg_contribution(pipe, commit)
    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — adaLN Survival Experiments")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
