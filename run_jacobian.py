"""
Theory Validation: Jacobian Attenuation Measurement

Empirically measure the Jacobian attenuation factor α that appears in the
Coherent Accumulation equation:

    Lift(0..K) ∝ Σ_{ℓ=0}^{K} S(ℓ) × α^ℓ

The theory predicts α ≈ 0.773 (fitted from cumulative layer data).

Method: For a perturbation δa_ℓ at layer ℓ, measure ||δa_{ℓ+1}|| / ||δa_ℓ||
— the ratio of perturbation norms between consecutive layers. If the theory
is correct, this ratio should be approximately α.

We use Jacobian-vector products (JVPs) via finite differences:
    J_{ℓ→ℓ+1} v ≈ (f(a_ℓ + ε·v) - f(a_ℓ)) / ε

where f is the transformer block at layer ℓ+1.
"""
import os
import torch
import numpy as np
from pathlib import Path
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


def measure_perturbation_propagation(pipe, concept_name="brightness", method="mean_diff",
                                      n_images=50, eps=0.01):
    """Measure how a perturbation at layer ℓ propagates to layer ℓ+1.

    For each image:
    1. Run normal forward pass, record all layer activations
    2. Inject concept vector perturbation at layer ℓ
    3. Record perturbed activations at all subsequent layers
    4. Compute ||δa_{ℓ+k}|| / ||δa_ℓ|| for each k
    """
    transformer = get_dit_transformer(pipe)
    blocks = transformer.transformer_blocks
    n_layers = len(blocks)

    vectors = load_concept_vectors(concept_name, method=method)
    if not vectors:
        print("  No vectors found!")
        return {}

    # We'll measure propagation from each source layer
    results = {}  # source_layer -> {target_layer: mean_ratio}

    source_layers = [0, 5, 10, 15, 20, 25]

    for source_layer in source_layers:
        if source_layer not in vectors:
            continue

        v = vectors[source_layer].to(DEVICE, DTYPE)
        v_norm = v.norm()

        # Storage for perturbation norms at each target layer
        all_ratios = {tgt: [] for tgt in range(source_layer, n_layers)}

        for img_idx in tqdm(range(n_images), desc=f"Source L{source_layer}"):
            class_id = torch.randint(0, NUM_CLASSES, (1,)).tolist()

            # --- Clean forward pass ---
            clean_acts = {}  # layer -> activation tensor

            clean_hooks = []
            for layer_idx in range(n_layers):
                def make_clean_hook(idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                        else:
                            h = output
                        # Take conditional half
                        bs = h.shape[0]
                        if bs > 1 and bs % 2 == 0:
                            h = h[:bs // 2]
                        clean_acts[idx] = h.detach().clone()
                    return hook_fn
                clean_hooks.append(blocks[layer_idx].register_forward_hook(make_clean_hook(layer_idx)))

            with torch.no_grad():
                pipe(class_id, num_inference_steps=NUM_INFERENCE_STEPS,
                     guidance_scale=GUIDANCE_SCALE, output_type="pil")

            for h in clean_hooks:
                h.remove()

            # --- Perturbed forward pass ---
            perturbed_acts = {}

            perturb_hooks = []
            for layer_idx in range(n_layers):
                def make_perturb_hook(idx, src, vec, epsilon):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            rest = output[1:]
                        else:
                            h = output
                            rest = None

                        bs = h.shape[0]
                        if bs > 1 and bs % 2 == 0:
                            h_cond = h[:bs // 2]
                        else:
                            h_cond = h

                        # Inject perturbation at source layer
                        if idx == src:
                            h_cond = h_cond + epsilon * vec.unsqueeze(0).unsqueeze(0)
                            if bs > 1 and bs % 2 == 0:
                                h = torch.cat([h_cond, h[bs // 2:]], dim=0)
                            else:
                                h = h_cond

                        # Record activation (conditional half only)
                        perturbed_acts[idx] = h_cond.detach().clone()

                        if rest is not None:
                            return (h,) + rest
                        return h
                    return hook_fn
                perturb_hooks.append(blocks[layer_idx].register_forward_hook(
                    make_perturb_hook(layer_idx, source_layer, v, eps)
                ))

            with torch.no_grad():
                pipe(class_id, num_inference_steps=NUM_INFERENCE_STEPS,
                     guidance_scale=GUIDANCE_SCALE, output_type="pil")

            for h in perturb_hooks:
                h.remove()

            # Compute perturbation norms
            delta_source = (perturbed_acts[source_layer] - clean_acts[source_layer]).norm().item()
            if delta_source < 1e-10:
                continue

            for tgt_layer in range(source_layer, n_layers):
                if tgt_layer in clean_acts and tgt_layer in perturbed_acts:
                    delta = (perturbed_acts[tgt_layer] - clean_acts[tgt_layer]).norm().item()
                    ratio = delta / (delta_source + 1e-10)
                    all_ratios[tgt_layer].append(ratio)

        # Compute means
        results[source_layer] = {}
        for tgt_layer in range(source_layer, n_layers):
            if all_ratios[tgt_layer]:
                results[source_layer][tgt_layer] = np.mean(all_ratios[tgt_layer])

    return results


def main():
    timer = Timer()

    print("="*60)
    print("  Theory Validation: Jacobian Attenuation Measurement")
    print("="*60)

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    try:
        results = measure_perturbation_propagation(pipe, n_images=30, eps=0.01)

        print("\n" + "="*60)
        print("  Perturbation Propagation Results")
        print("="*60)

        # For each source layer, fit the decay rate
        all_alphas = []
        for src_layer in sorted(results.keys()):
            tgt_data = results[src_layer]
            print(f"\n  Source Layer {src_layer}:")
            print(f"    {'Target':>8s}  {'Ratio':>8s}  {'Distance':>8s}  {'Per-layer α':>12s}")

            prev_ratio = 1.0
            for tgt_layer in sorted(tgt_data.keys()):
                ratio = tgt_data[tgt_layer]
                distance = tgt_layer - src_layer
                if distance > 0:
                    alpha = ratio ** (1.0 / distance)
                    per_layer = ratio / prev_ratio if prev_ratio > 0 else 0
                    print(f"    L{tgt_layer:2d}       {ratio:8.4f}  {distance:8d}  {alpha:12.4f}")
                    all_alphas.append(alpha)
                prev_ratio = ratio

            # Log the decay from source to source+5
            if src_layer + 5 in tgt_data:
                alpha_5 = tgt_data[src_layer + 5] ** (1.0 / 5)
                append_result(commit, "THEORY", "jacobian_alpha", alpha_5, 3.9, "keep",
                              f"attenuation L{src_layer}->L{src_layer+5} alpha={alpha_5:.4f}")

        if all_alphas:
            mean_alpha = np.mean(all_alphas)
            print(f"\n  Mean per-layer attenuation: α = {mean_alpha:.4f}")
            print(f"  Theory prediction:          α = 0.773")
            print(f"  Difference:                 {abs(mean_alpha - 0.773):.4f}")

            append_result(commit, "THEORY", "mean_alpha", mean_alpha, 3.9, "keep",
                          f"mean Jacobian attenuation factor (theory predicts 0.773)")

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — Jacobian Attenuation Measurement")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
