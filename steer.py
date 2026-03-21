"""
LASD steering evaluation pipeline.
This file is modifiable during experiments.
"""
import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    DEFAULT_EPSILON, EPSILON_RANGE, EVAL_N_IMAGES, EVAL_BATCH_SIZE,
    CONCEPTS, VECTORS_DIR,
)
from eval import (
    get_concept_labels, concept_accuracy, compute_diversity,
    print_summary, Timer, get_peak_memory_mb,
)
from extract import load_dit_pipeline, get_dit_transformer


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

class SteeringHook:
    """Inject concept vectors into DiT residual stream during denoising."""

    def __init__(self, transformer, concept_vectors, epsilon=DEFAULT_EPSILON,
                 strategy="A", epsilon_schedule=None):
        """
        Args:
            transformer: DiT transformer module
            concept_vectors: dict of {layer_idx: (d,) tensor}
            epsilon: steering strength (float or dict of {layer: float})
            strategy: "A" (shared vector), "B" (time-binned), "C" (interpolated)
            epsilon_schedule: optional callable(timestep) -> float
        """
        self.transformer = transformer
        self.concept_vectors = concept_vectors
        self.epsilon = epsilon
        self.strategy = strategy
        self.epsilon_schedule = epsilon_schedule
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        blocks = self.transformer.transformer_blocks
        for layer_idx, vec in self.concept_vectors.items():
            if layer_idx < len(blocks):
                hook = blocks[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx, vec)
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_idx, concept_vec):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # Get epsilon for this layer
            eps = self.epsilon if isinstance(self.epsilon, (int, float)) else self.epsilon.get(layer_idx, 0.0)

            # Apply steering: add eps * v to all spatial positions
            v = concept_vec.to(hidden.device, hidden.dtype)
            # hidden: (batch, num_patches, hidden_dim)
            # v: (hidden_dim,) -> broadcast to (1, 1, hidden_dim)
            orig_norm = hidden.norm(dim=-1, keepdim=True)
            hidden = hidden + eps * v.unsqueeze(0).unsqueeze(0)
            # Norm-clipping guardrail (Activation Transport insight):
            # prevent steering from pushing activations OOD
            new_norm = hidden.norm(dim=-1, keepdim=True)
            max_norm = orig_norm * 1.5  # allow 50% norm increase max
            scale = torch.clamp(max_norm / new_norm.clamp(min=1e-8), max=1.0)
            hidden = hidden * scale

            if rest is not None:
                return (hidden,) + rest
            return hidden
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# Generation with steering
# ---------------------------------------------------------------------------

def generate_steered(pipe, n_images, concept_vectors, epsilon=DEFAULT_EPSILON,
                     class_ids=None, strategy="A"):
    """Generate images with concept steering applied."""
    transformer = get_dit_transformer(pipe)

    steering = SteeringHook(transformer, concept_vectors, epsilon=epsilon, strategy=strategy)

    images = []
    used_class_ids = []

    for i in tqdm(range(0, n_images, EVAL_BATCH_SIZE), desc=f"Steering (eps={epsilon})"):
        batch_size = min(EVAL_BATCH_SIZE, n_images - i)

        if class_ids is not None:
            cids = class_ids[i:i + batch_size]
        else:
            cids = torch.randint(0, NUM_CLASSES, (batch_size,)).tolist()

        with torch.no_grad():
            output = pipe(
                cids,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
            )

        images.extend(output.images)
        used_class_ids.extend(cids)

    steering.remove_hooks()
    return images, used_class_ids


def generate_baseline(pipe, n_images, class_ids=None):
    """Generate images without steering (baseline)."""
    images = []
    used_class_ids = []

    for i in tqdm(range(0, n_images, EVAL_BATCH_SIZE), desc="Baseline"):
        batch_size = min(EVAL_BATCH_SIZE, n_images - i)

        if class_ids is not None:
            cids = class_ids[i:i + batch_size]
        else:
            cids = torch.randint(0, NUM_CLASSES, (batch_size,)).tolist()

        with torch.no_grad():
            output = pipe(
                cids,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
            )

        images.extend(output.images)
        used_class_ids.extend(cids)

    return images, used_class_ids


# ---------------------------------------------------------------------------
# Load concept vectors
# ---------------------------------------------------------------------------

METHOD_FILE_MAP = {"rfm": "rfm", "mean_diff": "meandiff", "pca": "pca", "logreg": "logreg"}

def load_concept_vectors(concept_name, method="rfm", layers=None):
    """Load pre-extracted concept vectors from disk."""
    vectors = {}
    vec_dir = Path(VECTORS_DIR)
    file_suffix = METHOD_FILE_MAP.get(method, method)

    for f in vec_dir.glob(f"{concept_name}_layer*_{file_suffix}.pt"):
        data = torch.load(f, map_location="cpu", weights_only=True)
        layer_idx = data.get("layer", int(f.stem.split("layer")[1].split("_")[0]))
        if layers is None or layer_idx in layers:
            vectors[layer_idx] = data["vector"]

    if not vectors:
        print(f"  WARNING: No vectors found for {concept_name}/{method} in {vec_dir}")

    return vectors


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_steering(pipe, concept_name, method="rfm", epsilon=DEFAULT_EPSILON,
                      n_images=EVAL_N_IMAGES, layers=None):
    """Full steering evaluation: steered vs baseline."""

    concept_cfg = CONCEPTS[concept_name]

    # Load concept vectors
    vectors = load_concept_vectors(concept_name, method=method, layers=layers)
    if not vectors:
        return None

    # Use fixed class ids for fair comparison
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    # Generate baseline
    print(f"\nGenerating baseline ({n_images} images)...")
    baseline_images, _ = generate_baseline(pipe, n_images, class_ids=class_ids)

    # Generate steered
    print(f"Generating steered ({n_images} images, eps={epsilon}, method={method})...")
    steered_images, _ = generate_steered(
        pipe, n_images, vectors, epsilon=epsilon, class_ids=class_ids
    )

    # Save sample images for inspection
    import os
    save_dir = os.path.join("results", f"{concept_name}_{method}_eps{epsilon}")
    os.makedirs(save_dir, exist_ok=True)
    for i, (b_img, s_img) in enumerate(zip(baseline_images[:10], steered_images[:10])):
        b_img.save(os.path.join(save_dir, f"baseline_{i:03d}.png"))
        s_img.save(os.path.join(save_dir, f"steered_{i:03d}.png"))
    print(f"  Saved sample images to {save_dir}/")

    # Label both sets
    baseline_labels = get_concept_labels(concept_cfg, baseline_images, class_ids)
    steered_labels = get_concept_labels(concept_cfg, steered_images, class_ids)

    # Metrics
    baseline_positive_rate = (baseline_labels == 1).mean()
    steered_positive_rate = (steered_labels == 1).mean()

    baseline_div = compute_diversity(baseline_images)
    steered_div = compute_diversity(steered_images)

    return {
        "concept_acc": float(steered_positive_rate),
        "baseline_acc": float(baseline_positive_rate),
        "lift": float(steered_positive_rate - baseline_positive_rate),
        "diversity_baseline": baseline_div,
        "diversity_steered": steered_div,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LASD steering evaluation")
    parser.add_argument("--mode", required=True,
                        choices=["evaluate", "ablation", "compose", "baseline"])
    parser.add_argument("--concept", default="brightness")
    parser.add_argument("--method", default="rfm",
                        choices=["rfm", "mean_diff", "pca", "logreg"])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--n_images", type=int, default=EVAL_N_IMAGES)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices")
    parser.add_argument("--sweep", default=None,
                        help="Parameter to sweep for ablation mode")
    parser.add_argument("--concepts", default=None,
                        help="Comma-separated concepts for compose mode")
    args = parser.parse_args()

    timer = Timer()
    target_layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    print(f"Loading model {MODEL_ID}...")
    pipe = load_dit_pipeline()

    # ------------------------------------------------------------------
    if args.mode == "baseline":
        print(f"Generating {args.n_images} baseline images...")
        images, class_ids = generate_baseline(pipe, args.n_images)

        for concept_name, concept_cfg in CONCEPTS.items():
            labels = get_concept_labels(concept_cfg, images, class_ids)
            pos_rate = (labels == 1).mean()
            print(f"  {concept_name}: positive_rate={pos_rate:.3f}")

        print_summary(
            n_images=args.n_images,
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )

    # ------------------------------------------------------------------
    elif args.mode == "evaluate":
        results = evaluate_steering(
            pipe, args.concept, method=args.method,
            epsilon=args.epsilon, n_images=args.n_images, layers=target_layers,
        )

        if results:
            print(f"\n  Concept: {args.concept}, Method: {args.method}, Eps: {args.epsilon}")
            print(f"  Baseline positive rate: {results['baseline_acc']:.3f}")
            print(f"  Steered positive rate:  {results['concept_acc']:.3f}")
            print(f"  Lift:                   {results['lift']:.3f}")

            print_summary(
                concept_acc=results["concept_acc"],
                baseline_acc=results["baseline_acc"],
                lift=results["lift"],
                method=args.method,
                epsilon=args.epsilon,
                peak_vram_mb=get_peak_memory_mb(),
                wall_seconds=timer.elapsed(),
            )

    # ------------------------------------------------------------------
    elif args.mode == "ablation":
        print(f"Ablation: sweeping epsilon for {args.concept}/{args.method}")
        for eps in EPSILON_RANGE:
            results = evaluate_steering(
                pipe, args.concept, method=args.method,
                epsilon=eps, n_images=args.n_images // 2,  # smaller for speed
                layers=target_layers,
            )
            if results:
                print(f"  eps={eps:.3f}: concept_acc={results['concept_acc']:.3f}, "
                      f"lift={results['lift']:.3f}")

        print_summary(
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )

    # ------------------------------------------------------------------
    elif args.mode == "compose":
        concept_names = args.concepts.split("+") if args.concepts else ["brightness", "colorfulness"]
        print(f"Composing concepts: {concept_names}")

        # Load and sum vectors
        combined_vectors = {}
        for concept_name in concept_names:
            vectors = load_concept_vectors(concept_name, method=args.method, layers=target_layers)
            for layer_idx, vec in vectors.items():
                if layer_idx not in combined_vectors:
                    combined_vectors[layer_idx] = vec.clone()
                else:
                    combined_vectors[layer_idx] = combined_vectors[layer_idx] + vec

        # Normalize combined vectors
        for layer_idx in combined_vectors:
            combined_vectors[layer_idx] = combined_vectors[layer_idx] / len(concept_names)

        # Generate
        class_ids = torch.randint(0, NUM_CLASSES, (args.n_images,)).tolist()
        steered_images, _ = generate_steered(
            pipe, args.n_images, combined_vectors, epsilon=args.epsilon, class_ids=class_ids,
        )

        # Evaluate each concept
        for concept_name in concept_names:
            if concept_name in CONCEPTS:
                labels = get_concept_labels(CONCEPTS[concept_name], steered_images, class_ids)
                pos_rate = (labels == 1).mean()
                print(f"  {concept_name}: positive_rate={pos_rate:.3f}")

        print_summary(
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )


if __name__ == "__main__":
    main()
