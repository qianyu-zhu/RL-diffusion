"""
Stage 9: Adaptive Steering (DriftLite-inspired)

Inspired by DriftLite (Ren et al., ICLR 2026), which shows that the optimal
drift control for guided diffusion can be computed by solving a small linear
system at each denoising step.

Key idea: instead of fixed epsilon, compute per-step optimal coefficients
that minimize the variance of the residual potential.

In DriftLite's framework:
    b_t(x) = Σ_i θ_t^i s_i(x)     — control drift as linear combo of bases
    θ_t = A_t^{-1} c_t              — optimal coefficients from variance minimization

Our activation-space analogue:
    h'_t = h_t + Σ_i θ_t^i v_i     — modified hidden state
    where v_i are concept vectors (mean-diff, PCA, etc.)
    and θ_t^i are computed to maximize expected concept shift while
    minimizing activation-space disruption.

Experiments:
1. Per-step adaptive epsilon using online variance estimation
2. Multi-basis steering: combine mean-diff + PCA + logreg as 3 bases
3. Layer-adaptive: different eps per layer solved jointly
4. Compare adaptive vs best fixed-eps configurations
"""
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS, HIDDEN_DIM,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, CONCEPTS, VECTORS_DIR,
)
from eval import (
    get_concept_labels, compute_diversity, print_summary, Timer, get_peak_memory_mb,
)
from extract import load_dit_pipeline, get_dit_transformer
from steer import (
    load_concept_vectors, generate_baseline,
    SteeringHook, METHOD_FILE_MAP,
)

RESULTS_FILE = "results.tsv"


def continuous_brightness(images):
    return np.array([np.array(img).astype(np.float32).mean() / 255.0 for img in images])


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


# ---------------------------------------------------------------------------
# Adaptive Steering Hook (DriftLite-inspired)
# ---------------------------------------------------------------------------

class AdaptiveSteeringHook:
    """Steering with per-step adaptive coefficients.

    Instead of fixed epsilon, uses an online estimate of the optimal
    coefficient at each denoising step based on the relationship between
    the concept vector and current activation statistics.

    Strategy: at each step, compute eps_t that maximizes the projection
    of the perturbation onto the concept direction while constraining
    the norm increase.

    Formally: eps_t = argmin_eps ||h + eps*v||^2 - lambda * (v^T (h + eps*v))
    which gives: eps_t = lambda * ||v||^2 / (2 * ||v||^2) - (v^T h) / ||v||^2
    Simplified: eps_t = (lambda/2) - (v^T h_mean) / ||v||^2

    We use a running estimate of h_mean from the current batch.
    """

    def __init__(self, transformer, concept_vectors, base_epsilon=-0.5,
                 adaptation_mode="norm_match", norm_clip=5.0):
        """
        Args:
            concept_vectors: dict {layer_idx: (d,) tensor}
            base_epsilon: base steering strength (scaled adaptively)
            adaptation_mode: "norm_match" | "projection_max" | "variance_min"
        """
        self.transformer = transformer
        self.concept_vectors = concept_vectors
        self.base_epsilon = base_epsilon
        self.adaptation_mode = adaptation_mode
        self.norm_clip = norm_clip
        self.hooks = []
        self._step_count = 0
        self._adaptive_eps = {}  # per-layer per-step eps
        self._register_hooks()

    def _register_hooks(self):
        blocks = self.transformer.transformer_blocks

        # Step counter on block 0
        self._fwd_count = 0
        def counter_hook(module, input, output):
            self._fwd_count += 1
            return output
        self._counter_hook = blocks[0].register_forward_hook(counter_hook)

        for layer_idx, vec in self.concept_vectors.items():
            if layer_idx < len(blocks):
                hook = blocks[layer_idx].register_forward_hook(
                    self._make_adaptive_hook(layer_idx, vec)
                )
                self.hooks.append(hook)

    def _make_adaptive_hook(self, layer_idx, concept_vec):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            v = concept_vec.to(hidden.device, hidden.dtype)

            # Compute adaptive epsilon based on current activations
            step_frac = self._fwd_count / max(NUM_INFERENCE_STEPS, 1)

            if self.adaptation_mode == "norm_match":
                # Scale eps so perturbation magnitude matches a fraction of activation norm
                h_norm = hidden.norm(dim=-1, keepdim=True).mean()
                v_norm = v.norm()
                # eps * v_norm should be proportional to h_norm * |base_eps|
                eps = self.base_epsilon * (h_norm / (v_norm + 1e-8)).item()
                # Decay over denoising (early steps matter more)
                eps *= (1.0 - 0.5 * step_frac)

            elif self.adaptation_mode == "projection_max":
                # Maximize concept projection: eps proportional to how much
                # the hidden states already align with the concept direction
                h_mean = hidden.mean(dim=(0, 1))  # (d,)
                projection = torch.dot(h_mean, v) / (v.norm() + 1e-8)
                # If projection is positive, the concept is present — push more
                # If negative, the concept is absent — push harder
                sign = -1 if self.base_epsilon < 0 else 1
                eps = self.base_epsilon * (1.0 + 0.5 * sign * projection.item())

            elif self.adaptation_mode == "variance_min":
                # DriftLite-inspired: minimize variance of the residual
                # Approximate: find eps that moves the batch centroid toward concept
                h_mean = hidden.mean(dim=(0, 1))  # (d,)
                h_var = hidden.var(dim=(0, 1)).mean()  # scalar
                proj = torch.dot(h_mean, v)
                v_sq = v.dot(v)
                # Optimal eps from variance minimization:
                # d/deps Var[h + eps*v] = 2*eps*||v||^2 + 2*Cov(h, v)
                # Setting to zero: eps = -Cov(h,v)/||v||^2
                # But we also want to steer, so add the base_eps contribution
                cov_hv = proj.item()
                eps_opt = -cov_hv / (v_sq.item() + 1e-8)
                # Blend optimal with base
                eps = 0.5 * eps_opt + 0.5 * self.base_epsilon

            else:
                eps = self.base_epsilon

            # Apply steering with computed eps
            orig_norm = hidden.norm(dim=-1, keepdim=True)
            hidden = hidden + eps * v.unsqueeze(0).unsqueeze(0)

            if self.norm_clip > 0:
                new_norm = hidden.norm(dim=-1, keepdim=True)
                max_norm = orig_norm * self.norm_clip
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
        if hasattr(self, '_counter_hook'):
            self._counter_hook.remove()


class MultiBasisSteeringHook:
    """Steer with multiple basis vectors, combining optimally.

    Uses mean-diff + PCA + logreg as 3 basis vectors per layer.
    Coefficients are fixed but optimized via grid search.
    """

    def __init__(self, transformer, basis_vectors, coefficients, norm_clip=5.0):
        """
        Args:
            basis_vectors: dict {layer_idx: list of (d,) tensors} — multiple bases per layer
            coefficients: list of floats — one per basis
        """
        self.transformer = transformer
        self.basis_vectors = basis_vectors
        self.coefficients = coefficients
        self.norm_clip = norm_clip
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        blocks = self.transformer.transformer_blocks
        for layer_idx, bases in self.basis_vectors.items():
            if layer_idx < len(blocks):
                hook = blocks[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx, bases)
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_idx, bases):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            orig_norm = hidden.norm(dim=-1, keepdim=True)

            # Add weighted combination of basis vectors
            for v, coeff in zip(bases, self.coefficients):
                v_dev = v.to(hidden.device, hidden.dtype)
                hidden = hidden + coeff * v_dev.unsqueeze(0).unsqueeze(0)

            if self.norm_clip > 0:
                new_norm = hidden.norm(dim=-1, keepdim=True)
                max_norm = orig_norm * self.norm_clip
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
# Generation with adaptive/multi-basis steering
# ---------------------------------------------------------------------------

def generate_adaptive(pipe, n_images, concept_vectors, base_epsilon=-0.5,
                      adaptation_mode="norm_match", class_ids=None):
    """Generate images with adaptive steering."""
    transformer = get_dit_transformer(pipe)

    hook = AdaptiveSteeringHook(
        transformer, concept_vectors, base_epsilon=base_epsilon,
        adaptation_mode=adaptation_mode,
    )

    images = []
    used_cids = []
    for i in tqdm(range(0, n_images, EVAL_BATCH_SIZE), desc=f"Adaptive({adaptation_mode})"):
        batch_size = min(EVAL_BATCH_SIZE, n_images - i)
        cids = class_ids[i:i+batch_size] if class_ids else torch.randint(0, NUM_CLASSES, (batch_size,)).tolist()

        hook._fwd_count = 0
        with torch.no_grad():
            output = pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                         guidance_scale=GUIDANCE_SCALE, output_type="pil")
        images.extend(output.images)
        used_cids.extend(cids)

    hook.remove_hooks()
    return images, used_cids


def generate_multibasis(pipe, n_images, basis_vectors, coefficients, class_ids=None):
    """Generate images with multi-basis steering."""
    transformer = get_dit_transformer(pipe)

    hook = MultiBasisSteeringHook(transformer, basis_vectors, coefficients)

    images = []
    used_cids = []
    for i in tqdm(range(0, n_images, EVAL_BATCH_SIZE), desc="MultiBasis"):
        batch_size = min(EVAL_BATCH_SIZE, n_images - i)
        cids = class_ids[i:i+batch_size] if class_ids else torch.randint(0, NUM_CLASSES, (batch_size,)).tolist()

        with torch.no_grad():
            output = pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                         guidance_scale=GUIDANCE_SCALE, output_type="pil")
        images.extend(output.images)
        used_cids.extend(cids)

    hook.remove_hooks()
    return images, used_cids


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_adaptive_eps(pipe, commit):
    """Compare adaptive epsilon strategies against fixed epsilon."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Adaptive Epsilon (DriftLite-inspired)")
    print("="*60)

    n_images = 200
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    # Baseline
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_bright = continuous_brightness(baseline_imgs)
    baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()
    print(f"  Baseline pos rate: {baseline_rate:.3f}, mean bright: {baseline_bright.mean():.3f}")

    # Load all-layer mean_diff vectors (our best config)
    vectors = load_concept_vectors("brightness", method="mean_diff")

    mem_gb = 3.9

    # Test each adaptation mode
    for mode in ["norm_match", "projection_max", "variance_min"]:
        for base_eps in [-0.5, -0.3, -0.1]:
            imgs, _ = generate_adaptive(
                pipe, n_images, vectors, base_epsilon=base_eps,
                adaptation_mode=mode, class_ids=class_ids,
            )
            bright = continuous_brightness(imgs)
            rate = (np.array([1 if b > 0.5 else -1 for b in bright]) == 1).mean()
            lift = rate - baseline_rate

            status = "keep" if abs(lift) > 0.01 else "discard"
            append_result(commit, "S9-A", "lift", lift, mem_gb, status,
                          f"adaptive {mode} base_eps={base_eps} all-layer")
            print(f"  {mode} eps={base_eps}: lift={lift:+.3f}, mean_bright={bright.mean():.3f}")


def run_multibasis(pipe, commit):
    """Multi-basis steering: combine mean-diff + PCA + logreg."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Multi-Basis Steering")
    print("="*60)

    n_images = 200
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_bright = continuous_brightness(baseline_imgs)
    baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()

    # Load all three basis vectors for each layer
    methods = ["mean_diff", "pca", "logreg"]
    all_layers = list(range(NUM_LAYERS))

    basis_vectors = {}
    for layer_idx in all_layers:
        bases = []
        for method in methods:
            vecs = load_concept_vectors("brightness", method=method, layers=[layer_idx])
            if layer_idx in vecs:
                bases.append(vecs[layer_idx])
        if bases:
            basis_vectors[layer_idx] = bases

    if not basis_vectors:
        print("  No basis vectors found, skipping")
        return

    print(f"  Loaded {len(methods)} bases × {len(basis_vectors)} layers")

    mem_gb = 3.9

    # Grid search over coefficient combinations
    coeff_combos = [
        ([-0.5, 0.0, 0.0], "MD-only"),           # mean-diff only (baseline)
        ([0.0, -0.5, 0.0], "PCA-only"),           # PCA only
        ([0.0, 0.0, -0.5], "LR-only"),            # logreg only
        ([-0.3, -0.2, 0.0], "MD+PCA"),            # mean-diff + PCA
        ([-0.3, 0.0, -0.2], "MD+LR"),             # mean-diff + logreg
        ([-0.2, -0.2, -0.1], "all3-equal"),        # all three
        ([-0.4, -0.1, 0.0], "MD-heavy+PCA"),      # weighted toward mean-diff
        ([-0.5, 0.5, 0.0], "MD-PCA-oppose"),      # opposing (test orthogonality)
    ]

    for coeffs, name in coeff_combos:
        # Pad coefficients if fewer bases available
        actual_coeffs = coeffs[:len(methods)]

        imgs, _ = generate_multibasis(
            pipe, n_images, basis_vectors, actual_coeffs, class_ids=class_ids,
        )
        bright = continuous_brightness(imgs)
        rate = (np.array([1 if b > 0.5 else -1 for b in bright]) == 1).mean()
        lift = rate - baseline_rate

        status = "keep" if abs(lift) > 0.01 else "discard"
        append_result(commit, "S9-MB", "lift", lift, mem_gb, status,
                      f"multibasis {name} coeffs={coeffs}")
        print(f"  {name} {coeffs}: lift={lift:+.3f}, mean_bright={bright.mean():.3f}")


def run_layer_adaptive(pipe, commit):
    """Layer-adaptive: different epsilon per layer."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Layer-Adaptive Epsilon")
    print("="*60)

    n_images = 200
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_bright = continuous_brightness(baseline_imgs)
    baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()

    from steer import generate_steered

    mem_gb = 3.9

    # Strategies for per-layer epsilon
    strategies = {
        "uniform_-0.5": {i: -0.5 for i in range(28)},
        "front_heavy": {i: -0.8 * max(0, 1 - i/14) for i in range(28)},
        "back_heavy": {i: -0.8 * max(0, i/27 - 0.5) * 2 for i in range(28)},
        "V_shape": {i: -0.5 * (1 - abs(i - 14) / 14) for i in range(28)},
        "U_shape": {i: -0.5 * abs(i - 14) / 14 for i in range(28)},
        "exponential_decay": {i: -0.8 * (0.9 ** i) for i in range(28)},
        "first10_only": {i: -0.5 if i < 10 else 0.0 for i in range(28)},
        "odd_layers": {i: -0.5 if i % 2 == 1 else 0.0 for i in range(28)},
    }

    vectors = load_concept_vectors("brightness", method="mean_diff")

    for name, per_layer_eps in strategies.items():
        # Filter out zero-eps layers
        active_vectors = {l: v for l, v in vectors.items() if per_layer_eps.get(l, 0) != 0}
        if not active_vectors:
            continue

        transformer = get_dit_transformer(pipe)
        steering = SteeringHook(
            transformer, active_vectors,
            epsilon=per_layer_eps,
            norm_clip=5.0,
        )

        images = []
        for i in range(0, n_images, EVAL_BATCH_SIZE):
            batch_size = min(EVAL_BATCH_SIZE, n_images - i)
            cids = class_ids[i:i+batch_size]
            steering.current_step = 0
            steering._fwd_count = 0
            with torch.no_grad():
                output = pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                             guidance_scale=GUIDANCE_SCALE, output_type="pil")
            images.extend(output.images)

        steering.remove_hooks()

        bright = continuous_brightness(images)
        rate = (np.array([1 if b > 0.5 else -1 for b in bright]) == 1).mean()
        lift = rate - baseline_rate

        n_active = sum(1 for v in per_layer_eps.values() if v != 0)
        status = "keep" if abs(lift) > 0.01 else "discard"
        append_result(commit, "S9-LA", "lift", lift, mem_gb, status,
                      f"layer_adaptive {name} ({n_active} active layers)")
        print(f"  {name} ({n_active} layers): lift={lift:+.3f}, mean_bright={bright.mean():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timer = Timer()

    print("="*60)
    print("  STAGE 9: Adaptive Steering (DriftLite-inspired)")
    print("="*60)

    print(f"\nLoading model {MODEL_ID}...")
    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    print(f"  Commit: {commit}")

    try:
        # 1. Adaptive epsilon strategies
        run_adaptive_eps(pipe, commit)

        # 2. Multi-basis steering
        run_multibasis(pipe, commit)

        # 3. Layer-adaptive epsilon
        run_layer_adaptive(pipe, commit)

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — Stage 9 Adaptive Experiments")
    print(f"  Total wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"  Peak GPU memory: {get_peak_memory_mb():.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
