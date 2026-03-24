"""
Stage 7: Diagnostic experiments for concept type boundaries.

Experiments:
1. Warm/cool color temperature concept — non-trivial visual property, not class-determined
2. Texture concepts: smooth vs textured (also not class-determined)
3. Semantic steering at extreme epsilon (5, 10, 20) — test if class conditioning can be overridden
4. Quality/diversity evaluation of best configurations
5. Continuous metric tracking (mean brightness, colorfulness values)
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS, HIDDEN_DIM,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, CONCEPTS, VECTORS_DIR,
)
from eval import (
    get_concept_labels, compute_diversity, print_summary, Timer, get_peak_memory_mb,
)
from extract import (
    load_dit_pipeline, get_dit_transformer,
    ActivationCollector, generate_and_collect,
)


def collect_activations(pipe, n_samples, layers):
    """Helper: generate images and collect activations at specified layers."""
    transformer = get_dit_transformer(pipe)
    collector = ActivationCollector(transformer, layers=layers)
    images, activations, class_ids = generate_and_collect(pipe, n_samples, collector)
    collector.remove_hooks()
    return images, activations, class_ids
from steer import (
    load_concept_vectors, generate_steered, generate_baseline, SteeringHook,
)

RESULTS_FILE = "results.tsv"
COMMIT = "PENDING"  # Will be filled after git commit


# ---------------------------------------------------------------------------
# New concept labelers (not in eval.py — these are experimental)
# ---------------------------------------------------------------------------

def label_warmth(images):
    """Warm (1) vs cool (-1) based on red-blue channel balance."""
    labels = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            warmth = arr[:, :, 0].mean() - arr[:, :, 2].mean()  # R - B
            labels.append(1 if warmth > 0.02 else -1)
        else:
            labels.append(0)
    return np.array(labels)


def label_texture(images):
    """Textured (1) vs smooth (-1) based on high-frequency content (Laplacian variance)."""
    labels = []
    for img in images:
        arr = np.array(img.convert("L")).astype(np.float32) / 255.0
        # Simple Laplacian: sum of absolute second differences
        lap_h = np.abs(arr[:, 2:] + arr[:, :-2] - 2 * arr[:, 1:-1])
        lap_v = np.abs(arr[2:, :] + arr[:-2, :] - 2 * arr[1:-1, :])
        texture_score = (lap_h.mean() + lap_v.mean()) / 2
        labels.append(1 if texture_score > 0.03 else -1)
    return np.array(labels)


def label_contrast(images):
    """High contrast (1) vs low contrast (-1)."""
    labels = []
    for img in images:
        arr = np.array(img.convert("L")).astype(np.float32) / 255.0
        contrast = arr.std()
        labels.append(1 if contrast > 0.25 else -1)
    return np.array(labels)


def continuous_brightness(images):
    """Return continuous brightness score per image."""
    return np.array([np.array(img).astype(np.float32).mean() / 255.0 for img in images])


def continuous_colorfulness(images):
    """Return continuous colorfulness score per image."""
    scores = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            scores.append(arr.std(axis=2).mean())
        else:
            scores.append(0.0)
    return np.array(scores)


def continuous_warmth(images):
    """Return continuous warmth (R-B) per image."""
    scores = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            scores.append(float(arr[:, :, 0].mean() - arr[:, :, 2].mean()))
        else:
            scores.append(0.0)
    return np.array(scores)


# ---------------------------------------------------------------------------
# Extract concept vectors for new concepts
# ---------------------------------------------------------------------------

def extract_new_concept_vectors(pipe, concept_name, labeler_fn, n_samples=500,
                                 methods=("mean_diff", "pca"), layers=None):
    """Extract concept vectors for a custom labeler."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    if layers is None:
        layers = list(range(NUM_LAYERS))

    transformer = get_dit_transformer(pipe)

    print(f"\n{'='*60}")
    print(f"  Extracting vectors for: {concept_name}")
    print(f"  Samples: {n_samples}, Layers: {len(layers)}")
    print(f"{'='*60}")

    # Check if vectors already exist
    vec_dir = Path(VECTORS_DIR)
    existing = list(vec_dir.glob(f"{concept_name}_layer*_meandiff.pt"))
    if len(existing) >= len(layers):
        print(f"  Vectors already exist ({len(existing)} files). Skipping extraction.")
        return

    # Generate images and collect activations
    images, activations, class_ids = collect_activations(pipe, n_samples, layers)

    # Label images
    labels = labeler_fn(images)
    pos_mask = labels == 1
    neg_mask = labels == -1
    print(f"  Labels: {pos_mask.sum()} positive, {neg_mask.sum()} negative")

    if pos_mask.sum() < 10 or neg_mask.sum() < 10:
        print(f"  WARNING: Too few samples in one class. Skipping.")
        return

    # Extract vectors for each layer
    vec_dir.mkdir(exist_ok=True)
    for layer_idx in layers:
        acts = activations[layer_idx].numpy()  # (n_samples, hidden_dim)
        pos_acts = acts[pos_mask]
        neg_acts = acts[neg_mask]

        if "mean_diff" in methods:
            md_vec = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
            md_vec = md_vec / (np.linalg.norm(md_vec) + 1e-8)
            torch.save(
                {"vector": torch.from_numpy(md_vec).float(), "layer": layer_idx,
                 "method": "mean_diff", "concept": concept_name},
                vec_dir / f"{concept_name}_layer{layer_idx}_meandiff.pt"
            )

        if "pca" in methods:
            n_pairs = min(len(pos_acts), len(neg_acts))
            diff = pos_acts[:n_pairs] - neg_acts[:n_pairs]  # matched pairs
            if len(diff) > 1:
                pca = PCA(n_components=1)
                pca.fit(diff)
                pca_vec = pca.components_[0]
                # Orient: positive correlation with positive class
                if np.dot(pca_vec, pos_acts.mean(axis=0) - neg_acts.mean(axis=0)) < 0:
                    pca_vec = -pca_vec
                pca_vec = pca_vec / (np.linalg.norm(pca_vec) + 1e-8)
                torch.save(
                    {"vector": torch.from_numpy(pca_vec).float(), "layer": layer_idx,
                     "method": "pca", "concept": concept_name},
                    vec_dir / f"{concept_name}_layer{layer_idx}_pca.pt"
                )

        if "logreg" in methods:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            binary = (labels[labels != 0] + 1) // 2  # 0/1
            valid_acts = acts[labels != 0]
            lr.fit(valid_acts, binary)
            lr_vec = lr.coef_[0]
            lr_vec = lr_vec / (np.linalg.norm(lr_vec) + 1e-8)
            torch.save(
                {"vector": torch.from_numpy(lr_vec).float(), "layer": layer_idx,
                 "method": "logreg", "concept": concept_name},
                vec_dir / f"{concept_name}_layer{layer_idx}_logreg.pt"
            )

    print(f"  Saved vectors to {vec_dir}/")


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def append_result(commit, stage, metric, value, memory_gb, status, description):
    """Append a result to results.tsv."""
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def run_quality_eval(pipe, commit):
    """Evaluate quality metrics (diversity, continuous scores) for best configs."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Quality & Diversity Evaluation")
    print("="*60)

    n_images = 200
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    # Generate baseline
    print("\nGenerating baseline...")
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)

    baseline_bright = continuous_brightness(baseline_imgs)
    baseline_color = continuous_colorfulness(baseline_imgs)
    baseline_warmth = continuous_warmth(baseline_imgs)
    baseline_div = compute_diversity(baseline_imgs)

    print(f"  Baseline: brightness={baseline_bright.mean():.3f}±{baseline_bright.std():.3f}, "
          f"colorfulness={baseline_color.mean():.4f}±{baseline_color.std():.4f}, "
          f"warmth={baseline_warmth.mean():.4f}±{baseline_warmth.std():.4f}, "
          f"diversity={baseline_div:.3f}")

    # Best configs to evaluate
    configs = [
        ("brightness", "mean_diff", -0.5, None, False, "all-layer MD eps=-0.5"),
        ("brightness", "mean_diff", -0.5, [0], False, "layer0 MD eps=-0.5"),
        ("brightness", "mean_diff", -0.5, list(range(5)), False, "first5 MD eps=-0.5"),
        ("brightness", "mean_diff", -0.1, [0], True, "layer0 norm_scale eps=-0.1"),
        ("colorfulness", "mean_diff", -0.5, None, False, "colorfulness all-layer MD eps=-0.5"),
    ]

    mem_gb = 3.9

    for concept, method, eps, layers, norm_scale, desc in configs:
        vectors = load_concept_vectors(concept, method=method, layers=layers)
        if not vectors:
            continue

        print(f"\nSteering: {desc}...")
        steered_imgs, _ = generate_steered(
            pipe, n_images, vectors, epsilon=eps, class_ids=class_ids,
            norm_scale=norm_scale,
        )

        s_bright = continuous_brightness(steered_imgs)
        s_color = continuous_colorfulness(steered_imgs)
        s_warmth = continuous_warmth(steered_imgs)
        s_div = compute_diversity(steered_imgs)

        bright_shift = s_bright.mean() - baseline_bright.mean()
        color_shift = s_color.mean() - baseline_color.mean()
        warmth_shift = s_warmth.mean() - baseline_warmth.mean()
        div_ratio = s_div / (baseline_div + 1e-8)

        print(f"  Steered:  brightness={s_bright.mean():.3f}±{s_bright.std():.3f}, "
              f"colorfulness={s_color.mean():.4f}±{s_color.std():.4f}")
        print(f"  Shifts:   brightness={bright_shift:+.3f}, colorfulness={color_shift:+.4f}, "
              f"warmth={warmth_shift:+.4f}")
        print(f"  Diversity: baseline={baseline_div:.3f}, steered={s_div:.3f}, ratio={div_ratio:.3f}")

        append_result(commit, "S7-Q", "bright_shift", bright_shift, mem_gb, "keep",
                      f"quality {desc}")
        append_result(commit, "S7-Q", "diversity_ratio", div_ratio, mem_gb, "keep",
                      f"diversity {desc}")


def run_warmth_concept(pipe, commit):
    """Extract and test warm/cool color temperature steering."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Warm/Cool Color Temperature Concept")
    print("="*60)

    # Extract vectors
    key_layers = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 27]
    extract_new_concept_vectors(
        pipe, "warmth", label_warmth, n_samples=500,
        methods=("mean_diff", "pca"), layers=key_layers,
    )

    # Probe accuracy first
    from sklearn.linear_model import LogisticRegression

    print("\nProbing warmth concept...")
    images, activations, class_ids = collect_activations(pipe, 200, key_layers)
    labels = label_warmth(images)

    pos_rate = (labels == 1).mean()
    print(f"  Warmth base rate: {pos_rate:.3f} positive")

    for layer_idx in [0, 5, 10, 15, 20, 25]:
        if layer_idx not in activations:
            continue
        acts = activations[layer_idx].numpy()
        binary = (labels[labels != 0] + 1) // 2
        valid_acts = acts[labels != 0]
        if len(binary) < 20:
            continue
        lr = LogisticRegression(max_iter=1000)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(lr, valid_acts, binary, cv=5)
        print(f"  Layer {layer_idx:2d}: probe_acc = {scores.mean():.3f} ± {scores.std():.3f}")
        append_result(commit, "S7-W", "probe_acc", scores.mean(), 3.9, "keep",
                      f"warmth probe layer{layer_idx}")

    # Steering tests
    n_images = 100
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_labels = label_warmth(baseline_imgs)
    baseline_rate = (baseline_labels == 1).mean()

    for method in ["mean_diff", "pca"]:
        for eps in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
            # All-layer
            vectors = load_concept_vectors("warmth", method=method, layers=key_layers)
            if not vectors:
                continue

            steered_imgs, _ = generate_steered(
                pipe, n_images, vectors, epsilon=eps, class_ids=class_ids,
            )
            steered_labels = label_warmth(steered_imgs)
            steered_rate = (steered_labels == 1).mean()
            lift = steered_rate - baseline_rate

            status = "keep" if abs(lift) > 0.01 else "discard"
            append_result(commit, "S7-W", "lift", lift, 3.9, status,
                          f"warmth {method} all-layer eps={eps}")

        # Single layer 0
        for eps in [-0.5, 0.5]:
            vectors = load_concept_vectors("warmth", method=method, layers=[0])
            if not vectors:
                continue
            steered_imgs, _ = generate_steered(
                pipe, n_images, vectors, epsilon=eps, class_ids=class_ids,
            )
            steered_labels = label_warmth(steered_imgs)
            lift = (steered_labels == 1).mean() - baseline_rate
            status = "keep" if abs(lift) > 0.01 else "discard"
            append_result(commit, "S7-W", "lift", lift, 3.9, status,
                          f"warmth {method} layer0 eps={eps}")


def run_contrast_concept(pipe, commit):
    """Extract and test contrast steering."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Contrast Concept")
    print("="*60)

    key_layers = [0, 5, 10, 15, 20, 25, 27]
    extract_new_concept_vectors(
        pipe, "contrast", label_contrast, n_samples=500,
        methods=("mean_diff", "pca"), layers=key_layers,
    )

    n_images = 100
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_labels = label_contrast(baseline_imgs)
    baseline_rate = (baseline_labels == 1).mean()

    for method in ["mean_diff", "pca"]:
        for eps in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
            vectors = load_concept_vectors("contrast", method=method, layers=key_layers)
            if not vectors:
                continue
            steered_imgs, _ = generate_steered(
                pipe, n_images, vectors, epsilon=eps, class_ids=class_ids,
            )
            steered_labels = label_contrast(steered_imgs)
            lift = (steered_labels == 1).mean() - baseline_rate
            status = "keep" if abs(lift) > 0.01 else "discard"
            append_result(commit, "S7-C", "lift", lift, 3.9, status,
                          f"contrast {method} all-layer eps={eps}")


def run_semantic_extreme_eps(pipe, commit):
    """Test semantic concepts at extreme epsilon to see if class conditioning can be overridden."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Semantic Concepts at Extreme Epsilon")
    print("="*60)

    n_images = 100

    for concept_name in ["animal", "natural"]:
        concept_cfg = CONCEPTS[concept_name]

        for method in ["mean_diff", "pca"]:
            for layer_spec, layer_name in [
                (None, "all-layer"),
                ([0], "layer0"),
                ([20], "layer20"),
            ]:
                vectors = load_concept_vectors(concept_name, method=method, layers=layer_spec)
                if not vectors:
                    continue

                for eps in [-5.0, -10.0, -20.0, 5.0, 10.0, 20.0]:
                    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

                    # Baseline
                    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
                    baseline_labels = get_concept_labels(concept_cfg, baseline_imgs, class_ids)
                    baseline_rate = (baseline_labels == 1).mean()

                    # Steered
                    steered_imgs, _ = generate_steered(
                        pipe, n_images, vectors, epsilon=eps, class_ids=class_ids,
                        norm_clip=0,  # no norm clip — let it push hard
                    )
                    steered_labels = get_concept_labels(concept_cfg, steered_imgs, class_ids)
                    steered_rate = (steered_labels == 1).mean()
                    lift = steered_rate - baseline_rate

                    status = "keep" if abs(lift) > 0.01 else "discard"
                    append_result(commit, "S7-SEM", "lift", lift, 3.9, status,
                                  f"semantic {concept_name} {method} {layer_name} eps={eps} no_clip")

                    # Also check if images are destroyed
                    bright_baseline = continuous_brightness(baseline_imgs).mean()
                    bright_steered = continuous_brightness(steered_imgs).mean()
                    div_steered = compute_diversity(steered_imgs)

                    print(f"  {concept_name} {method} {layer_name} eps={eps}: "
                          f"lift={lift:+.3f}, brightness={bright_steered:.3f} (was {bright_baseline:.3f}), "
                          f"diversity={div_steered:.3f}")


def run_texture_concept(pipe, commit):
    """Extract and test texture steering."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Texture Concept")
    print("="*60)

    key_layers = [0, 5, 10, 15, 20, 25, 27]
    extract_new_concept_vectors(
        pipe, "texture", label_texture, n_samples=500,
        methods=("mean_diff", "pca"), layers=key_layers,
    )

    n_images = 100
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    baseline_labels = label_texture(baseline_imgs)
    baseline_rate = (baseline_labels == 1).mean()

    for method in ["mean_diff", "pca"]:
        for eps in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
            vectors = load_concept_vectors("texture", method=method, layers=key_layers)
            if not vectors:
                continue
            steered_imgs, _ = generate_steered(
                pipe, n_images, vectors, epsilon=eps, class_ids=class_ids,
            )
            steered_labels = label_texture(steered_imgs)
            lift = (steered_labels == 1).mean() - baseline_rate
            status = "keep" if abs(lift) > 0.01 else "discard"
            append_result(commit, "S7-T", "lift", lift, 3.9, status,
                          f"texture {method} all-layer eps={eps}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timer = Timer()

    print("="*60)
    print("  STAGE 7: Concept Type Boundaries & Quality Evaluation")
    print("="*60)

    print(f"\nLoading model {MODEL_ID}...")
    pipe = load_dit_pipeline()

    # Get commit hash
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    print(f"  Commit: {commit}")

    # Run experiments in order of impact
    try:
        # 1. Quality eval of best configs (fast, 200 images)
        run_quality_eval(pipe, commit)

        # 2. Warm/cool color concept (medium, ~500+100 images)
        run_warmth_concept(pipe, commit)

        # 3. Contrast concept (medium, ~500+100 images)
        run_contrast_concept(pipe, commit)

        # 4. Texture concept (medium, ~500+100 images)
        run_texture_concept(pipe, commit)

        # 5. Semantic at extreme eps (slow but important, ~600 images)
        run_semantic_extreme_eps(pipe, commit)

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — Stage 7 Experiments")
    print(f"  Total wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"  Peak GPU memory: {get_peak_memory_mb():.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
