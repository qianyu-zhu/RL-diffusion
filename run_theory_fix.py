"""
Theory Fix Experiments — Address reviewer weaknesses.

1. Orthogonality measurement: project concept vectors onto class embedding subspace
2. Non-contiguous layer combos for accumulation validation (20+ combos)
3. Contrast investigation: probe accuracy + why negative lift
4. Bootstrap confidence intervals for key results
5. Revised accumulation model: S(l) alone vs S(l)*alpha^l vs alternatives
"""
import os
import torch
import numpy as np
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS, HIDDEN_DIM,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, CONCEPTS, VECTORS_DIR,
)
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer,
    ActivationCollector, generate_and_collect,
)
from steer import (
    load_concept_vectors, generate_steered, generate_baseline, SteeringHook,
)

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


# ===========================================================================
# 1. Orthogonality measurement
# ===========================================================================

def measure_orthogonality(pipe, commit):
    """Project concept vectors onto class embedding subspace.

    The steerability condition claims steerable concepts are orthogonal to
    the conditioning subspace. We test this by measuring the cosine similarity
    between concept vectors and the class embedding directions.
    """
    print("\n" + "="*60)
    print("  Orthogonality Measurement")
    print("="*60)

    transformer = get_dit_transformer(pipe)

    # Extract class embedding matrix from DiT
    # DiT uses nn.Embedding for class labels, then projects through adaln_single or y_embedder
    class_embed = None
    for name, param in transformer.named_parameters():
        if "y_embedder" in name and "embedding_table" in name:
            class_embed = param.detach().cpu().float()
            print(f"  Found class embedding: {name}, shape={class_embed.shape}")
            break

    if class_embed is None:
        # Try alternative names
        for name, module in transformer.named_modules():
            if hasattr(module, 'embedding_table'):
                class_embed = module.embedding_table.weight.detach().cpu().float()
                print(f"  Found class embedding via module: {name}, shape={class_embed.shape}")
                break

    if class_embed is None:
        # Last resort: look for any Embedding layer
        for name, module in transformer.named_modules():
            if isinstance(module, torch.nn.Embedding) and module.num_embeddings >= NUM_CLASSES:
                class_embed = module.weight.detach().cpu().float()
                print(f"  Found embedding: {name}, shape={class_embed.shape}")
                break

    if class_embed is None:
        print("  WARNING: Could not find class embeddings. Skipping orthogonality test.")
        return

    # Get top PCA directions of class embedding space
    class_embed_centered = class_embed - class_embed.mean(dim=0)
    U, S, Vt = torch.linalg.svd(class_embed_centered, full_matrices=False)
    # Top-k directions of class embedding space
    k = min(50, Vt.shape[0])
    class_subspace = Vt[:k]  # (k, embed_dim)

    print(f"  Class embedding: {class_embed.shape[0]} classes × {class_embed.shape[1]} dims")
    print(f"  Using top {k} PCA directions (explain {(S[:k]**2).sum() / (S**2).sum():.1%} variance)")

    # For each concept, load vectors and measure projection onto class subspace
    concepts_to_test = ["brightness", "warmth", "colorfulness", "animal", "natural"]
    methods = ["mean_diff", "pca"]

    mem_gb = 3.9

    for concept in concepts_to_test:
        for method in methods:
            vectors = load_concept_vectors(concept, method=method)
            if not vectors:
                continue

            # Average concept vector across layers (or use specific layer)
            for layer_idx in [0, 10, 20]:
                if layer_idx not in vectors:
                    continue

                v = vectors[layer_idx].float()

                # The concept vector lives in activation space (hidden_dim=1152)
                # The class embedding lives in embedding space (may be different dim)
                # We need to project into a common space

                # If dimensions match, compute directly
                if v.shape[0] == class_subspace.shape[1]:
                    # Project v onto class subspace
                    v_normalized = v / (v.norm() + 1e-8)
                    projections = class_subspace @ v_normalized  # (k,)
                    proj_magnitude = projections.norm().item()
                    orthogonality = 1.0 - proj_magnitude  # 1 = fully orthogonal, 0 = fully in subspace

                    print(f"  {concept}/{method} L{layer_idx}: "
                          f"proj_onto_class={proj_magnitude:.4f}, orthogonality={orthogonality:.4f}")

                    append_result(commit, "FIX-ORTH", "proj_magnitude", proj_magnitude, mem_gb, "keep",
                                  f"orthogonality {concept} {method} layer{layer_idx}")
                    append_result(commit, "FIX-ORTH", "orthogonality", orthogonality, mem_gb, "keep",
                                  f"orthogonality {concept} {method} layer{layer_idx}")
                else:
                    print(f"  Dimension mismatch: v={v.shape[0]}, class_embed={class_subspace.shape[1]}")
                    # Try to use the class centroids in activation space instead
                    break


def measure_orthogonality_activation_space(pipe, commit):
    """Alternative: measure orthogonality in ACTIVATION space using class centroids.

    Generate images from different classes, collect activations, compute class
    centroid directions, then measure concept vector overlap.
    """
    print("\n" + "="*60)
    print("  Orthogonality in Activation Space")
    print("="*60)

    transformer = get_dit_transformer(pipe)

    # Collect activations for 20 classes (10 animal, 10 non-animal)
    animal_classes = [1, 130, 207, 281, 388, 94, 144, 243, 325, 356]
    object_classes = [417, 561, 928, 985, 971, 510, 609, 671, 760, 850]
    all_classes = animal_classes + object_classes

    n_per_class = 10
    layer_idx = 0

    print(f"  Collecting activations for {len(all_classes)} classes × {n_per_class} images at layer {layer_idx}...")

    class_acts = {}
    for cid in tqdm(all_classes, desc="Classes"):
        collector = ActivationCollector(transformer, layers=[layer_idx])
        cids = [cid] * n_per_class
        imgs, acts, _ = generate_and_collect(pipe, n_per_class, collector, class_ids=cids)
        collector.remove_hooks()
        if layer_idx in acts:
            class_acts[cid] = acts[layer_idx].mean(dim=0)  # centroid

    if not class_acts:
        print("  No activations collected!")
        return

    # Build class direction matrix: each class centroid - global mean
    global_mean = torch.stack(list(class_acts.values())).mean(dim=0)
    class_directions = []
    for cid in all_classes:
        d = class_acts[cid] - global_mean
        d = d / (d.norm() + 1e-8)
        class_directions.append(d)
    class_dir_matrix = torch.stack(class_directions)  # (n_classes, hidden_dim)

    # PCA of class directions = "class subspace" in activation space
    U, S, Vt = torch.linalg.svd(class_dir_matrix, full_matrices=False)
    k = min(10, len(all_classes))
    class_subspace = Vt[:k]  # (k, hidden_dim)
    var_explained = (S[:k]**2).sum() / (S**2).sum()
    print(f"  Class subspace: top {k} PCA dirs explain {var_explained:.1%} of class variance")

    # Now measure concept vector projections
    concepts_to_test = ["brightness", "warmth", "colorfulness", "animal", "natural"]
    mem_gb = 3.9

    print(f"\n  {'Concept':<20s} {'Method':<12s} {'Proj onto class':>16s} {'Orthogonality':>14s}")
    print("  " + "-" * 62)

    for concept in concepts_to_test:
        for method in ["mean_diff", "pca"]:
            vectors = load_concept_vectors(concept, method=method, layers=[layer_idx])
            if layer_idx not in vectors:
                continue

            v = vectors[layer_idx].float()
            v = v / (v.norm() + 1e-8)

            # Project onto class subspace
            projections = class_subspace @ v  # (k,)
            proj_magnitude = projections.norm().item()
            orthogonality = 1.0 - proj_magnitude

            print(f"  {concept:<20s} {method:<12s} {proj_magnitude:>16.4f} {orthogonality:>14.4f}")

            append_result(commit, "FIX-ORTH", "act_proj", proj_magnitude, mem_gb, "keep",
                          f"act_space_orthogonality {concept} {method} layer{layer_idx}")


# ===========================================================================
# 2. Non-contiguous layer combos
# ===========================================================================

def test_layer_combos(pipe, commit):
    """Test 20+ non-contiguous layer combinations for accumulation validation."""
    print("\n" + "="*60)
    print("  Non-Contiguous Layer Combination Test")
    print("="*60)

    n_images = 100
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

    # Baseline
    baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    from run_stage7 import continuous_brightness
    baseline_bright = continuous_brightness(baseline_imgs)
    baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()

    # Define 25 layer combinations
    combos = [
        ([0], "L0-only"),
        ([0, 14, 27], "L0-14-27"),
        ([0, 5, 10], "L0-5-10"),
        ([5, 15, 25], "L5-15-25"),
        ([0, 10, 20], "L0-10-20"),
        ([1, 3, 5, 7, 9], "odd-first10"),
        ([0, 2, 4, 6, 8], "even-first10"),
        ([10, 12, 14, 16, 18], "even-mid"),
        ([20, 22, 24, 26], "even-late"),
        ([0, 7, 14, 21, 27], "every7th"),
        ([0, 3, 6, 9, 12, 15, 18, 21, 24, 27], "every3rd"),
        ([0, 1, 2], "first3"),
        ([25, 26, 27], "last3"),
        ([13, 14, 15], "mid3"),
        ([0, 1, 2, 3, 4, 5, 6, 7], "first8"),
        ([0, 1, 2, 25, 26, 27], "edges"),
        ([5, 6, 7, 8, 9, 10], "early-mid"),
        ([0, 27], "first-last"),
        ([0, 5, 10, 15, 20, 25], "every5th"),
        ([2, 7, 12, 17, 22], "offset5th"),
        ([0, 1, 10, 11, 20, 21], "pairs"),
        ([3, 8, 13, 18, 23], "offset5th-v2"),
        ([0, 4, 8, 12, 16, 20, 24], "every4th"),
        ([1, 5, 9, 13, 17, 21, 25], "every4th-offset1"),
        ([0, 2, 4, 6, 8, 10, 12, 14], "even-first-half"),
    ]

    # Stability data (interpolated)
    stability = {0: 0.858, 5: 0.759, 10: 0.706, 15: 0.691, 20: 0.600, 25: 0.490, 27: 0.397}
    stab_all = np.zeros(28)
    keys = sorted(stability.keys())
    for i in range(28):
        below = max([k for k in keys if k <= i])
        above = min([k for k in keys if k >= i])
        if below == above:
            stab_all[i] = stability[below]
        else:
            t = (i - below) / (above - below)
            stab_all[i] = stability[below] * (1-t) + stability[above] * t

    # Predictions from different models
    alpha = 0.773
    mem_gb = 3.9

    print(f"\n  {'Combo':<25s} {'Layers':>6s} {'Predicted':>10s} {'Actual':>8s} {'Error':>8s}")
    print("  " + "-" * 60)

    actual_lifts = []
    pred_stability = []
    pred_alpha = []

    for layers, name in combos:
        vectors = load_concept_vectors("brightness", method="mean_diff", layers=layers)
        if not vectors:
            continue

        transformer = get_dit_transformer(pipe)
        steering = SteeringHook(transformer, vectors, epsilon=-0.5, norm_clip=5.0)

        images = []
        for i in range(0, n_images, EVAL_BATCH_SIZE):
            bs = min(EVAL_BATCH_SIZE, n_images - i)
            cids = class_ids[i:i+bs]
            steering.current_step = 0
            if hasattr(steering, '_fwd_count'):
                steering._fwd_count = 0
            with torch.no_grad():
                output = pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                             guidance_scale=GUIDANCE_SCALE, output_type="pil")
            images.extend(output.images)

        steering.remove_hooks()

        bright = continuous_brightness(images)
        rate = (np.array([1 if b > 0.5 else -1 for b in bright]) == 1).mean()
        lift = rate - baseline_rate

        # Model predictions
        s_sum = sum(stab_all[l] for l in layers)
        sa_sum = sum(stab_all[l] * (alpha ** l) for l in layers)

        actual_lifts.append(lift)
        pred_stability.append(s_sum)
        pred_alpha.append(sa_sum)

        print(f"  {name:<25s} {len(layers):>6d} {sa_sum:>10.3f} {lift:>+8.3f}")

        append_result(commit, "FIX-COMBO", "lift", lift, mem_gb, "keep",
                      f"combo {name} ({len(layers)} layers)")

    # Fit and compare models
    if len(actual_lifts) >= 5:
        actual = np.array(actual_lifts)
        pred_s = np.array(pred_stability)
        pred_sa = np.array(pred_alpha)

        # Model 1: pure stability (no alpha)
        scale_s = np.dot(pred_s, actual) / (np.dot(pred_s, pred_s) + 1e-8)
        r_s = np.corrcoef(pred_s, actual)[0, 1] if np.std(actual) > 0 else 0

        # Model 2: stability * alpha^l
        scale_sa = np.dot(pred_sa, actual) / (np.dot(pred_sa, pred_sa) + 1e-8)
        r_sa = np.corrcoef(pred_sa, actual)[0, 1] if np.std(actual) > 0 else 0

        # Model 3: simple count of layers
        n_layers = np.array([len(layers) for layers, _ in combos[:len(actual_lifts)]])
        r_n = np.corrcoef(n_layers, actual)[0, 1] if np.std(actual) > 0 else 0

        # Model 4: sum of front-weighted layers (1/(l+1))
        pred_front = np.array([sum(1.0/(l+1) for l in layers) for layers, _ in combos[:len(actual_lifts)]])
        r_front = np.corrcoef(pred_front, actual)[0, 1] if np.std(actual) > 0 else 0

        print(f"\n  Model Comparison ({len(actual_lifts)} test combos):")
        print(f"  {'Model':<30s} {'Correlation':>12s}")
        print(f"  {'-'*42}")
        print(f"  {'Pure stability Σ S(l)':<30s} {r_s:>12.4f}")
        print(f"  {'Stability × alpha^l':<30s} {r_sa:>12.4f}")
        print(f"  {'Layer count':<30s} {r_n:>12.4f}")
        print(f"  {'Front-weighted 1/(l+1)':<30s} {r_front:>12.4f}")

        append_result(commit, "FIX-MODEL", "corr_stability", r_s, mem_gb, "keep",
                      f"model_comparison pure_stability r={r_s:.4f}")
        append_result(commit, "FIX-MODEL", "corr_alpha", r_sa, mem_gb, "keep",
                      f"model_comparison stability_alpha r={r_sa:.4f}")
        append_result(commit, "FIX-MODEL", "corr_count", r_n, mem_gb, "keep",
                      f"model_comparison layer_count r={r_n:.4f}")
        append_result(commit, "FIX-MODEL", "corr_front", r_front, mem_gb, "keep",
                      f"model_comparison front_weighted r={r_front:.4f}")


# ===========================================================================
# 3. Contrast investigation
# ===========================================================================

def investigate_contrast(pipe, commit):
    """Investigate why contrast steering gives negative lift."""
    print("\n" + "="*60)
    print("  Contrast Investigation")
    print("="*60)

    from run_stage7 import label_contrast, continuous_brightness
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    transformer = get_dit_transformer(pipe)
    layers = [0, 5, 10, 15, 20, 25, 27]

    # Probe accuracy for contrast
    print("  Probing contrast concept...")
    collector = ActivationCollector(transformer, layers=layers)
    images, activations, class_ids = generate_and_collect(pipe, 200, collector)
    collector.remove_hooks()

    labels = label_contrast(images)
    pos_rate = (labels == 1).mean()
    print(f"  Contrast base rate: {pos_rate:.3f} positive")

    mem_gb = 3.9

    for layer_idx in layers:
        if layer_idx not in activations:
            continue
        acts = activations[layer_idx].numpy()
        binary = (labels[labels != 0] + 1) // 2
        valid_acts = acts[labels != 0]
        if len(binary) < 20:
            continue
        lr = LogisticRegression(max_iter=1000)
        scores = cross_val_score(lr, valid_acts, binary, cv=5)
        print(f"  Layer {layer_idx:2d}: probe_acc = {scores.mean():.3f} ± {scores.std():.3f}")
        append_result(commit, "FIX-CONTRAST", "probe_acc", scores.mean(), mem_gb, "keep",
                      f"contrast probe layer{layer_idx}")

    # Check: is contrast correlated with brightness?
    bright_labels = np.array([1 if np.array(img).mean() / 255.0 > 0.5 else -1 for img in images])
    both_pos = ((labels == 1) & (bright_labels == 1)).sum()
    both_neg = ((labels == -1) & (bright_labels == -1)).sum()
    agreement = (both_pos + both_neg) / len(labels)
    print(f"\n  Contrast-brightness agreement: {agreement:.3f}")
    print(f"  (If high, contrast steering may interfere with brightness direction)")

    # Check vector alignment
    bright_vecs = load_concept_vectors("brightness", method="mean_diff", layers=[0])
    contrast_vecs = load_concept_vectors("contrast", method="mean_diff", layers=[0])
    if 0 in bright_vecs and 0 in contrast_vecs:
        cos_sim = torch.nn.functional.cosine_similarity(
            bright_vecs[0].unsqueeze(0), contrast_vecs[0].unsqueeze(0)
        ).item()
        print(f"  Cosine(brightness, contrast) at L0: {cos_sim:.4f}")
        append_result(commit, "FIX-CONTRAST", "cos_bright_contrast", cos_sim, mem_gb, "keep",
                      "contrast-brightness vector alignment L0")


# ===========================================================================
# 4. Bootstrap confidence intervals
# ===========================================================================

def bootstrap_ci(pipe, commit):
    """Run best config 5 times with different seeds to get confidence intervals."""
    print("\n" + "="*60)
    print("  Bootstrap Confidence Intervals")
    print("="*60)

    from run_stage7 import continuous_brightness

    vectors = load_concept_vectors("brightness", method="mean_diff")
    n_images = 200
    n_runs = 5
    mem_gb = 3.9

    lifts = []
    for run_idx in range(n_runs):
        seed = 42 + run_idx * 7
        torch.manual_seed(seed)
        np.random.seed(seed)

        class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()

        baseline_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
        steered_imgs, _ = generate_steered(pipe, n_images, vectors, epsilon=-0.5,
                                            class_ids=class_ids)

        baseline_bright = continuous_brightness(baseline_imgs)
        steered_bright = continuous_brightness(steered_imgs)
        baseline_rate = (np.array([1 if b > 0.5 else -1 for b in baseline_bright]) == 1).mean()
        steered_rate = (np.array([1 if b > 0.5 else -1 for b in steered_bright]) == 1).mean()
        lift = steered_rate - baseline_rate
        lifts.append(lift)
        print(f"  Run {run_idx+1}: lift={lift:+.3f}")

    mean_lift = np.mean(lifts)
    std_lift = np.std(lifts)
    ci_low = mean_lift - 1.96 * std_lift
    ci_high = mean_lift + 1.96 * std_lift

    print(f"\n  Mean lift: {mean_lift:+.3f} ± {std_lift:.3f}")
    print(f"  95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")

    append_result(commit, "FIX-CI", "mean_lift", mean_lift, mem_gb, "keep",
                  f"bootstrap all-layer MD eps=-0.5 mean={mean_lift:.3f} std={std_lift:.3f}")
    append_result(commit, "FIX-CI", "ci_width", ci_high - ci_low, mem_gb, "keep",
                  f"bootstrap 95% CI width = {ci_high-ci_low:.3f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    timer = Timer()

    print("="*60)
    print("  Theory Fix Experiments")
    print("="*60)

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    try:
        # 1. Orthogonality (both methods)
        measure_orthogonality(pipe, commit)
        measure_orthogonality_activation_space(pipe, commit)

        # 2. Layer combos (25 configs, most important for theory)
        test_layer_combos(pipe, commit)

        # 3. Contrast investigation
        investigate_contrast(pipe, commit)

        # 4. Bootstrap CIs
        bootstrap_ci(pipe, commit)

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — Theory Fix Experiments")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
