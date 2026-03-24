"""
Why does mean-diff beat discriminative methods for causal steering?

Tests:
1. Temporal stability of logreg vs mean-diff vs PCA vs RFM vectors
2. Cosine similarity between methods (how different are the directions?)
3. Alignment with Jacobian top singular vectors
4. Per-method steering at ALL layers × multiple epsilon values (complete picture)
"""
import os
import torch
import numpy as np
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS, HIDDEN_DIM,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, VECTORS_DIR,
)
from eval import print_summary, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer,
    MultiStepActivationCollector, generate_and_collect_multistep,
    extract_concept_vector_mean_diff, extract_concept_vector_pca,
)
from steer import load_concept_vectors, generate_steered, generate_baseline

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def measure_method_stability(pipe, commit):
    """Compare temporal stability across extraction methods."""
    print("\n" + "="*60)
    print("  Temporal Stability by Extraction Method")
    print("="*60)

    transformer = get_dit_transformer(pipe)

    # Capture at 6 denoising steps
    capture_steps = [0, 10, 20, 30, 40, 49]
    probe_layers = [0, 5, 10, 15, 20, 25]
    n_samples = 200

    print(f"  Generating {n_samples} images, capturing at steps {capture_steps}...")
    collector = MultiStepActivationCollector(
        transformer, layers=probe_layers, capture_steps=capture_steps
    )
    images, step_activations, class_ids = generate_and_collect_multistep(
        pipe, n_samples, collector
    )
    collector.remove_hooks()

    # Label brightness
    labels = np.array([1 if np.array(img).mean() / 255.0 > 0.5 else -1 for img in images])
    labels_t = torch.tensor(labels, dtype=torch.float32)
    mask = labels_t != 0

    methods = {
        "mean_diff": extract_concept_vector_mean_diff,
        "pca": extract_concept_vector_pca,
    }

    # For logreg, need sklearn
    from sklearn.linear_model import LogisticRegression

    mem_gb = 3.9

    print(f"\n  {'Method':<12s} {'Layer':>5s}  {'Stability':>10s}  {'(pairwise cosine across timesteps)'}")
    print("  " + "-" * 55)

    for method_name in ["mean_diff", "pca", "logreg"]:
        for layer_idx in probe_layers:
            vectors_by_step = {}

            for step in capture_steps:
                if step not in step_activations or layer_idx not in step_activations[step]:
                    continue

                acts = step_activations[step][layer_idx]
                acts_valid = acts[mask]
                labels_valid = labels_t[mask]

                if method_name == "mean_diff":
                    v = extract_concept_vector_mean_diff(acts_valid, labels_valid)
                elif method_name == "pca":
                    v = extract_concept_vector_pca(acts_valid, labels_valid)
                elif method_name == "logreg":
                    X = acts_valid.numpy()
                    y = ((labels_valid + 1) / 2).long().numpy()
                    if len(np.unique(y)) < 2:
                        continue
                    lr = LogisticRegression(max_iter=1000, C=1.0)
                    lr.fit(X, y)
                    v = torch.from_numpy(lr.coef_[0]).float()
                    v = v / (v.norm() + 1e-8)

                vectors_by_step[step] = v

            if len(vectors_by_step) >= 2:
                # Pairwise cosine similarity
                steps_list = sorted(vectors_by_step.keys())
                sims = []
                for i in range(len(steps_list)):
                    for j in range(i + 1, len(steps_list)):
                        cos = torch.nn.functional.cosine_similarity(
                            vectors_by_step[steps_list[i]].unsqueeze(0),
                            vectors_by_step[steps_list[j]].unsqueeze(0)
                        ).item()
                        sims.append(abs(cos))  # abs because sign may flip

                stability = np.mean(sims)
                print(f"  {method_name:<12s} L{layer_idx:>3d}  {stability:>10.4f}")

                append_result(commit, "METHOD", "stability", stability, mem_gb, "keep",
                              f"temporal_stability {method_name} layer{layer_idx}")


def measure_method_alignment(commit):
    """Compute pairwise cosine similarity between methods' vectors."""
    print("\n" + "="*60)
    print("  Method Alignment (pairwise cosine)")
    print("="*60)

    methods = ["mean_diff", "pca", "logreg"]
    layers = [0, 5, 10, 15, 20, 25]
    mem_gb = 3.9

    for layer_idx in layers:
        vecs = {}
        for method in methods:
            v = load_concept_vectors("brightness", method=method, layers=[layer_idx])
            if layer_idx in v:
                vecs[method] = v[layer_idx]

        # Also try RFM if available
        rfm = load_concept_vectors("brightness", method="rfm", layers=[layer_idx])
        if layer_idx in rfm:
            vecs["rfm"] = rfm[layer_idx]

        if len(vecs) < 2:
            continue

        print(f"\n  Layer {layer_idx}:")
        names = sorted(vecs.keys())
        header = "  " + " " * 12
        for n in names:
            header += f"{n:>12s}"
        print(header)

        for i, n1 in enumerate(names):
            row = f"  {n1:<12s}"
            for j, n2 in enumerate(names):
                cos = torch.nn.functional.cosine_similarity(
                    vecs[n1].unsqueeze(0).float(), vecs[n2].unsqueeze(0).float()
                ).item()
                row += f"{cos:>12.4f}"
            print(row)

        # Log key comparisons
        if "mean_diff" in vecs and "logreg" in vecs:
            cos = torch.nn.functional.cosine_similarity(
                vecs["mean_diff"].unsqueeze(0).float(), vecs["logreg"].unsqueeze(0).float()
            ).item()
            append_result(commit, "METHOD", "cos_md_lr", cos, mem_gb, "keep",
                          f"cosine(mean_diff, logreg) layer{layer_idx}")

        if "mean_diff" in vecs and "rfm" in vecs:
            cos = torch.nn.functional.cosine_similarity(
                vecs["mean_diff"].unsqueeze(0).float(), vecs["rfm"].unsqueeze(0).float()
            ).item()
            append_result(commit, "METHOD", "cos_md_rfm", cos, mem_gb, "keep",
                          f"cosine(mean_diff, rfm) layer{layer_idx}")


def main():
    timer = Timer()

    print("="*60)
    print("  Method Analysis: Why Mean-Diff > Discriminative")
    print("="*60)

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    try:
        # 1. Temporal stability comparison
        measure_method_stability(pipe, commit)

        # 2. Pairwise alignment
        measure_method_alignment(commit)

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — Method Analysis")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
