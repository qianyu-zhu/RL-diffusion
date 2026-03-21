"""
LASD concept vector extraction pipeline.
This file is modifiable during experiments.
"""
import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    RFM_ITERATIONS, RFM_RIDGE_LAMBDA, DEFAULT_N_SAMPLES, NFA_N_SAMPLES,
    CACHE_DIR, VECTORS_DIR, CONCEPTS, EVAL_BATCH_SIZE,
)
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dit_pipeline():
    """Load DiT pipeline from HuggingFace."""
    from diffusers import DiTPipeline
    pipe = DiTPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    return pipe


def get_dit_transformer(pipe):
    """Extract the DiT transformer from the pipeline."""
    return pipe.transformer


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

class ActivationCollector:
    """Register hooks on DiT blocks to collect residual-stream activations."""

    def __init__(self, transformer, layers=None):
        self.transformer = transformer
        self.activations = {}  # layer_idx -> tensor
        self.hooks = []
        target_layers = layers if layers is not None else list(range(len(transformer.transformer_blocks)))
        self._register_hooks(target_layers)

    def _register_hooks(self, layers):
        """Hook into DiT transformer blocks."""
        blocks = self.transformer.transformer_blocks
        for idx in layers:
            hook = blocks[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is the hidden states after this block
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # With CFG, batch is doubled (conditional + unconditional).
            # Take only the first half (conditional predictions).
            batch_size = hidden.shape[0]
            if batch_size > 1 and batch_size % 2 == 0:
                hidden = hidden[:batch_size // 2]
            # Spatially pool: (batch, num_patches, hidden_dim) -> (batch, hidden_dim)
            pooled = hidden.mean(dim=1).detach().cpu()
            self.activations[layer_idx] = pooled
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# Data generation + activation collection
# ---------------------------------------------------------------------------

def generate_and_collect(pipe, n_samples, collector, class_ids=None):
    """Generate images and collect activations at each layer.

    Returns:
        images: list of PIL images
        activations: dict of {layer_idx: (n_samples, hidden_dim) tensor}
        used_class_ids: list of class ids used
    """
    from diffusers import DiTPipeline

    images = []
    all_activations = {}
    used_class_ids = []

    for i in tqdm(range(0, n_samples, EVAL_BATCH_SIZE), desc="Generating"):
        batch_size = min(EVAL_BATCH_SIZE, n_samples - i)

        if class_ids is not None:
            cids = class_ids[i:i + batch_size]
        else:
            cids = torch.randint(0, NUM_CLASSES, (batch_size,)).tolist()

        collector.clear()

        with torch.no_grad():
            output = pipe(
                cids,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil",
            )

        images.extend(output.images)
        used_class_ids.extend(cids)

        # Collect activations (from the last denoising step)
        for layer_idx, act in collector.activations.items():
            if layer_idx not in all_activations:
                all_activations[layer_idx] = []
            all_activations[layer_idx].append(act)

    # Stack activations
    stacked = {}
    for layer_idx, act_list in all_activations.items():
        if act_list:
            stacked[layer_idx] = torch.cat(act_list, dim=0)

    return images, stacked, used_class_ids


# ---------------------------------------------------------------------------
# RFM / AGOP
# ---------------------------------------------------------------------------

def laplace_kernel(X, Y, bandwidth=1.0):
    """Compute Laplace kernel matrix K(X, Y)."""
    # X: (n, d), Y: (m, d)
    dists = torch.cdist(X, Y, p=1)  # L1 distance
    return torch.exp(-dists / bandwidth)


def kernel_ridge_regression(K_train, y, lam):
    """Solve kernel ridge regression: alpha = (K + lam*I)^{-1} y"""
    n = K_train.shape[0]
    alpha = torch.linalg.solve(K_train + lam * torch.eye(n), y.float())
    return alpha


def compute_agop(X, K_train, alpha, bandwidth=1.0):
    """Compute Average Gradient Outer Product for Laplace kernel.

    For Laplace kernel K(x,y) = exp(-||x-y||_1 / bw),
    the gradient w.r.t. x is: dK/dx = K(x,y) * (-sign(x-y) / bw)
    So df/dx = sum_j alpha_j * dK(x, x_j)/dx

    AGOP = (1/n) sum_i grad_f(x_i) grad_f(x_i)^T
    """
    n, d = X.shape

    # Compute gradients of the prediction function at each training point
    grads = torch.zeros(n, d)

    for i in range(n):
        # diff: (n, d)
        diff = X[i:i+1] - X  # (1, d) - (n, d) = (n, d)
        signs = torch.sign(diff)  # (n, d)
        k_vals = K_train[i]  # (n,)

        # grad f(x_i) = sum_j alpha_j * K(x_i, x_j) * (-sign(x_i - x_j) / bw)
        # shape: (d,)
        grad = (alpha * k_vals).unsqueeze(1) * (-signs / bandwidth)  # (n, d)
        grads[i] = grad.sum(dim=0)

    # AGOP = (1/n) * grads^T @ grads
    agop = (grads.T @ grads) / n

    return agop, grads


def extract_concept_vector_rfm(activations, labels, n_iter=RFM_ITERATIONS,
                                lam=RFM_RIDGE_LAMBDA, bandwidth=1.0):
    """Extract concept vector using Recursive Feature Machines.

    Args:
        activations: (n, d) tensor
        labels: (n,) tensor of +1/-1
        n_iter: number of RFM iterations
        lam: ridge regularization
        bandwidth: Laplace kernel bandwidth

    Returns:
        concept_vector: (d,) tensor — leading eigenvector of final AGOP
        agop: (d, d) tensor — final AGOP matrix
    """
    X = activations.float()
    y = labels.float()
    n, d = X.shape

    # Normalize
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-8)
    X_norm = (X - X_mean) / X_std

    M = torch.eye(d)  # feature weighting matrix

    for r in range(n_iter):
        # Reweight features
        X_weighted = X_norm @ M.float()

        # Compute kernel
        K = laplace_kernel(X_weighted, X_weighted, bandwidth=bandwidth)

        # Solve KRR
        alpha = kernel_ridge_regression(K, y, lam)

        # Compute AGOP
        agop, _ = compute_agop(X_weighted, K, alpha, bandwidth=bandwidth)

        # Update M with sqrt of AGOP diagonal (simplified)
        eigvals, eigvecs = torch.linalg.eigh(agop)
        # Take sqrt of eigenvalues for reweighting
        M = eigvecs @ torch.diag(eigvals.clamp(min=0).sqrt()) @ eigvecs.T

    # Final concept vector: leading eigenvector of AGOP
    eigvals, eigvecs = torch.linalg.eigh(agop)
    concept_vec = eigvecs[:, -1]  # largest eigenvalue

    # Orient: positive correlation with positive labels
    corr = torch.corrcoef(torch.stack([X_norm @ concept_vec, y]))[0, 1]
    if corr < 0:
        concept_vec = -concept_vec

    # Transform back to original space
    concept_vec = concept_vec / X_std

    return concept_vec, agop


def extract_concept_vector_mean_diff(activations, labels):
    """Baseline: mean difference between positive and negative examples."""
    pos_mask = labels == 1
    neg_mask = labels == -1
    v = activations[pos_mask].mean(dim=0) - activations[neg_mask].mean(dim=0)
    v = v / v.norm()
    return v


def extract_concept_vector_pca(activations, labels):
    """Baseline: top PC of positive-class activations."""
    pos_mask = labels == 1
    X_pos = activations[pos_mask].float()
    X_pos = X_pos - X_pos.mean(dim=0)
    _, _, V = torch.linalg.svd(X_pos, full_matrices=False)
    return V[0]  # top right singular vector


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def linear_probe(activations, labels, test_frac=0.2):
    """Train a linear probe (logistic regression) and return accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = activations.numpy()
    y = labels.numpy()

    # Filter out zeros
    mask = y != 0
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return 0.0, None

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=42)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)

    # Weight vector as concept direction
    weight = torch.tensor(clf.coef_[0], dtype=torch.float32)
    weight = weight / weight.norm()

    return acc, weight


def mlp_probe(activations, labels, test_frac=0.2):
    """Train a small MLP probe and return accuracy (measures nonlinearity gap)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    X = activations.numpy()
    y = labels.numpy()
    mask = y != 0
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return 0.0

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


# ---------------------------------------------------------------------------
# NFA validation
# ---------------------------------------------------------------------------

def validate_nfa(pipe, timesteps, n_samples=NFA_N_SAMPLES):
    """Validate the Neural Feature Ansatz in DiT: does W^T W ∝ AGOP?

    For each layer, compare the weight matrix's gram matrix with the AGOP
    computed over activations.
    """
    transformer = get_dit_transformer(pipe)
    blocks = transformer.transformer_blocks

    results = {}

    # Collect activations
    collector = ActivationCollector(transformer)
    images, activations, class_ids = generate_and_collect(pipe, n_samples, collector)
    collector.remove_hooks()

    for layer_idx in range(len(blocks)):
        if layer_idx not in activations:
            continue

        acts = activations[layer_idx].float()  # (n, d)

        # Get weight matrix from the block's MLP (ff layer)
        block = blocks[layer_idx]
        # DiT blocks typically have: norm1, attn, norm2, ff
        # We look at the first linear layer of the feedforward network
        try:
            W = None
            for name, param in block.named_parameters():
                if 'ff.net.0' in name and 'weight' in name:
                    W = param.detach().cpu().float()
                    break
                elif 'proj' in name and 'weight' in name and W is None:
                    W = param.detach().cpu().float()

            if W is None:
                continue

            # Neural Feature Matrix: W^T W
            if W.dim() == 2:
                nfm = W.T @ W
            else:
                continue

            # Compute AGOP from activations (using identity kernel as simple approximation)
            acts_centered = acts - acts.mean(dim=0)
            # Simple AGOP approximation: covariance of activations
            agop_approx = (acts_centered.T @ acts_centered) / acts_centered.shape[0]

            # Cosine similarity between flattened matrices
            nfm_flat = nfm.flatten()
            agop_flat = agop_approx.flatten()

            cos_sim = torch.nn.functional.cosine_similarity(
                nfm_flat.unsqueeze(0), agop_flat.unsqueeze(0)
            ).item()

            results[layer_idx] = cos_sim

        except Exception as e:
            results[layer_idx] = f"error: {e}"

    return results


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------

def compute_temporal_stability(concept_vectors_by_t):
    """Compute mean pairwise cosine similarity of concept vectors across timesteps.

    Args:
        concept_vectors_by_t: dict of {timestep: (d,) tensor}

    Returns:
        stability: float in [-1, 1]
        pairwise: dict of {(t1, t2): cosine_sim}
    """
    timesteps = sorted(concept_vectors_by_t.keys())
    if len(timesteps) < 2:
        return 1.0, {}

    pairwise = {}
    sims = []
    for i, t1 in enumerate(timesteps):
        for t2 in timesteps[i+1:]:
            v1 = concept_vectors_by_t[t1].float()
            v2 = concept_vectors_by_t[t2].float()
            cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            pairwise[(t1, t2)] = cos
            sims.append(cos)

    return float(np.mean(sims)), pairwise


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LASD extraction pipeline")
    parser.add_argument("--mode", required=True,
                        choices=["nfa_validate", "probe", "extract", "stability"])
    parser.add_argument("--model", default="dit-s2")
    parser.add_argument("--concepts", default="brightness,colorfulness",
                        help="Comma-separated concept names")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--timesteps", default="100,300,500,700,900",
                        help="Comma-separated timesteps for NFA validation")
    args = parser.parse_args()

    timer = Timer()
    os.makedirs(VECTORS_DIR, exist_ok=True)

    print(f"Loading model {MODEL_ID}...")
    pipe = load_dit_pipeline()
    transformer = get_dit_transformer(pipe)

    target_layers = None
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]

    # ------------------------------------------------------------------
    if args.mode == "nfa_validate":
        print("Stage 0: NFA Validation")
        results = validate_nfa(pipe, [int(t) for t in args.timesteps.split(",")])

        cos_values = [v for v in results.values() if isinstance(v, float)]
        mean_cos = np.mean(cos_values) if cos_values else 0.0

        for layer, cos in sorted(results.items()):
            print(f"  Layer {layer}: nfa_cosine = {cos:.4f}" if isinstance(cos, float)
                  else f"  Layer {layer}: {cos}")

        print_summary(
            nfa_cosine=mean_cos,
            n_layers_tested=len(cos_values),
            n_layers_above_07=sum(1 for v in cos_values if v > 0.7),
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )

    # ------------------------------------------------------------------
    elif args.mode == "probe":
        print("Stage 1: Linearity Probing")
        concept_names = args.concepts.split(",")

        collector = ActivationCollector(transformer, layers=target_layers)
        images, activations, class_ids = generate_and_collect(
            pipe, args.n_samples, collector
        )
        collector.remove_hooks()

        for concept_name in concept_names:
            if concept_name not in CONCEPTS:
                print(f"  Unknown concept: {concept_name}, skipping")
                continue

            concept_cfg = CONCEPTS[concept_name]
            labels = get_concept_labels(concept_cfg, images, class_ids)
            labels_t = torch.tensor(labels)

            print(f"\n  Concept: {concept_name}")
            print(f"  Positive: {(labels == 1).sum()}, Negative: {(labels == -1).sum()}")

            best_acc = 0.0
            best_layer = -1

            for layer_idx, acts in sorted(activations.items()):
                lin_acc, _ = linear_probe(acts, labels_t)
                mlp_acc = mlp_probe(acts, labels_t)

                print(f"    Layer {layer_idx}: linear={lin_acc:.3f}, mlp={mlp_acc:.3f}, "
                      f"gap={mlp_acc - lin_acc:.3f}")

                if lin_acc > best_acc:
                    best_acc = lin_acc
                    best_layer = layer_idx

            print(f"  Best: layer {best_layer}, acc={best_acc:.3f}")

        print_summary(
            probe_acc=best_acc,
            best_layer=best_layer,
            n_samples=args.n_samples,
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )

    # ------------------------------------------------------------------
    elif args.mode == "extract":
        print("Extracting concept vectors")
        concept_names = args.concepts.split(",")

        collector = ActivationCollector(transformer, layers=target_layers)
        images, activations, class_ids = generate_and_collect(
            pipe, args.n_samples, collector
        )
        collector.remove_hooks()

        for concept_name in concept_names:
            if concept_name not in CONCEPTS:
                continue

            concept_cfg = CONCEPTS[concept_name]
            labels = get_concept_labels(concept_cfg, images, class_ids)
            labels_t = torch.tensor(labels, dtype=torch.float32)

            # Filter valid labels
            mask = labels_t != 0

            for layer_idx, acts in sorted(activations.items()):
                acts_valid = acts[mask]
                labels_valid = labels_t[mask]

                # RFM extraction
                vec_rfm, agop = extract_concept_vector_rfm(acts_valid, labels_valid)

                # Save
                save_path = Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_rfm.pt"
                torch.save({
                    "vector": vec_rfm,
                    "agop": agop,
                    "method": "rfm",
                    "concept": concept_name,
                    "layer": layer_idx,
                    "n_samples": int(mask.sum()),
                }, save_path)
                print(f"  Saved {save_path}")

                # Also save baselines
                vec_md = extract_concept_vector_mean_diff(acts_valid, labels_valid)
                torch.save({"vector": vec_md, "method": "mean_diff"},
                           Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_meandiff.pt")

                vec_pca = extract_concept_vector_pca(acts_valid, labels_valid)
                torch.save({"vector": vec_pca, "method": "pca"},
                           Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_pca.pt")

                # Logistic regression weight (theory: ≈ value gradient)
                _, vec_lr = linear_probe(acts_valid, labels_valid.long())
                if vec_lr is not None:
                    torch.save({"vector": vec_lr, "method": "logreg", "layer": layer_idx},
                               Path(VECTORS_DIR) / f"{concept_name}_layer{layer_idx}_logreg.pt")

        print_summary(
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )

    # ------------------------------------------------------------------
    elif args.mode == "stability":
        print("Stage 1b: Temporal Stability Analysis")
        # For now, collect activations from the last step only
        # TODO: hook into multiple denoising steps
        print("  (temporal stability requires multi-step hooks — not yet implemented)")
        print_summary(
            stability=0.0,
            peak_vram_mb=get_peak_memory_mb(),
            wall_seconds=timer.elapsed(),
        )


if __name__ == "__main__":
    main()
