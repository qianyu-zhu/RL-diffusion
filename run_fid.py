"""
FID computation for best steering configurations.

Uses torchmetrics FID (if available) or scipy-based computation
against a reference set of unsteered images.
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE,
)
from eval import print_summary, Timer, get_peak_memory_mb
from extract import load_dit_pipeline
from steer import load_concept_vectors, generate_steered, generate_baseline

RESULTS_FILE = "results.tsv"


def append_result(commit, stage, metric, value, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"  >> {stage} | {metric}={value:.3f} | {status} | {description}")


def compute_inception_features(images, batch_size=16):
    """Extract Inception v3 features for FID computation."""
    from torchvision import transforms
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # Remove classification head
    model = model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        tensors = torch.stack([transform(img) for img in batch]).to(DEVICE)
        with torch.no_grad():
            feat = model(tensors)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(features_real, features_fake):
    """Compute FID between two sets of Inception features."""
    mu_real = features_real.mean(axis=0)
    mu_fake = features_fake.mean(axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)

    diff = mu_real - mu_fake
    # Matrix sqrt via eigendecomposition
    covmean_sq = sigma_real @ sigma_fake
    eigvals, eigvecs = np.linalg.eigh(covmean_sq)
    eigvals = np.maximum(eigvals, 0)  # numerical stability
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def main():
    timer = Timer()

    print("="*60)
    print("  FID Computation")
    print("="*60)

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    n_images = 500
    torch.manual_seed(42)
    class_ids = torch.randint(0, NUM_CLASSES, (n_images,)).tolist()
    mem_gb = 3.9

    # Reference: unsteered images
    print(f"\nGenerating {n_images} reference images...")
    ref_imgs, _ = generate_baseline(pipe, n_images, class_ids=class_ids)
    print("Computing reference features...")
    ref_features = compute_inception_features(ref_imgs)

    # Configs to evaluate
    configs = [
        ("brightness", "mean_diff", -0.5, None, "all-layer MD -0.5"),
        ("brightness", "mean_diff", -0.5, [0], "L0 MD -0.5"),
        ("brightness", "mean_diff", -0.5, list(range(5)), "first5 MD -0.5"),
        ("brightness", "mean_diff", -0.1, [0], "L0 MD -0.1"),
        ("warmth", "mean_diff", -0.5, None, "warmth all-layer -0.5"),
    ]

    for concept, method, eps, layers, desc in configs:
        print(f"\n--- {desc} ---")
        vectors = load_concept_vectors(concept, method=method, layers=layers)
        if not vectors:
            continue

        steered_imgs, _ = generate_steered(pipe, n_images, vectors, epsilon=eps,
                                            class_ids=class_ids)
        steered_features = compute_inception_features(steered_imgs)
        fid = compute_fid(ref_features, steered_features)

        print(f"  FID = {fid:.1f}")
        append_result(commit, "FID", "fid", fid, mem_gb, "keep", f"FID {desc}")

    print(f"\n{'='*60}")
    print(f"  DONE — FID Computation")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
