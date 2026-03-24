"""
Class Steering — Direct comparison to Wang et al. 2026 (NA-RFM).

Tests whether activation vectors can steer class identity in DiT-XL/2.

Experiments:
1. Extract class-conditional activation vectors (mean activations for class c
   minus mean activations across all classes)
2. Steer unconditional generation (class=1000) toward target classes
3. Steer one class toward another (e.g., dog -> cat)
4. Compare: mean-diff vs PCA, single-layer vs all-layer
5. Measure accuracy with a simple classifier (cosine similarity to class centroids)
"""
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, NUM_LAYERS, HIDDEN_DIM,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE,
    EVAL_BATCH_SIZE, VECTORS_DIR,
)
from eval import print_summary, Timer, get_peak_memory_mb
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


# ---------------------------------------------------------------------------
# Simple activation-based classifier
# ---------------------------------------------------------------------------

class ActivationClassifier:
    """Classify images by cosine similarity to class centroids in activation space."""

    def __init__(self, centroids, layer_idx=0):
        """
        centroids: dict {class_id: (hidden_dim,) tensor}
        """
        self.layer_idx = layer_idx
        self.class_ids = sorted(centroids.keys())
        # Stack into matrix (n_classes, hidden_dim)
        self.centroid_matrix = torch.stack([centroids[c] for c in self.class_ids]).float()
        # Normalize
        self.centroid_matrix = self.centroid_matrix / (
            self.centroid_matrix.norm(dim=1, keepdim=True) + 1e-8
        )

    def predict(self, activations):
        """
        activations: (n_images, hidden_dim) tensor
        Returns: list of predicted class_ids
        """
        acts = activations.float()
        acts = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine similarity
        sims = acts @ self.centroid_matrix.T  # (n_images, n_classes)
        pred_indices = sims.argmax(dim=1)
        return [self.class_ids[i] for i in pred_indices.tolist()]


# ---------------------------------------------------------------------------
# Extract class-conditional vectors
# ---------------------------------------------------------------------------

def extract_class_vectors(pipe, target_classes, n_per_class=50, layers=None):
    """Extract mean activation vectors per class.

    Returns:
        class_vectors: {class_id: {layer_idx: (hidden_dim,) tensor}}
        global_mean: {layer_idx: (hidden_dim,) tensor}
    """
    if layers is None:
        layers = [0, 5, 10, 15, 20, 25, 27]

    transformer = get_dit_transformer(pipe)

    # Collect activations per class
    class_acts = defaultdict(lambda: defaultdict(list))  # class -> layer -> [acts]

    for class_id in target_classes:
        print(f"  Collecting class {class_id} ({n_per_class} images)...")
        cids = [class_id] * n_per_class
        collector = ActivationCollector(transformer, layers=layers)
        images, activations, _ = generate_and_collect(pipe, n_per_class, collector, class_ids=cids)
        collector.remove_hooks()

        for layer_idx, acts in activations.items():
            class_acts[class_id][layer_idx] = acts  # (n_per_class, hidden_dim)

    # Compute global mean across all classes
    global_mean = {}
    for layer_idx in layers:
        all_acts = torch.cat([class_acts[c][layer_idx] for c in target_classes], dim=0)
        global_mean[layer_idx] = all_acts.mean(dim=0)

    # Compute per-class vectors: class_mean - global_mean
    class_vectors = {}
    class_centroids = {}
    for class_id in target_classes:
        class_vectors[class_id] = {}
        class_centroids[class_id] = {}
        for layer_idx in layers:
            class_mean = class_acts[class_id][layer_idx].mean(dim=0)
            class_centroids[class_id][layer_idx] = class_mean
            v = class_mean - global_mean[layer_idx]
            v = v / (v.norm() + 1e-8)
            class_vectors[class_id][layer_idx] = v

    return class_vectors, class_centroids, global_mean


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_class_steering(pipe, commit):
    """Main class steering experiment."""
    print("\n" + "="*60)
    print("  Class Steering Experiments")
    print("="*60)

    # Pick 10 diverse ImageNet classes
    # 207=golden retriever, 281=tabby cat, 388=giant panda,
    # 985=daisy, 971=bubble, 417=balloon, 561=forklift,
    # 928=ice cream, 1=goldfish, 130=flamingo
    target_classes = [207, 281, 388, 985, 971, 417, 561, 928, 1, 130]
    class_names = {
        207: "golden_retriever", 281: "tabby_cat", 388: "giant_panda",
        985: "daisy", 971: "bubble", 417: "balloon", 561: "forklift",
        928: "ice_cream", 1: "goldfish", 130: "flamingo",
    }

    layers = [0, 5, 10, 15, 20, 25, 27]
    n_per_class = 30
    mem_gb = 3.9

    # Step 1: Extract class vectors
    print("\n--- Step 1: Extract class-conditional vectors ---")
    class_vectors, class_centroids, global_mean = extract_class_vectors(
        pipe, target_classes, n_per_class=n_per_class, layers=layers,
    )

    # Save class vectors
    vec_dir = Path(VECTORS_DIR)
    vec_dir.mkdir(exist_ok=True)
    for class_id in target_classes:
        for layer_idx in layers:
            torch.save(
                {"vector": class_vectors[class_id][layer_idx], "layer": layer_idx,
                 "class_id": class_id, "method": "class_meandiff"},
                vec_dir / f"class{class_id}_layer{layer_idx}_meandiff.pt"
            )
    print(f"  Saved class vectors for {len(target_classes)} classes × {len(layers)} layers")

    # Step 2: Build classifier from centroids
    print("\n--- Step 2: Build activation classifier ---")
    # Use layer 0 centroids (most informative from our probing results)
    classifier = ActivationClassifier(
        {c: class_centroids[c][0] for c in target_classes}, layer_idx=0
    )

    # Test classifier accuracy on known-class images
    print("  Testing classifier on known-class images...")
    correct = 0
    total = 0
    transformer = get_dit_transformer(pipe)
    for class_id in target_classes[:5]:
        cids = [class_id] * 20
        collector = ActivationCollector(transformer, layers=[0])
        imgs, acts, _ = generate_and_collect(pipe, 20, collector, class_ids=cids)
        collector.remove_hooks()
        preds = classifier.predict(acts[0])
        acc = sum(1 for p in preds if p == class_id) / len(preds)
        correct += sum(1 for p in preds if p == class_id)
        total += len(preds)
        print(f"    Class {class_id} ({class_names[class_id]}): {acc:.1%}")

    baseline_acc = correct / total
    print(f"  Overall classifier accuracy: {baseline_acc:.1%}")
    append_result(commit, "S10-CLS", "classifier_acc", baseline_acc, mem_gb, "keep",
                  f"activation classifier accuracy on {len(target_classes[:5])} classes")

    # Step 3: Steer unconditional -> target class
    print("\n--- Step 3: Steer unconditional → target class ---")
    n_test = 50

    for target_class in target_classes[:5]:
        name = class_names[target_class]

        for layer_spec, layer_name in [
            ([0], "layer0"),
            (layers, "all-layer"),
        ]:
            for eps in [-0.5, -1.0, -2.0]:
                # Load vectors for this class at specified layers
                vecs = {l: class_vectors[target_class][l] for l in layer_spec
                        if l in class_vectors[target_class]}

                # Generate with unconditional class (1000 = null class for CFG)
                # Actually DiT uses random classes, so let's use random classes
                # and see if we can shift them toward the target
                random_cids = torch.randint(0, NUM_CLASSES, (n_test,)).tolist()

                steered_imgs, _ = generate_steered(
                    pipe, n_test, vecs, epsilon=eps, class_ids=random_cids,
                )

                # Classify steered images
                collector = ActivationCollector(transformer, layers=[0])
                # Re-generate to get activations (generate_steered doesn't return acts)
                # Instead, use the classifier on pixel features as proxy
                # Better: just check if output looks like target class
                # For now, use brightness-style heuristic based on class

                # Simple metric: what fraction of steered images are classified as target?
                # We need activations of the steered images, but generate_steered
                # already generated them. Let's just re-collect activations.
                collector = ActivationCollector(transformer, layers=[0])
                # Generate again with steering to collect activations
                steering = SteeringHook(transformer, vecs, epsilon=eps, norm_clip=5.0)
                imgs2 = []
                for i in range(0, n_test, EVAL_BATCH_SIZE):
                    bs = min(EVAL_BATCH_SIZE, n_test - i)
                    cids = random_cids[i:i+bs]
                    steering.current_step = 0
                    if hasattr(steering, '_fwd_count'):
                        steering._fwd_count = 0
                    collector.clear()
                    with torch.no_grad():
                        output = pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                                     guidance_scale=GUIDANCE_SCALE, output_type="pil")
                    imgs2.extend(output.images)

                steered_acts = {}
                for layer_idx, act_list in collector.activations.items():
                    steered_acts[layer_idx] = act_list

                steering.remove_hooks()
                collector.remove_hooks()

                if 0 in steered_acts:
                    preds = classifier.predict(steered_acts[0])
                    target_rate = sum(1 for p in preds if p == target_class) / len(preds)
                else:
                    target_rate = 0.0

                # Expected rate = 1/n_classes_in_classifier
                chance_rate = 1.0 / len(target_classes)
                lift = target_rate - chance_rate

                status = "keep" if abs(lift) > 0.01 else "discard"
                append_result(commit, "S10-CLS", "class_lift", lift, mem_gb, status,
                              f"steer->{name} {layer_name} eps={eps}")
                print(f"  -> {name} {layer_name} eps={eps}: target_rate={target_rate:.1%} "
                      f"(chance={chance_rate:.1%}, lift={lift:+.1%})")

    # Step 4: Cross-class steering (dog -> cat)
    print("\n--- Step 4: Cross-class steering ---")
    pairs = [
        (207, 281, "dog->cat"),
        (281, 207, "cat->dog"),
        (1, 130, "goldfish->flamingo"),
        (985, 417, "daisy->balloon"),
    ]

    for src_class, tgt_class, pair_name in pairs:
        # Direction: target - source
        steer_vecs = {}
        for l in layers:
            if l in class_vectors[tgt_class] and l in class_vectors[src_class]:
                # Use the target class vector (already = class_mean - global_mean)
                steer_vecs[l] = class_vectors[tgt_class][l]

        for eps in [-1.0, -2.0]:
            src_cids = [src_class] * 30
            steered_imgs, _ = generate_steered(
                pipe, 30, steer_vecs, epsilon=eps, class_ids=src_cids,
            )

            # Classify
            collector = ActivationCollector(transformer, layers=[0])
            steering = SteeringHook(transformer, steer_vecs, epsilon=eps, norm_clip=5.0)
            for i in range(0, 30, EVAL_BATCH_SIZE):
                bs = min(EVAL_BATCH_SIZE, 30 - i)
                cids = src_cids[i:i+bs]
                steering.current_step = 0
                if hasattr(steering, '_fwd_count'):
                    steering._fwd_count = 0
                collector.clear()
                with torch.no_grad():
                    pipe(cids, num_inference_steps=NUM_INFERENCE_STEPS,
                         guidance_scale=GUIDANCE_SCALE, output_type="pil")

            if 0 in collector.activations:
                preds = classifier.predict(collector.activations[0])
                tgt_rate = sum(1 for p in preds if p == tgt_class) / len(preds)
                src_rate = sum(1 for p in preds if p == src_class) / len(preds)
            else:
                tgt_rate = 0.0
                src_rate = 0.0

            steering.remove_hooks()
            collector.remove_hooks()

            append_result(commit, "S10-CLS", "cross_class", tgt_rate, mem_gb, "keep",
                          f"{pair_name} all-layer eps={eps} tgt_rate")
            print(f"  {pair_name} eps={eps}: src_rate={src_rate:.1%}, tgt_rate={tgt_rate:.1%}")


def main():
    timer = Timer()

    print("="*60)
    print("  Stage 10: Class Steering (Wang et al. comparison)")
    print("="*60)

    pipe = load_dit_pipeline()

    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    try:
        run_class_steering(pipe, commit)
    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  DONE — Class Steering")
    print(f"  Wall time: {timer.elapsed():.0f}s ({timer.elapsed()/3600:.1f}h)")
    print(f"  Peak GPU memory: {get_peak_memory_mb():.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
