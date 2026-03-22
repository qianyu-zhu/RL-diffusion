"""
Autonomous Stage 4 runner: Composition, Negative Steering, and Scale.
Runs after Stages 2-3 to test multi-concept and generalization.

Experiments:
  1. Two-concept composition (brightness + colorfulness)
  2. Three-concept composition (brightness + colorfulness + natural)
  3. Negative steering (concept suppression with -ε)
  4. Cross-class generalization test
  5. Semantic concept steering (animal, natural)
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_ID, DEVICE, DTYPE, NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS,
    NUM_INFERENCE_STEPS, GUIDANCE_SCALE, EVAL_BATCH_SIZE,
    CONCEPTS, VECTORS_DIR,
)
from eval import get_concept_labels, print_summary, Timer, get_peak_memory_mb
from extract import (
    load_dit_pipeline, get_dit_transformer, ActivationCollector,
    generate_and_collect, extract_concept_vector_mean_diff,
    extract_concept_vector_pca, linear_probe,
)
from steer import (
    load_concept_vectors, generate_steered, generate_baseline,
)

RESULTS_FILE = "results.tsv"
EVAL_N = 100


def log_result(commit, stage, metric, value, memory_gb, status, description):
    header = "commit\tstage\tmetric\tvalue\tmemory_gb\tstatus\tdescription\n"
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(header)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{stage}\t{metric}\t{value:.3f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def get_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def phase(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")


def read_stage2_best():
    """Read results.tsv to find best layer/eps from Stage 2."""
    best = {"layer": 25, "eps": 0.5}
    if not os.path.exists(RESULTS_FILE):
        return best
    best_lift = -999
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.startswith("commit"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            stage, value, status, desc = parts[1], float(parts[3]), parts[5], parts[6]
            if stage == "S2" and status == "keep" and value > best_lift:
                best_lift = value
                import re
                m = re.search(r'layer(\d+)', desc)
                if m:
                    best["layer"] = int(m.group(1))
                m = re.search(r'eps=([+-]?[\d.]+)', desc)
                if m:
                    best["eps"] = float(m.group(1))
    return best


def main():
    overall_timer = Timer()
    commit = get_commit()

    print(f"LASD Stage 4 — Composition & Scale")
    print(f"Commit: {commit}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    best = read_stage2_best()
    best_layer = best["layer"]
    best_eps = best["eps"]
    print(f"Using best config: layer={best_layer}, eps={best_eps}")

    pipe = load_dit_pipeline()
    transformer = get_dit_transformer(pipe)

    # Shared baseline and class ids
    baseline_cids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
    print(f"\nGenerating {EVAL_N} baseline images...")
    baseline_images, _ = generate_baseline(pipe, EVAL_N, class_ids=baseline_cids)

    # Compute all baseline rates
    baseline_rates = {}
    for concept_name, concept_cfg in CONCEPTS.items():
        labels = get_concept_labels(concept_cfg, baseline_images, baseline_cids)
        baseline_rates[concept_name] = (labels == 1).mean()
        print(f"  {concept_name} baseline: {baseline_rates[concept_name]:.3f}")

    # =====================================================================
    # Phase 1: Two-concept composition (brightness + colorfulness)
    # =====================================================================
    phase("Phase 1: Two-Concept Composition")

    # Load and combine vectors
    vecs_b = load_concept_vectors("brightness", method="mean_diff", layers=[best_layer])
    vecs_c = load_concept_vectors("colorfulness", method="mean_diff", layers=[best_layer])

    if vecs_b and vecs_c:
        combined = {}
        for layer_idx in set(vecs_b.keys()) | set(vecs_c.keys()):
            v = torch.zeros(vecs_b.get(layer_idx, vecs_c[layer_idx]).shape)
            if layer_idx in vecs_b:
                v = v + vecs_b[layer_idx]
            if layer_idx in vecs_c:
                v = v + vecs_c[layer_idx]
            v = v / v.norm()
            combined[layer_idx] = v

        for eps in [best_eps, best_eps * 0.5, best_eps * 2.0]:
            print(f"\n  brightness+colorfulness eps={eps}:")
            steered, _ = generate_steered(
                pipe, EVAL_N, combined, epsilon=eps, class_ids=baseline_cids
            )

            for concept_name in ["brightness", "colorfulness"]:
                labels = get_concept_labels(CONCEPTS[concept_name], steered, baseline_cids)
                rate = (labels == 1).mean()
                lift = rate - baseline_rates[concept_name]
                print(f"    {concept_name}: rate={rate:.3f}, lift={lift:+.3f}")

                log_result(commit, "S4", "lift", lift, get_gpu_mem_gb(),
                           "keep" if abs(lift) > 0.02 else "discard",
                           f"compose brightness+colorfulness {concept_name} eps={eps}")

    # =====================================================================
    # Phase 2: Negative steering (concept suppression)
    # =====================================================================
    phase("Phase 2: Negative Steering (Suppression)")

    for concept_name in ["brightness", "colorfulness"]:
        vectors = load_concept_vectors(concept_name, method="mean_diff", layers=[best_layer])
        if not vectors:
            continue

        for eps in [-best_eps, -best_eps * 2]:
            print(f"\n  {concept_name} suppression eps={eps}:")
            steered, _ = generate_steered(
                pipe, EVAL_N, vectors, epsilon=eps, class_ids=baseline_cids
            )
            labels = get_concept_labels(CONCEPTS[concept_name], steered, baseline_cids)
            rate = (labels == 1).mean()
            lift = rate - baseline_rates[concept_name]
            print(f"    rate={rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S4", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"negative {concept_name} eps={eps}")

    # =====================================================================
    # Phase 3: Cross-class generalization
    # =====================================================================
    phase("Phase 3: Cross-Class Generalization")

    # Test if steering works differently on specific class groups
    # Split: animals (0-397) vs objects (398-999)
    animal_cids = torch.randint(0, 398, (EVAL_N,)).tolist()
    object_cids = torch.randint(398, 1000, (EVAL_N,)).tolist()

    vectors = load_concept_vectors("brightness", method="mean_diff", layers=[best_layer])
    if vectors:
        for group_name, cids in [("animals", animal_cids), ("objects", object_cids)]:
            print(f"\n  brightness on {group_name}:")

            # Baseline for this group
            base, _ = generate_baseline(pipe, EVAL_N, class_ids=cids)
            base_labels = get_concept_labels(CONCEPTS["brightness"], base, cids)
            base_rate = (base_labels == 1).mean()

            steered, _ = generate_steered(
                pipe, EVAL_N, vectors, epsilon=best_eps, class_ids=cids
            )
            steer_labels = get_concept_labels(CONCEPTS["brightness"], steered, cids)
            steer_rate = (steer_labels == 1).mean()
            lift = steer_rate - base_rate

            print(f"    baseline={base_rate:.3f}, steered={steer_rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S4", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"cross_class brightness on {group_name} eps={best_eps}")

    # =====================================================================
    # Phase 4: Semantic concept steering (animal, natural)
    # =====================================================================
    phase("Phase 4: Semantic Concept Steering")

    # Need to extract vectors for semantic concepts first
    print("  Extracting semantic concept vectors (300 samples)...")
    collector = ActivationCollector(transformer, layers=[best_layer])
    imgs, acts, cids = generate_and_collect(pipe, 300, collector)
    collector.remove_hooks()

    for concept_name in ["animal", "natural"]:
        concept_cfg = CONCEPTS[concept_name]
        labels = get_concept_labels(concept_cfg, imgs, cids)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask = labels_t != 0

        n_pos = (labels == 1).sum()
        n_neg = (labels == -1).sum()
        print(f"\n  {concept_name}: {n_pos} positive, {n_neg} negative")

        if n_pos < 10 or n_neg < 10:
            print(f"    Too few samples, skipping")
            continue

        if best_layer not in acts:
            continue

        acts_valid = acts[best_layer][mask]
        labels_valid = labels_t[mask]

        # Extract vectors
        vec_md = extract_concept_vector_mean_diff(acts_valid, labels_valid)

        # Probe accuracy
        acc, _ = linear_probe(acts[best_layer], labels_t.long())
        print(f"    probe accuracy at layer {best_layer}: {acc:.3f}")

        # Steer
        vecs = {best_layer: vec_md}

        # Generate baseline with mixed classes
        sem_cids = torch.randint(0, NUM_CLASSES, (EVAL_N,)).tolist()
        sem_base, _ = generate_baseline(pipe, EVAL_N, class_ids=sem_cids)
        sem_base_labels = get_concept_labels(concept_cfg, sem_base, sem_cids)
        sem_base_rate = (sem_base_labels == 1).mean()

        for eps in [best_eps, best_eps * 2]:
            steered, _ = generate_steered(
                pipe, EVAL_N, vecs, epsilon=eps, class_ids=sem_cids
            )
            steer_labels = get_concept_labels(concept_cfg, steered, sem_cids)
            steer_rate = (steer_labels == 1).mean()
            lift = steer_rate - sem_base_rate

            print(f"    eps={eps}: baseline={sem_base_rate:.3f}, steered={steer_rate:.3f}, lift={lift:+.3f}")

            log_result(commit, "S4", "lift", lift, get_gpu_mem_gb(),
                       "keep" if abs(lift) > 0.02 else "discard",
                       f"semantic {concept_name} mean_diff layer{best_layer} eps={eps}")

    # =====================================================================
    # Phase 5: Orthogonality of concept directions
    # =====================================================================
    phase("Phase 5: Concept Direction Orthogonality")

    concepts_with_vecs = []
    for concept_name in ["brightness", "colorfulness", "animal", "natural"]:
        vecs = load_concept_vectors(concept_name, method="mean_diff", layers=[best_layer])
        if vecs and best_layer in vecs:
            concepts_with_vecs.append((concept_name, vecs[best_layer]))

    if len(concepts_with_vecs) >= 2:
        print(f"  Pairwise cosine similarity of concept vectors at layer {best_layer}:")
        for i, (n1, v1) in enumerate(concepts_with_vecs):
            for n2, v2 in concepts_with_vecs[i+1:]:
                cos = torch.nn.functional.cosine_similarity(
                    v1.float().unsqueeze(0), v2.float().unsqueeze(0)
                ).item()
                print(f"    {n1} vs {n2}: cos={cos:.3f}")

    # =====================================================================
    # Summary
    # =====================================================================
    phase("DONE — Stage 4")
    total_time = overall_timer.elapsed()
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Peak GPU memory: {get_gpu_mem_gb():.1f} GB")

    print("\n--- results.tsv (S4 entries) ---")
    with open(RESULTS_FILE) as f:
        for line in f:
            if "S4" in line or line.startswith("commit"):
                print(line.rstrip())


if __name__ == "__main__":
    main()
