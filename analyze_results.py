"""
Analysis and figure generation for LASD experiments.
Reads results.tsv and generates paper-ready figures and tables.
"""
import os
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# Try importing matplotlib; if not available, generate text tables only
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — generating text tables only")

RESULTS_FILE = "results.tsv"
FIGURES_DIR = "figures"


def read_results(filename=RESULTS_FILE):
    """Parse results.tsv into a list of dicts."""
    results = []
    if not os.path.exists(filename):
        print(f"No {filename} found")
        return results

    with open(filename) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                row["value"] = float(row["value"])
                row["memory_gb"] = float(row["memory_gb"])
            except (ValueError, KeyError):
                pass
            results.append(row)
    return results


def parse_description(desc):
    """Extract structured info from description string."""
    info = {}

    # Layer
    m = re.search(r'layer(\d+)', desc)
    if m:
        info["layer"] = int(m.group(1))

    # Layers (list)
    m = re.search(r'layers\[([^\]]+)\]', desc)
    if m:
        info["layers"] = m.group(1)

    # Epsilon
    m = re.search(r'eps=([+-]?[\d.]+)', desc)
    if m:
        info["eps"] = float(m.group(1))

    # Method
    for method in ["mean_diff", "rfm", "pca", "logreg"]:
        if method in desc:
            info["method"] = method
            break

    # Concept
    for concept in ["brightness", "colorfulness", "animal", "natural"]:
        if concept in desc:
            info["concept"] = concept
            break

    # Data efficiency M
    m = re.search(r'M=(\d+)', desc)
    if m:
        info["M"] = int(m.group(1))

    # RFM iter R
    m = re.search(r'R=(\d+)', desc)
    if m:
        info["R"] = int(m.group(1))

    # Schedule
    for sched in ["constant", "linear_decay", "linear_ramp", "cosine"]:
        if sched in desc:
            info["schedule"] = sched
            break

    # Norm clip
    m = re.search(r'clip_([.\d]+)', desc)
    if m:
        info["norm_clip"] = float(m.group(1))
    if "no_clip" in desc:
        info["norm_clip"] = float("inf")

    # Layer selection strategy
    for strat in ["single_best", "last_5", "last_10", "middle_5", "first_5", "all", "even"]:
        if strat in desc:
            info["layer_strategy"] = strat
            break

    # Strategy B bins
    m = re.search(r'bins=(\d+)', desc)
    if m:
        info["n_bins"] = int(m.group(1))

    return info


# =====================================================================
# Figure 1: Layer sweep — the predictive-causal gap
# =====================================================================

def figure_layer_sweep(results):
    """Layer sweep: lift vs layer index. The key figure."""
    layer_data = defaultdict(list)  # layer -> list of lifts

    for r in results:
        if r["stage"] != "S2":
            continue
        info = parse_description(r["description"])
        if "layer" in info and info.get("method") == "mean_diff":
            if info.get("concept", "brightness") == "brightness":
                layer_data[info["layer"]].append(r["value"])

    if not layer_data:
        print("  No layer sweep data found")
        return

    layers = sorted(layer_data.keys())
    # Take the max lift (best eps sign) for each layer
    lifts = [max(layer_data[l]) for l in layers]

    print("\n=== Figure 1: Layer Sweep (Predictive-Causal Gap) ===")
    print(f"  {'Layer':>5s}  {'Lift':>8s}")
    print(f"  {'-'*5}  {'-'*8}")
    for l, lift in zip(layers, lifts):
        bar = "█" * max(0, int(lift * 50)) if lift > 0 else "░" * max(0, int(-lift * 50))
        print(f"  {l:5d}  {lift:+8.3f}  {bar}")

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.bar(layers, lifts, color=["#2ecc71" if l > 0 else "#e74c3c" for l in lifts])
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Steering Lift (Δ positive rate)")
        ax.set_title("Brightness Steering: Lift by Layer (mean-diff, ε=0.5)")
        ax.set_xticks(layers)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(f"{FIGURES_DIR}/layer_sweep.pdf", bbox_inches="tight")
        fig.savefig(f"{FIGURES_DIR}/layer_sweep.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {FIGURES_DIR}/layer_sweep.{{pdf,png}}")


# =====================================================================
# Figure 2: Epsilon sweep — Pareto frontier
# =====================================================================

def figure_epsilon_sweep(results):
    """Epsilon vs lift at the best layer."""
    eps_data = {}

    for r in results:
        if r["stage"] != "S2":
            continue
        info = parse_description(r["description"])
        if "eps" in info and info.get("method") == "mean_diff":
            if info.get("concept", "brightness") == "brightness":
                eps = info["eps"]
                if eps not in eps_data or r["value"] > eps_data[eps]:
                    eps_data[eps] = r["value"]

    if not eps_data:
        print("  No epsilon sweep data found")
        return

    epsilons = sorted(eps_data.keys())
    lifts = [eps_data[e] for e in epsilons]

    print("\n=== Figure 2: Epsilon Sweep ===")
    print(f"  {'Epsilon':>8s}  {'Lift':>8s}")
    print(f"  {'-'*8}  {'-'*8}")
    for e, l in zip(epsilons, lifts):
        print(f"  {e:8.3f}  {l:+8.3f}")

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(epsilons, lifts, "o-", color="#3498db", markersize=8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Steering Strength (ε)")
        ax.set_ylabel("Steering Lift")
        ax.set_title("Epsilon Sweep at Best Layer")
        ax.grid(True, alpha=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(f"{FIGURES_DIR}/epsilon_sweep.pdf", bbox_inches="tight")
        fig.savefig(f"{FIGURES_DIR}/epsilon_sweep.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {FIGURES_DIR}/epsilon_sweep.{{pdf,png}}")


# =====================================================================
# Table 1: Method comparison
# =====================================================================

def table_method_comparison(results):
    """Compare extraction methods at best config."""
    method_data = {}

    for r in results:
        if r["stage"] != "S2":
            continue
        info = parse_description(r["description"])
        if "method" in info and info.get("concept", "brightness") == "brightness":
            method = info["method"]
            if method not in method_data or r["value"] > method_data[method]:
                method_data[method] = r["value"]

    if not method_data:
        print("  No method comparison data found")
        return

    print("\n=== Table 1: Method Comparison ===")
    print(f"  {'Method':>12s}  {'Lift':>8s}")
    print(f"  {'-'*12}  {'-'*8}")
    for method in ["mean_diff", "pca", "logreg", "rfm"]:
        if method in method_data:
            print(f"  {method:>12s}  {method_data[method]:+8.3f}")

    # LaTeX table
    print("\n  LaTeX:")
    print(r"  \begin{tabular}{lc}")
    print(r"  \toprule")
    print(r"  Method & Lift \\")
    print(r"  \midrule")
    for method in ["mean_diff", "pca", "logreg", "rfm"]:
        if method in method_data:
            name = {"mean_diff": "Mean-diff", "pca": "PCA", "logreg": "Logistic Reg.",
                    "rfm": "RFM (AGOP)"}[method]
            print(f"  {name} & {method_data[method]:+.3f} \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")


# =====================================================================
# Table 2: Ablation summary (Stage 3)
# =====================================================================

def table_ablations(results):
    """Summarize Stage 3 ablation results."""
    ablation_groups = defaultdict(list)

    for r in results:
        if r["stage"] != "S3":
            continue
        info = parse_description(r["description"])

        if "layer_strategy" in info:
            ablation_groups["layer_selection"].append(
                (info["layer_strategy"], r["value"]))
        elif "schedule" in info:
            ablation_groups["epsilon_schedule"].append(
                (info["schedule"], r["value"]))
        elif "norm_clip" in info:
            ablation_groups["norm_clip"].append(
                (str(info["norm_clip"]), r["value"]))
        elif "M" in info:
            ablation_groups["data_efficiency"].append(
                (str(info["M"]), r["value"]))
        elif "R" in info:
            ablation_groups["rfm_iterations"].append(
                (str(info["R"]), r["value"]))
        elif "n_bins" in info:
            ablation_groups["strategy_b"].append(
                (str(info["n_bins"]) + " bins", r["value"]))

    if not ablation_groups:
        print("  No ablation data found (Stage 3 not yet complete)")
        return

    print("\n=== Table 2: Ablation Summary ===")
    for group_name, entries in ablation_groups.items():
        print(f"\n  {group_name}:")
        print(f"    {'Setting':>20s}  {'Lift':>8s}")
        print(f"    {'-'*20}  {'-'*8}")
        for setting, lift in sorted(entries, key=lambda x: -x[1]):
            print(f"    {setting:>20s}  {lift:+8.3f}")


# =====================================================================
# Summary statistics
# =====================================================================

def summary(results):
    """Overall experiment summary."""
    stages = defaultdict(int)
    statuses = defaultdict(int)
    for r in results:
        stages[r["stage"]] += 1
        statuses[r["status"]] += 1

    print("\n=== Experiment Summary ===")
    print(f"  Total experiments: {len(results)}")
    print(f"  By stage: {dict(stages)}")
    print(f"  By status: {dict(statuses)}")

    # Best result per stage
    for stage in sorted(stages.keys()):
        stage_results = [r for r in results if r["stage"] == stage and r["status"] in ("keep", "gate_pass")]
        if stage_results:
            best = max(stage_results, key=lambda r: r["value"])
            print(f"  Best {stage}: {best['value']:+.3f} — {best['description']}")


def main():
    results = read_results()
    if not results:
        print("No results to analyze. Run experiments first.")
        return

    summary(results)
    figure_layer_sweep(results)
    figure_epsilon_sweep(results)
    table_method_comparison(results)
    table_ablations(results)


if __name__ == "__main__":
    main()
