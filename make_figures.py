"""
Generate paper figures from results data.
Run locally (no GPU needed).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# --- Figure 1: Pareto Frontier ---

# Brightness
bright_eps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
bright_fid = [61.5, 61.7, 61.0, 62.6, 66.3, 76.3, 102.7]
bright_lift = [0.116, 0.174, 0.362, 0.460, 0.582, 0.604, 0.638]

# Animal
animal_eps = [1, 2, 5, 7, 10]
animal_fid = [63.9, 69.2, 144.8, 216.3, 279.4]
animal_clip = [0.017, 0.044, 0.141, 0.209, 0.265]

# Natural
natural_eps = [1, 2, 5, 7, 10]
natural_fid = [64.1, 69.0, 159.4, 250.8, 328.9]
natural_clip = [0.008, 0.031, 0.108, 0.171, 0.186]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(bright_fid, bright_lift, 'o-', color='#e74c3c', markersize=6, linewidth=2)
for i, e in enumerate(bright_eps):
    ax.annotate(f'ε={e}', (bright_fid[i], bright_lift[i]), fontsize=7,
                textcoords="offset points", xytext=(5, 5))
ax.set_xlabel('FID ↓')
ax.set_ylabel('Pixel Lift ↑')
ax.set_title('Brightness (pixel metric)')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(animal_fid, animal_clip, 's-', color='#2ecc71', markersize=6, linewidth=2)
for i, e in enumerate(animal_eps):
    ax.annotate(f'ε={e}', (animal_fid[i], animal_clip[i]), fontsize=7,
                textcoords="offset points", xytext=(5, 5))
ax.set_xlabel('FID ↓')
ax.set_ylabel('CLIP Animal Lift ↑')
ax.set_title('Animal (CLIP metric)')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(natural_fid, natural_clip, '^-', color='#3498db', markersize=6, linewidth=2)
for i, e in enumerate(natural_eps):
    ax.annotate(f'ε={e}', (natural_fid[i], natural_clip[i]), fontsize=7,
                textcoords="offset points", xytext=(5, 5))
ax.set_xlabel('FID ↓')
ax.set_ylabel('CLIP Natural Lift ↑')
ax.set_title('Natural (CLIP metric)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/pareto_frontier.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/pareto_frontier.png', bbox_inches='tight', dpi=150)
print("Saved figures/pareto_frontier.pdf")

# --- Figure 2: Accumulation (predicted vs actual) ---

combo_data = [
    ("L0-only", 1, 0.150),
    ("L0-14-27", 3, 0.270),
    ("L0-5-10", 3, 0.200),
    ("L5-15-25", 3, 0.050),
    ("L0-10-20", 3, 0.200),
    ("odd-first10", 5, 0.220),
    ("even-first10", 5, 0.300),
    ("even-mid", 5, 0.120),
    ("even-late", 4, 0.050),
    ("every7th", 5, 0.300),
    ("every3rd", 10, 0.360),
    ("first3", 3, 0.410),
    ("last3", 3, 0.090),
    ("mid3", 3, 0.050),
    ("first8", 8, 0.520),
    ("edges", 6, 0.350),
    ("early-mid", 6, 0.120),
    ("first-last", 2, 0.200),
    ("every5th", 6, 0.270),
    ("offset5th", 5, 0.170),
    ("pairs", 6, 0.360),
    ("offset5th-v2", 5, 0.120),
    ("every4th", 7, 0.240),
    ("every4th-off1", 7, 0.270),
    ("even-first-half", 8, 0.320),
]

# Layer indices for each combo
combos_layers = {
    "L0-only": [0], "L0-14-27": [0,14,27], "L0-5-10": [0,5,10],
    "L5-15-25": [5,15,25], "L0-10-20": [0,10,20],
    "odd-first10": [1,3,5,7,9], "even-first10": [0,2,4,6,8],
    "even-mid": [10,12,14,16,18], "even-late": [20,22,24,26],
    "every7th": [0,7,14,21,27], "every3rd": [0,3,6,9,12,15,18,21,24,27],
    "first3": [0,1,2], "last3": [25,26,27], "mid3": [13,14,15],
    "first8": [0,1,2,3,4,5,6,7], "edges": [0,1,2,25,26,27],
    "early-mid": [5,6,7,8,9,10], "first-last": [0,27],
    "every5th": [0,5,10,15,20,25], "offset5th": [2,7,12,17,22],
    "pairs": [0,1,10,11,20,21], "offset5th-v2": [3,8,13,18,23],
    "every4th": [0,4,8,12,16,20,24], "every4th-off1": [1,5,9,13,17,21,25],
    "even-first-half": [0,2,4,6,8,10,12,14],
}

# Compute predictions
actual_lifts = [d[2] for d in combo_data]
pred_front = [sum(1.0/(l+1) for l in combos_layers[d[0]]) for d in combo_data]

# Scale predictions
scale = np.dot(pred_front, actual_lifts) / (np.dot(pred_front, pred_front) + 1e-8)
pred_scaled = [p * scale for p in pred_front]

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.scatter(pred_scaled, actual_lifts, c='#2c3e50', s=40, zorder=5)
lims = [0, max(max(pred_scaled), max(actual_lifts)) * 1.1]
ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect prediction')
ax.set_xlabel('Predicted Lift (front-weighted model)')
ax.set_ylabel('Measured Lift')
ax.set_title(f'Accumulation Equation Validation (r = {np.corrcoef(pred_scaled, actual_lifts)[0,1]:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/accumulation_validation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/accumulation_validation.png', bbox_inches='tight', dpi=150)
print("Saved figures/accumulation_validation.pdf")

# --- Figure 3: Method comparison (stability + alignment) ---

methods = ['Mean-diff', 'PCA', 'Logreg', 'RFM']
stability = [0.859, 0.591, 0.503, np.nan]
cos_md = [1.0, -0.84, 0.422, 0.001]
lift = [0.525, -0.160, 0.460, -0.050]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']

ax = axes[0]
bars = ax.bar(methods[:3], stability[:3], color=colors[:3], edgecolor='black', linewidth=0.5)
ax.set_ylabel('Temporal Stability $S$')
ax.set_title('(a) Temporal Stability at L0')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
bars = ax.bar(methods, cos_md, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('cos(method, mean-diff)')
ax.set_title('(b) Alignment with Mean-Diff')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylim(-1, 1.1)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[2]
bars = ax.bar(methods, lift, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Steering Lift')
ax.set_title('(c) Steering Effectiveness')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/method_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/method_comparison.png', bbox_inches='tight', dpi=150)
print("Saved figures/method_comparison.pdf")

# --- Figure 4: Wang comparison bar chart ---

configs = ['LASD\nall-layer', 'LASD\nfirst-5', 'LASD\nε=0.1', 'Wang\nRFM', 'Wang\nnorm-scale\nL25', 'Wang\nnorm-scale\nL0']
pixel_lifts = [0.640, 0.420, 0.235, 0.050, 0.680, 0.680]
fids = [92.4, 83.1, 88.2, 82.8, 269.0, 286.6]
divs = [0.925, 1.053, 1.047, 1.005, 0.200, 0.075]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
x = np.arange(len(configs))
colors_wang = ['#2ecc71', '#27ae60', '#1abc9c', '#95a5a6', '#e74c3c', '#c0392b']

ax = axes[0]
ax.bar(x, pixel_lifts, color=colors_wang, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Pixel Lift ↑')
ax.set_title('(a) Concept Shift')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(x, fids, color=colors_wang, edgecolor='black', linewidth=0.5)
ax.set_ylabel('FID ↓')
ax.set_title('(b) Quality Degradation')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[2]
ax.bar(x, divs, color=colors_wang, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Diversity Ratio')
ax.set_title('(c) Sample Diversity')
ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--', label='Baseline')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=7)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/wang_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/wang_comparison.png', bbox_inches='tight', dpi=150)
print("Saved figures/wang_comparison.pdf")

print("\nAll figures generated!")
