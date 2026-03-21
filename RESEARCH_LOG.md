# LASD Research Log

## Project: Linear Activation Steering for Diffusion Transformers
**Branch:** `autoresearch/mar21`
**Model:** DiT-XL/2-256 (675M params, class-conditional ImageNet 256x256)
**Hardware:** Apple M1, 16GB unified memory, MPS backend

---

## 2026-03-21: Session 1 — Setup and First Experiments

### Motivation

We proposed LASD: extract linear concept directions from DiT activations using Recursive Feature Machines (RFM / AGOP) and steer generation by adding these vectors to the residual stream during denoising — at zero inference cost. The approach was inspired by Beaglehole et al. (Science 2026), who demonstrated that RFM-based concept vectors steer LLMs effectively, justified by the Neural Feature Ansatz (NFA): W^T W ∝ AGOP.

### Setup

- Installed dependencies via `uv sync` (torch 2.10, diffusers, transformers, scikit-learn)
- Only DiT-XL/2 has pretrained checkpoints (DiT-S/2 does not exist as a released model)
- DiT-XL/2 runs on MPS at ~1 step/sec, ~14-25 sec/image with 25 inference steps
- Created experiment infrastructure: `config.py`, `eval.py`, `extract.py`, `steer.py`
- Concepts defined as heuristic labelers: brightness (mean pixel > 0.5), colorfulness (cross-channel std)

### Experiment 1: Stage 0 — NFA Validation

**Question:** Does W^T W ∝ AGOP hold in DiT-XL/2?

**Method:** For each of 28 layers, computed:
- Neural Feature Matrix: W_ℓ^T W_ℓ from the feedforward layer weights
- AGOP approximation: activation covariance (1/n) Σ a_i a_i^T
- Cosine similarity between flattened NFM and AGOP

**Result:** **FAILED.** Mean cosine similarity = 0.031 across all layers. Max was 0.056 at layer 26. Zero layers above the 0.7 threshold.

| Layers 0-9 | Layers 10-19 | Layers 20-27 |
|---|---|---|
| 0.020 - 0.055 | 0.022 - 0.044 | 0.014 - 0.056 |

**Interpretation:** The NFA does NOT hold in DiTs trained with denoising score matching. This is fundamentally different from classification ViTs where cosine approaches 1.0.

**Why it fails (theoretical analysis):**
- DiT weights are shared across all timesteps; only adaLN scale/shift γ(t), β(t) change
- The effective weights at timestep t are W_eff(t) = γ(t) ⊙ W, but AGOP was computed marginally
- Denoising score matching trains a FAMILY of functions indexed by t, not a single classifier
- The weights are a compromise across timesteps; AGOP at any single t reflects only a slice

**Implication:** The Beaglehole/Radhakrishnan theoretical justification for RFM does NOT transfer to diffusion models. However, this does NOT mean concepts aren't linear — the NFA is about weight-activation geometry, not about concept encoding.

**Time:** ~16 min (50 images × 25 steps + AGOP computation)

---

### Experiment 2: Stage 1 — Linearity Probing

**Question:** Are concepts linearly decodable from DiT activations?

**Method:** Generated 100 images, recorded activations at all 28 layers (last denoising step). Trained logistic regression (linear) and MLP (nonlinear) probes at each layer for brightness and colorfulness.

**Bug encountered:** CFG doubles the batch internally (conditional + unconditional). Activation hooks captured both halves, giving 200 activations for 100 images. Fixed by taking only the first half of the batch in hooks.

**Result:** **PASSED with flying colors.**

**Brightness:**
| Layer | Linear Acc | MLP Acc | Gap |
|---|---|---|---|
| 0 (best) | **0.950** | 0.950 | 0.000 |
| 1-8 | 0.850-0.900 | 0.750-0.850 | -0.050 to -0.100 |
| 9-16 | 0.800-0.900 | 0.650-0.800 | -0.050 to -0.150 |
| 17-27 | 0.700-0.800 | 0.650-0.850 | -0.100 to +0.050 |

**Colorfulness:**
| Layer | Linear Acc | MLP Acc | Gap |
|---|---|---|---|
| 2,5,6,8 (best early) | **0.950** | 0.900 | -0.050 |
| 25,26 (best late) | **0.950** | 0.900-0.950 | -0.050 to 0.000 |
| 10-24 (middle) | 0.700-0.900 | 0.700-0.900 | -0.100 to +0.050 |

**Key findings:**
1. **95% linear probe accuracy** — concepts are strongly linearly encoded
2. **Zero or negative linearity gap** — MLP probes are NO BETTER than linear (often worse due to overfitting on small data). The structure is genuinely linear, not just approximately linear.
3. **Layer structure:**
   - Brightness: monotonically decreases from layer 0 → 27. Best at earliest layers.
   - Colorfulness: **bimodal** — peaks at early (2-8) AND late (25-26) layers, dips in middle (10-24)
4. **The "information U-curve":** Both concepts show reduced decodability in middle layers, suggesting a semantic bottleneck where low-level statistics are compressed.

**Interpretation:**
- Early layers encode input statistics (brightness = mean pixel, colorfulness = cross-channel std) inherited from the VAE latent representation
- Middle layers perform abstract semantic computation (class-conditional features, composition) in a space where low-level statistics are compressed
- Late layers re-encode for output, recreating color-channel structure for the noise prediction

**Time:** ~32 min (100 images × 25 steps + probing)

---

### Theoretical Pivot: What Should the Steering Direction Be?

After the NFA failure, we reconsidered the theory from first principles.

**Three types of directions:**
1. **Predictive** (probe weight w): maximizes corr(w^T a, concept_label). Tells us WHERE information is encoded.
2. **Causal** (Jacobian-derived): direction whose perturbation maximally changes the concept in the OUTPUT. Tells us what to PUSH.
3. **Value gradient** (∇_a V): the optimal perturbation from stochastic optimal control theory. Accounts for full trajectory dynamics.

**Key insight from Marks & Tegmark (ACL 2024):** In LLMs, mean-difference directions (μ₊ - μ₋) are MORE causally effective than logistic regression probes despite lower classification accuracy (NIE 0.85-1.03 vs 0.05-0.61). The maximum-margin separator found by logistic regression doesn't align with the directions the model actually uses.

**Our theoretical contribution:** Linear activation steering is a FIRST-ORDER APPROXIMATION to the value gradient of the optimal stochastic control problem:
- FK steering estimates V(x_t, t) in INPUT space via particles
- LASD estimates ∇_a V in ACTIVATION space via linear probing
- These are two projections of the same control-theoretic object
- LASD is the zero-cost approximation; FK is the exact (but expensive) solution

**Predictions from theory:**
1. Mean-diff should outperform logistic regression for steering (paralleling Marks & Tegmark)
2. This gap should be LARGER in diffusion than in LLMs (compounding through 25 denoising steps amplifies OOD perturbations)
3. Late layers should be more causally effective than early layers (fewer transformations to distort the perturbation)
4. The NFA should hold for adaLN-modulated weights W_eff(t) = γ(t) ⊙ W, not raw weights

---

### Converged Research Direction

**Title:** "Activation-Space Value Gradients for Diffusion Steering"

**Core claim:** Linear activation steering in DiTs is the zero-cost approximation to optimal stochastic control in activation space. The linear probe weight approximates the value gradient. This connects FK steering (exact, expensive) to activation editing (approximate, free) via a single theoretical framework.

**What makes it novel:**
1. The value gradient interpretation of activation steering (new theory)
2. NFA failure explained by timestep-sharing + testable modified ansatz (new finding)
3. First predictive-vs-causal comparison in diffusion models (new experiment, extending Marks & Tegmark)
4. The "information U-curve" in DiTs (new empirical finding)
5. Bridges stochastic control (FK literature) and representation geometry (activation editing literature)

---

### Next Steps (Stage 2)

Once concept vector extraction completes:
1. Test steering with mean-diff, logreg, RFM, PCA vectors at multiple layers
2. Compare predictive accuracy vs. causal effectiveness for each method
3. Find the causally optimal layer (not just the most predictive layer)
4. Test the modified NFA with adaLN-conditioned weights
5. Compare LASD Pareto frontier to FK steering (k=4) as upper bound

---

---

## 2026-03-21: Session 2 — Theory Refinement and Critic Review

### Multi-Agent Theory Review

Deployed two agents: a ruthless NeurIPS critic and a novelty checker.

### Novelty Check: CLEAR

No existing paper connects activation steering to the value function / optimal control in diffusion models:
- All optimal control papers (Berner 2022, Uehara 2024, VARD 2025) work in **input space**
- All activation steering papers (Kwon 2023, Li 2024, Wang 2026, AcT 2025) are purely empirical with **no control-theoretic framework**
- One important new competitor: **Wang et al. (Feb 2026)** "General and Efficient Steering of Unconditional Diffusion" — uses RFM/AGOP vectors in U-Net activation space, but provides NO theoretical justification. Our framework would explain their results.

### Critic's 5 Serious Weaknesses

**1. Math doesn't quite work.** V(x_t, t) ≠ V_a(a_ℓ, t) because spatial pooling is lossy. The chain rule ∇_x V = J^T ∇_a V requires the Jacobian.
→ **Fix**: weaken to "motivated by" rather than "identical to"; or derive properly with bounds.

**2. "First-order approximation" may be vacuous.** Every smooth function is locally linear. Also: logistic regression weight ≠ ∇_a E[r|a] (it's ∇_a log-odds, a different function).
→ **Fix**: specify radius of validity; use linear regression instead of logistic for the true gradient.

**3. FK connection is rhetorical.** FK has consistency guarantees + per-step adaptation; LASD is a fixed perturbation with no guarantees.
→ **Fix**: frame as "same motivation, different approximation level" not "same method."

**4. Concepts are trivially simple.** Brightness = mean pixel, colorfulness = channel std. These are first-order statistics linear by construction.
→ **Fix**: test on semantic concepts (animal, natural, style) with 1000+ samples.

**5. Sample size is 50-100x too small.** 20 test samples gives CI of ±10%. NFA AGOP with 50 samples in 1152-d space is rank-deficient.
→ **Fix**: need proper hardware for scale; acknowledge limitation for M1 results.

### Revised Theoretical Position

The honest, defensible claim:

> "Linear activation steering in diffusion models is **motivated by** the optimal control formulation: the probe weight vector is an empirical approximation to the direction of maximal expected reward improvement in activation space. This provides a **unified lens** for understanding why activation editing works (value gradient under linearity), why the NFA fails in DiTs (denoising AGOP ≠ concept AGOP), and which directions are most effective for steering (causal > predictive, per Marks & Tegmark)."

This is NOT: "we proved LASD = optimal control." It IS: "we provide the first theoretical framework connecting activation steering to stochastic control, with predictions that can be empirically tested."

### Extraction Complete

168 concept vectors extracted: brightness × 28 layers × {RFM, mean-diff, PCA} + colorfulness × 28 layers × {RFM, mean-diff, PCA}.

### Stage 2 Initiated

First experiment: mean-diff steering for brightness at layer 0, ε=0.1.
Key test: does steering with the predictive direction actually cause a concept shift?

---

### Commits

| Hash | Description |
|---|---|
| `43ae5ec` | Initial commit (background-info.md) |
| `eb78574` | LASD experiment infrastructure (config, eval, extract, steer, setup) |
| `dff5d2a` | Fix CFG batch doubling in activation hooks |
| `7d881a1` | Add logreg weight extraction + norm-clipping guardrail |
| `e9952ce` | Add research log, literature review, and proposal |

### Results Summary

| Commit | Stage | Metric | Value | Status | Description |
|---|---|---|---|---|---|
| eb78574 | S0 | nfa_cosine | 0.031 | gate_fail | NFA doesn't hold in DiTs |
| dff5d2a | S1 | probe_acc | 0.950 | gate_pass | Strong linearity: 95% acc, zero gap |
