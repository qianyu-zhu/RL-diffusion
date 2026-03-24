# Local Papers Reference

## Competitive Positioning vs Wang et al. 2026 (NA-RFM)

**What they have that we don't (yet):**
- Class-conditional steering (96.6% on CIFAR-10) — we test concepts, not classes
- Noise Alignment stage (PCA Gaussian denoisers) — orthogonal technique
- U-Net experiments with real metrics (FID)

**What we have that they don't:**
- THEORY — they have none, we can provide the first theoretical framework
- DiT experiments — they only test U-Net
- All-layer steering (4× single-layer) — they only steer at one block
- Predictive-vs-causal analysis — they pick layers by probe accuracy (suboptimal)
- Concept type boundaries — we explain WHY semantic steering fails
- Quality/diversity analysis — their norm_scale formula DESTROYS diversity (0.098 ratio)
- Data efficiency — 50 samples sufficient
- Cross-architecture linearity (DiT + U-Net both show linear concepts)

**Key experimental result to highlight:**
- Their RFM beats their mean-diff on Kuvasz (30.5% vs 6.2%)
- Our RFM gives NEGATIVE lift (-0.05 to -0.11) while mean-diff gives +0.56
- This contradiction is explained by our theory: RFM optimizes predictive, not causal

---

## Primary Competitor

### Wang et al. 2026 — NA-RFM Steering
- **File:** `wang2026_narfm_steering.pdf`
- **Title:** "General and Efficient Steering of Unconditional Diffusion"
- **Authors:** Qingsong Wang, Mikhail Belkin, Yusu Wang (UC San Diego)
- **ArXiv:** 2602.11395 (Feb 2026)
- **Method:** Two-stage gradient-free steering: (1) Noise Alignment via PCA Gaussian denoisers for early steps, (2) RFM/AGOP concept vectors in U-Net activation space for later steps. Norm-scaling formula: h' = h + w * ||h|| * v.
- **Results:** CIFAR-10: 96.6% accuracy, FID 41.4. ImageNet (4 classes): 75.8% acc, FID 98. CelebA-HQ multi-attribute: 96% avg.
- **Architecture:** U-Net only (Improved DDPM, ADM, DDPM). No DiT/transformer experiments.
- **Theory:** NONE. Purely empirical. No explanation for why AGOP directions steer effectively.
- **Detailed method:**
  - RFM: Laplacian kernel, T=5 iterations, top eigenvector of AGOP. Bandwidth/reg tuned per dataset.
  - Norm scaling: h' = h + w_RFM * ||h|| * v_c (perturbation scales with activation norm)
  - Layer: last encoder block before bottleneck (8×8 resolution). Encoder-9 = 97.9%, Middle = 47.4%.
  - Two-stage: Noise Alignment (PCA Gaussian denoisers, early steps) + RFM (later steps).
  - Single fixed vector per class, trained at clean timestep, applied across full window.
  - DDIM 100 steps, eta=0.
- **RFM vs mean-diff ablation:** RFM beats mean-diff on Kuvasz (30.5% vs 6.2%) — mean-diff collapses to single template pose.
- **Key gaps:**
  - **No theoretical justification** — they say "it would be interesting to see why discriminative directions are effective for generation"
  - U-Net only, no DiT/transformers
  - Only 4 ImageNet classes tested (not scalable)
  - Fixed steering strength (no adaptive scheduling)
  - No analysis of when/why steering fails (Kuvasz 30.5% unexplained)
  - No concept type boundaries (which concepts are steerable?)
  - Per-class hyperparameter tuning (15+ param combos across experiments)
  - Forward-process activations only
  - No all-layer analysis — single block steering only
  - No predictive-vs-causal analysis
  - No compositionality study beyond linear combination

## Theoretical Foundations

### Ren et al. 2026 — DriftLite
- **File:** `ren2026_driftlite.pdf`
- **Title:** "DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models"
- **Authors:** Yinuo Ren, Wenhao Gao, Lexing Ying, Grant M. Rotskoff, Jiequn Han
- **ArXiv:** 2509.21655 (Sep 2025, ICLR 2026)
- **Method:** Exploits Fokker-Planck degree of freedom: any drift b_t(x) can be added with compensating residual potential. Proposes VCG (Variance-Controlling Guidance) and ECG (Energy-Controlling Guidance) — solve 3×3 linear system per step for optimal drift coefficients.
- **Results:** GMMs, particle systems, protein-ligand co-folding. VCG-SMC beats G-SMC dramatically on ESS and mode coverage.
- **Theory:** Proposition 3.1 (drift-potential equivalence), Proposition 3.2 (optimal curl-free control exists). Variational formulation via Poisson equation.
- **Relevance:** Their linear ansatz b_t = Σ θ_i s_i is analogous to our multi-vector steering. But they work in INPUT space, we work in ACTIVATION space.

### Marks & Tegmark 2024 — Geometry of Truth
- **File:** `marks2024_geometry_truth.pdf`
- **Title:** "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Statements"
- **Authors:** Samuel Marks, Max Tegmark (MIT)
- **ArXiv:** 2310.06824 (ACL 2024)
- **Key finding:** Mean-difference directions are MORE causally effective than logistic regression probes (NIE 0.85-1.03 vs 0.05-0.61). The maximum-margin separator doesn't align with the model's causal pathways.
- **Relevance:** We extend this predictive-vs-causal gap finding from LLMs to diffusion models. Our data shows mean-diff > logreg for causal steering in DiT, paralleling their result.

### Beaglehole et al. 2024 — AGOP Features
- **File:** `beaglehole2024_agop_features.pdf`
- **Title:** "Average gradient outer product as a mechanism for deep neural collapse" (or related AGOP/NFA work)
- **ArXiv:** 2212.14855
- **Key finding:** W^T W ∝ AGOP (the Neural Feature Ansatz). Justifies using AGOP eigenvectors as concept directions.
- **Relevance:** Our Stage 0 shows NFA FAILS in DiTs (cosine 0.031). This is a key negative result differentiating DiTs from classifiers.

## Activation Steering in Diffusion

### Kwon et al. 2023 — h-space
- **File:** `kwon2023_hspace.pdf`
- **Title:** "Diffusion Models Already Have a Semantic Latent Space"
- **Authors:** Mingi Kwon, Jaeseok Jeong, Youngjung Uh
- **ArXiv:** 2210.10960 (ICLR 2023)
- **Method:** Identifies h-space (U-Net bottleneck activations) as a semantic latent space. Edits via h' = h + Δh for image manipulation.
- **Key finding:** h-space edits are more disentangled and consistent than latent-space edits. Only steer first ~30% of denoising steps.
- **Relevance:** Foundational work for activation steering in diffusion. Our approach extends beyond single bottleneck to all-layer steering.

### Li et al. 2024 — AcT Steering
- **File:** `li2024_act_steering.pdf`
- **Title:** (Activation steering for text-to-image diffusion)
- **ArXiv:** 2404.01986
- **Relevance:** Another activation steering approach, empirical, no control-theoretic framework.
