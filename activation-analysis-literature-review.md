# Activation-Level Analysis, Probing, and Manipulation in Diffusion Models
## Comprehensive Literature Review

---

## 1. Diffusion Models Already Have a Semantic Latent Space (Kwon et al., ICLR 2023 Oral)

**Paper**: [arXiv:2210.10960](https://arxiv.org/abs/2210.10960) | [Code](https://github.com/kwonminki/Asyrp_official)

### Experimental Setup
- **Models**: DDPM++, iDDPM, ADM (unconditional diffusion models)
- **Datasets**: CelebA-HQ, AFHQ-dog, LSUN-church, LSUN-bedroom, METFACES

### What is h-space?
- **Definition**: h-space is the set of **bottleneck feature maps** of the U-Net across all timesteps. Each bottleneck activation h_t has lower spatial dimension but more channels than the output image.
- **Location**: The output of the U-Net's bottleneck (middle block), before the decoder/upsampling path begins.

### How They Extract Concept Directions
- **Optimization-based**: A neural network f_t predicts augmentations Δh_t to bottleneck features
- **Loss function**: λ_CLIP · L_direction + λ_recon · |P_edit_t - P_ref_t|
  - **Directional CLIP Loss**: Ensures the editing direction in CLIP image space is parallel to the direction between reference and target text embeddings
  - **Reconstruction Loss**: Maintains fidelity to the original image
- The method does NOT use PCA or mean-difference; it uses gradient-based optimization of Δh with CLIP guidance

### The Asymmetric Reverse Process (Asyrp)
- Modifies only the predicted-x0 component (P_t) while keeping the direction-toward-xt component (D_t) unchanged:
  - x_{t-1} = √α_{t-1} · P_t(ε̃_θ(x_t, t | Δh_t)) + D_t(ε_θ(x_t, t)) + σ_t · v_t
- This allows semantic editing via P_t while preserving structural/quality information via D_t

### Timestep Dependence
- **Editing window**: First ~30% of timesteps during reverse denoising (high-level semantics)
- **Quality boosting**: Last ~30% of timesteps (low-level details/quality)
- Editing at early (high-noise) timesteps captures semantic meaning; later timesteps handle fine detail

### Metrics
- **Directional CLIP Similarity (S_dir)**: Computed as 1 - L_direction; measures alignment between image edit direction and text direction
- **Segmentation Consistency (SC)**: mIoU between reference and edited segmentation masks
- **FID Score**: Distribution distance between edited and reference images

### Geometry of h-space
- **Homogeneity**: One h_t optimized for an image produces the same attribute change on other images
- **Linearity**: Editing effects scale continuously; opposite direction produces semantically opposite results
- **Robustness**: Stable across different input images
- **Consistency across timesteps**: The same semantic meaning persists through the denoising process
- **Compositionality**: Combinations of different Δh vectors yield their combined semantic changes

---

## 2. Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models (Haas et al., 2023)

**Paper**: [arXiv:2303.11073](https://arxiv.org/abs/2303.11073) | [Code](https://github.com/renhaa/semantic-diffusion)

### Experimental Setup
- **Model**: DDPM trained on CelebA-HQ (256×256); also tested on churches and bedrooms
- **Approach**: Unsupervised PCA-based direction discovery in h-space (no CLIP, no classifiers, no fine-tuning)

### How They Extract Concept Directions (PCA)
1. Generate n random samples, saving bottleneck activations {h_t^(i)} for each sample i
2. **Vectorize activations at each timestep t independently**
3. Compute principal components via Incremental PCA at each timestep
4. Concatenate the j-th principal component across all timesteps to form editing direction v_j

### Editing Procedure
- h_t^(edit) = h_t + γ · v_t, where γ controls edit strength
- Offset Δh_t is injected into both P_t and D_t terms (unlike Asyrp which only modifies P_t)
- Requires only a single forward pass per step

### What Semantic Directions Emerge as Principal Components
- **PC1**: Pose (yaw/pitch/roll)
- **PC2**: Gender
- **PC3**: Age
- **PC4**: Smile
- These emerge entirely unsupervised as the dominant axes of variation in h-space

### PCA vs. Random Directions (Critical Finding)
- **PCA directions**: Smooth, semantically interpretable edits
- **Random directions (same norm)**: Only minor changes at small scales, rapid degradation at larger scales
- This demonstrates that semantic information is concentrated along specific axes, not uniformly distributed

### Quantitative Results
- CLIP-based zero-shot classification for 5 attributes across 100 samples
- **Sample efficiency**: h-space requires ~10 labeled examples for convergence vs 200+ for x_t-based editing
- **Sequential editing**: Multiple directions combine additively while maintaining semantic integrity
- Disentanglement via Gram-Schmidt orthogonalization improves attribute independence

### Key Geometry Findings
- h-space is NOT fully specified by h alone (background varies when swapping h_t between samples)
- Semantic information is concentrated along low-rank principal subspace
- Vector arithmetic works: difference between smiling/non-smiling h produces smile transfer

---

## 3. Self-Discovering Interpretable Diffusion Latent Directions for Responsible T2I Generation (Li et al., CVPR 2024)

**Paper**: [arXiv:2311.17216](https://arxiv.org/abs/2311.17216) | [Code](https://github.com/hangligit/InterpretDiffusion)

### Experimental Setup
- **Model**: Stable Diffusion v1.4, guidance scale 7.5
- **Approach**: Self-supervised direction discovery in h-space without external models (no CLIP, no attribute classifiers)

### How They Extract Concept Directions
- Generate images with concept-inclusive prompt y+, then reconstruct using concept-removed prompt y-
- Optimize a **learnable vector c** to compensate for missing concept information:
  - c* = argmin_c Σ ||ε - ε_θ(x_t+, t, π(y-), c)||²
- 10K optimization steps on 1K synthesized images per concept
- Gradients only update the learnable vector; model is frozen

### Concepts Discovered
- **Fairness**: Gender (male/female), racial attributes (Black/White/Asian)
- **Safety**: Anti-sexual, anti-violence, anti-harassment, anti-hate, anti-illegal activity, anti-self-harm, anti-shocking
- **General semantics**: Running, wearing glasses, age

### Metrics
- **Deviation ratio (Δ)**: Attribute distribution imbalance across professions (fairness)
- **Inappropriate image proportion**: Via NudeNet and Q16 classifiers on I2P benchmark (safety)
- **CLIPScore**: Text-image alignment
- **FID**: Image quality

### Key Geometry Findings
- **Disentanglement**: Learned vectors manipulate single attributes while leaving others unchanged
- **Linearity**: h ← h + λc produces smooth interpolation between concepts
- **Compositionality**: Independently trained vectors combine additively (c_M = Σc_s)
- **Timestep consistency**: A single vector per concept works across ALL timesteps (no timestep-specific vectors needed)

---

## 4. Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models (Gandikota et al., ECCV 2024)

**Paper**: [arXiv:2311.12092](https://arxiv.org/abs/2311.12092) | [Code](https://github.com/rohitgandikota/sliders) | [Project](https://sliders.baulab.info/)

### Experimental Setup
- **Model**: Stable Diffusion v1.4 and SDXL
- **Approach**: Fine-tune low-rank (LoRA) adaptors to capture concept directions in weight space

### How They Extract Concept Directions
- **Not activation-space directions** -- this operates in **parameter space**
- Train LoRA adaptor using guided score: ε_θ*(X, c_t, t) = ε_θ(X, c_t, t) + η(ε_θ(X, c+, t) - ε_θ(X, c-, t))
- This shifts the distribution of target concept to exhibit more c+ attributes and fewer c- attributes
- Uses classifier-free guidance logic at **training time** rather than inference
- **LoRA rank**: Typically rank 4, alpha=1
- Low-rank constraint both for efficiency and to precisely isolate the edit direction

### Concept Pair Definition
- Text-based: "young person" (c-) vs "old person" (c+)
- Visual: Optimize LoRA from image pairs (x_A, x_B) using gradient differences
- **Preservation concepts**: Additional prompts (e.g., race-specific compositions) to avoid entanglement

### Slider Strength Control
- At inference: scale the LoRA adaptor output by a continuous factor
- Sliders are composable: multiple LoRA adaptors can be combined

### Metrics
- **CLIP scores**: Text-image alignment
- **LPIPS**: Perceptual distance measuring interference with non-target attributes
- Evaluated over 20 randomly sampled images per rank/concept/scale combination

### Relation to Activation Analysis
- Concept Sliders operate in weight space rather than activation space, but the insight is related: low-rank perturbations to model weights correspond to approximately linear directions in the function space of the denoiser, paralleling the linear structure found in h-space

---

## 5. Erasing Concepts from Diffusion Models (Gandikota et al., ICCV 2023)

**Paper**: [arXiv:2303.07345](https://arxiv.org/abs/2303.07345) | [Code](https://github.com/rohitgandikota/erasing) | [Project](https://erasing.baulab.info/)

### Experimental Setup
- **Model**: Stable Diffusion (exact version unspecified in available materials)
- **Approach**: Fine-tune model weights to permanently erase concepts using negative guidance as teacher

### Method (Erased Stable Diffusion / ESD)
- Frozen pre-trained model predicts noise for erasure prompt
- Edited model is trained to guide in the **opposite direction** using classifier-free guidance at training time
- Two variants:
  - **ESD-x**: Fine-tunes **cross-attention KV projections only** -- narrow erasure that activates only when concept is explicitly mentioned
  - **ESD-u**: Fine-tunes **unconditional/non-cross-attention layers** -- generalized erasure independent of prompt wording (useful for NSFW removal)

### Concept Specification
- Text-only: concept names specified via text prompts
- No image-based specification needed

### Applications Tested
- Erasing 5 modern artist styles
- Removing sexually explicit content
- User study for perception of removed styles

### Relation to Activation Analysis
- ESD-x demonstrates that **cross-attention layers encode concept-specific conditioning** that can be surgically modified
- ESD-u demonstrates that **unconditional generation pathways** encode broader concept knowledge
- The fact that different parameter subsets (cross-attention vs. non-cross-attention) control different aspects of concept expression reveals architectural structure in how concepts are represented

---

## 6. Diffusion Self-Guidance for Controllable Image Generation (Epstein et al., NeurIPS 2023)

**Paper**: [arXiv:2306.00986](https://arxiv.org/abs/2306.00986) | [Project](https://dave.ml/selfguidance/)

### Experimental Setup
- **Model**: **Imagen** (Google), generating 1024×1024 images
- **Approach**: Use internal U-Net representations as guidance signals during sampling (zero-shot, no training)

### Which Activations Encode What
- **Cross-attention maps** (36 total across encoder, bottleneck, decoder at 8×8, 16×16, 32×32 resolutions):
  - Object **position** (centroid): centroid(k) = weighted spatial average of attention map A
  - Object **size**: fraction of above-threshold attention pixels
  - Object **shape**: thresholded binary attention mask
- **Decoder activations** (penultimate layer before prediction readout):
  - Object **appearance**: attention-masked average of spatial activations

### Mathematical Formulation
- Position: centroid(k) = (1/ΣA) × [Σw·A, Σh·A]^T, guided via L1 loss
- Size: size(k) = (1/HW) × ΣA_thresh, using soft sigmoid thresholding
- Appearance: appearance(k) = Σ(shape(k) ⊙ Ψ) / Σshape(k), with stop-gradient on attention

### Timestep Schedule
- First 3N/16 steps: self-guidance applied
- Last N/32 steps: no guidance
- Remaining 25N/32 steps: alternating guidance on/off (N=1024 total steps)

### Key Findings About Representation Geometry
- **Entanglement**: "Appearance features often contain undesirable information about spatial layout" -- position and appearance are NOT fully disentangled
- **Cross-object coupling**: "Correlations in attention maps prevent fully disentangled control between interacting objects"
- **No formal linearity analysis** -- the paper uses gradient-based guidance rather than testing linear separability
- Evaluation is purely qualitative (no quantitative metrics reported for steering quality)

---

## 7. Label-Efficient Semantic Segmentation with Diffusion Models (Baranchuk et al., ICLR 2022)

**Paper**: [arXiv:2112.03126](https://arxiv.org/abs/2112.03126) | [Code](https://github.com/yandex-research/ddpm-segmentation)

### Experimental Setup
- **Model**: Pretrained DDPM (unconditional), U-Net architecture
- **Task**: Use intermediate activations as pixel-level features for semantic segmentation

### Which Activations Were Probed
- Feature maps from U-Net **decoder blocks** (18 blocks total)
- Activations extracted at specific diffusion timesteps during a single forward pass on noisy input
- Selecting the ideal timestep and decoder block is non-trivial

### How Probes Were Trained
- **MLP ensemble**: Separate MLP trained per pixel in a pixel-wise manner
- Input: pixel feature map activations at different timesteps, rescaled as necessary
- Output: class label prediction per pixel
- This is the foundational "linear probe on diffusion activations" protocol referenced by subsequent work

### Key Findings
- Intermediate activations capture semantic information sufficient for segmentation, especially in the few-shot regime
- Outperforms alternatives in label-scarce settings
- Different layers and timesteps carry different levels of semantic information
- Established the methodology later extended by "Not All Diffusion Model Activations" (NeurIPS 2024)

---

## 8. Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features (NeurIPS 2024)

**Paper**: [arXiv:2410.03558](https://arxiv.org/abs/2410.03558) | [Code](https://github.com/Darkbblue/generic-diffusion-feature)

### Experimental Setup
- **Models**: Stable Diffusion v1.5, SDXL, Playground v2, VideoCrafter2
- **Note**: Authors explicitly state uncertainty about generalization to DiT models due to "markedly different architecture"

### Massive Activation Taxonomy Probed
- **Inter-module activations** (prior standard)
- **Self-attention queries and keys** (within ViT blocks)
- **Cross-attention queries** (from text-image fusion layers)
- **ViT block outputs** (complete transformer block activations)
- **Increment activations** (residual connection additions)
- **Upsampler/downsampler outputs**
- From SDXL alone: reduced candidates from **279 to 63** (78% reduction) via qualitative filtering

### Discriminative Tasks
- Semantic correspondence (SPair-71k, PCK@0.1 metric)
- Semantic segmentation (ADE20K, CityScapes, mIoU metric)
- Label-scarce segmentation (Horse-21, 30 labeled images)

### Three Universal Properties Discovered

**Property 1: Asymmetric Diffusion Noises**
- Noise decreases through down-stage, nearly absent in early up-stage, resurfaces in late up-stage
- Down-stage is heavily contaminated; early up-stage is cleanest

**Property 2: In-Resolution Granularity Changes**
- Modern "fat" U-Nets have fewer resolutions but enlarged blocks
- Information granularity varies substantially WITHIN single resolutions due to embedded ViT modules

**Property 3: Locality Without Positional Embeddings**
- Embedded ViTs lack positional embeddings
- Self-attention activations are generally consistent with spatial locality despite no explicit positional encoding

### Key Quantitative Finding
- SDXL **without** intra-ViT activation consideration: 66.00 PCK@0.1img (underperforms SDv1.5's 75.14)
- SDXL **with** ViT activations: 81.72 PCK@0.1img (substantially outperforms SDv1.5's 77.78)
- Best features: cross-attention queries and deep ViT block outputs from higher resolutions
- Self-attention keys suppress diffusion noise via spatial structure focus

---

## 9. ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features (ICML 2025)

**Paper**: [arXiv:2502.04320](https://arxiv.org/abs/2502.04320)

### Experimental Setup
- **Models**: **Flux-Schnell** (primary), Stable Diffusion 3.5 Turbo, CogVideoX (video)
- **Task**: Zero-shot image segmentation via DiT attention outputs
- **This is the most significant DiT interpretability work to date**

### Method: Linear Projections in DiT Output Space
1. Single-token concepts (e.g., "cat", "sky") encoded via T5
2. Concept embeddings processed through multi-modal attention (MMAttn) layers using K, Q, V projections
3. **One-directional attention**: concepts attend to image tokens but image/text tokens ignore concepts
4. **Critical innovation**: Compute dot-product similarity between image attention outputs (o_x) and concept attention outputs (o_c):
   - φ(o_x, o_c) = softmax(o_x · o_c^T)
5. This **output-space projection** yields far sharper saliency maps than cross-attention maps

### Which Layers Are Most Interpretable
- **Later MMAttn layers encode richer features**
- Progressively improving performance from shallow to deep layers
- **Last 10 of 18 MMAttn layers** used in final experiments
- Combining all layers outperforms any individual layer

### Benchmarks and Results
- **ImageNet-Segmentation** (4,276 images, 445 categories): 83.07% Acc / 71.04% mIoU (Flux)
- **PascalVOC 2012**: Single-class (930 images) and multi-class (1,449 images)
- Outperforms 15 baseline methods including DAAM, CLIP-based methods, and DINO variants
- Metrics: Pixelwise Accuracy, mIoU, mAP

### Timestep-Dependent Findings (Critical)
- **Optimal segmentation requires SOME noise**: timestep 500 performs best, NOT timestep 0
- Intermediate noise levels preserve semantic structure while maintaining generative activation patterns
- This is counterintuitive and important for any activation probing work on diffusion models

### DiT vs U-Net Comparison
- U-Net: shallow cross-attention between fixed prompt tokens and image patches; vocabulary limited to tokens in prompt
- DiT: prompt embeddings evolve through layers; open-vocabulary querying; access to attention output representations
- Performance gap: ConceptAttention (Flux: 83.07% Acc) >> DAAM on SDXL (79.41%) >> SD2 (64.52%)

### Key Geometry Finding
- **Output space representations substantially outperform cross-attention or value spaces** for segmentation
- Both cross-attention and self-attention together maximize performance; either alone underperforms
- DiT representations are highly transferable to vision tasks

---

## 10. Latent Space Disentanglement in Diffusion Transformers (2024)

**Paper**: [arXiv:2408.13335](https://arxiv.org/abs/2408.13335)

### Experimental Setup
- **Model**: **Stable Diffusion V3** (MM-DiT architecture), compared with UNet-based SD v2.1
- **Approach**: Analyze and exploit disentanglement properties unique to DiT architecture

### Key Architecture Insight
- DiTs concatenate image and text embeddings, allowing joint processing through self-attention
- This creates LESS entangled semantic representations than UNet cross-attention

### Disentanglement Metric (SDE)
- **Semantic Disentanglement mEtric**: Measures decomposability and effectiveness via image reconstruction distances under different text conditions
- Attention classification accuracy across semantic tokens shows "near 0.5" classification ratios for DiT, indicating minimal cross-semantic leakage

### Editing Procedure (Extract-Manipulate-Sample)
- **Linear manipulation**: c̃ = c + α·n, where n = c₁ - c₀ is the editing direction
- **Proposition 1**: Editing directions behave as unit vectors with concentration-of-measure guarantees
- **Proposition 2**: Extended editing directions for different semantics form **orthogonal sets** in m·d dimensional space

### Evaluation Metrics
- SDE (Semantic Disentanglement mEtric)
- MLLM-VQA (GPT-4 assessment of gradual semantic changes)
- Background preservation: PSNR, LPIPS, SSIM
- Semantic consistency: CLIPScore

### Key Finding
- **Neither text (C) nor image (Z_t) latent spaces alone satisfy effectiveness** -- the combined space Z_t ⊕ C achieves both decomposability and editing efficacy through bidirectional self-attention information flow
- DiT architecture provides inherently better disentanglement than U-Net

---

## 11. Latent Space Editing in Transformer-Based Flow Matching (Hu et al., AAAI 2024)

**Paper**: [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/27998) | [Code](https://github.com/dongzhuoyao/uspace)

### Key Contribution
- Introduces **u-space** for editing in transformer-based flow matching models (U-ViT architecture)
- u-space is controllable, accumulative, and composable
- Demonstrates that flow matching + transformer backbone latent structures support editing similar to DDPM h-space
- Editing becomes as easy as replacing, removing, or appending prompts

---

## 12. Beyond Surface Statistics: Scene Representations in a Latent Diffusion Model (Chen et al., 2023)

**Paper**: [arXiv:2306.05720](https://arxiv.org/abs/2306.05720) | [Code](https://github.com/yc015/scene-representation-diffusion-model)

### Experimental Setup
- **Model**: Stable Diffusion (2D, pretrained, no explicit depth information in training)
- **Activations probed**: Self-attention layer activations during intermediate denoising

### How Linear Probes Were Trained
- Trainable weight matrices project intermediate activations to predict image properties
- Search for a **linear axis** inside activation space that best aligns with target property

### Scene Attributes Probed
- **3D depth** (despite model never seeing depth data)
- **Foreground/background distinction** (salient object detection)

### Activation Intervention
- Modified intermediate activations using the projection learned by probe
- Changed pixel foreground/background properties WITHOUT altering model weights, latent vectors, seeds, or prompts
- Demonstrates causal (not merely correlational) role of linear directions

### Key Findings
- Representations are **linearly separable and decodable** via simple linear classifiers
- Representations emerge **surprisingly early in the denoising process** -- well before a human can make sense of the noisy images
- The model encodes 3D scene structure (depth) despite being trained only on 2D images

---

## 13. Do Diffusion Models Learn Semantically Meaningful and Efficient Representations? (2024)

**Paper**: [arXiv:2402.03305](https://arxiv.org/abs/2402.03305)

### Experimental Setup
- **Model**: Conditional DDPM with standard U-Net (3 down/up blocks, self-attention, skip connections)
- **Dataset**: Synthetic 2D Gaussian bumps on 32×32 images, parameterized by (μ_x, μ_y) position
- **Layer probed**: Layer 4 (late encoder, before bottleneck)

### Probing Method
- UMAP reduction to 3D embeddings
- 1D linear regressions on UMAP embeddings to predict x, y positions
- R² values as metric for representation quality

### Three Phases of Representation Learning
- **Phase A** (early): No manifold structure; generation failures include missing/multiple/mislocated bumps
- **Phase B** (intermediate): 2D or quasi-2D but unordered manifold; wrong-location failures dominate
- **Phase C** (terminal): Ordered 2D manifold with correct generation

### Key Geometry Findings
- Models develop **ordered 2D manifolds** corresponding to 2 independent semantic dimensions
- **Factorization is incomplete**: When trained with imbalanced data (d_x=0.1 vs d_y=1.0), x and y learning rates remain coupled rather than independent
- Representations become "semantically meaningful" but the paper remains **ambiguous on intrinsic linearity** (uses nonlinear UMAP + linear regression)
- Avoids bottleneck layer due to "diminishing signals"

---

## 14. Exploring the Latent Space of Diffusion Models Directly Through SVD (2025)

**Paper**: [arXiv:2502.02225](https://arxiv.org/abs/2502.02225)

### Key Contribution
- Applies SVD to latent codes directly and discovers that **representative information of attributes is captured by singular vectors**
- Fine-grained attributes (colors, textures) change along decreasing singular values at earlier timesteps
- Framework learns arbitrary attributes from one pair of latent codes defined by text prompts in Stable Diffusion
- No data collection requirements; maintains identity fidelity

---

## 15. Activation Transport (AcT) for Diffusion Models (ICLR 2025 Spotlight)

**Paper**: [Apple ML Research](https://machinelearning.apple.com/research/transporting-activations)

### Experimental Setup
- **Models**: SDXL-Lightning, FLUX.1.dev, FLUX.1.Schnell
- **Approach**: Optimal transport theory applied to activation distributions

### How Steering Vectors Are Computed
- Learn the optimal transport map between source and target activation distributions
- **Linear-AcT** simplifies to: (1) independent 1D map per neuron, (2) linear maps
- Ensures transported activations comply with target distribution (stays in-distribution)

### Which Layers Are Intervened On
- **LayerNorm layers** across all models
- Intervening on "all layernorm layers" for demonstrations

### Concepts Steered
- Content addition/removal (incrementally adding trees to scenes)
- Artistic style (anime, cyberpunk, watercolor)
- Negation handling (removing unwanted concepts)

### Key Geometry Finding
- Prior steering methods (ActAdd, mean-difference) can **shift activations out of distribution**, disrupting model dynamics
- AcT addresses this by respecting activation distributions via optimal transport
- Explicit control via λ between 0 (no transport) and 1 (full transport)

---

## 16. SteerDiff: Steering towards Safe Text-to-Image Diffusion Models (2024)

**Paper**: [arXiv:2410.02710](https://arxiv.org/abs/2410.02710)

### Method
- Operates in **text embedding space** (not internal U-Net activations)
- Identifies inappropriate concepts within text embeddings
- Applies **linear transformation** to steer embeddings toward safer content
- No additional training required; operates as intermediary between prompt and diffusion model
- Benchmarked against multiple red-teaming strategies

---

## 17. Understanding Diffusion Models: A Unified Perspective (Luo, 2022)

**Paper**: [arXiv:2208.11970](https://arxiv.org/abs/2208.11970) | [Blog](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html)

### Relevance to Interpretability
- This is primarily a **theoretical tutorial**, not an empirical interpretability study
- Derives VDMs as special case of Markovian Hierarchical VAE
- Shows that optimizing a VDM reduces to learning to predict one of three objectives: original input, original noise, or score function
- The ELBO can be decomposed into interpretable components
- Provides mathematical foundations but does NOT probe internal representations

---

## Cross-Paper Synthesis: Key Patterns

### 1. Where Semantic Information Lives

| Architecture | Key Location | Evidence |
|---|---|---|
| U-Net (DDPM/iDDPM/ADM) | Bottleneck (h-space) | Kwon, Haas, Li -- PCA/optimization find interpretable directions |
| U-Net (Stable Diffusion) | Decoder self-attention + bottleneck | Baranchuk, Chen -- linear probes decode semantics |
| U-Net (Imagen) | Cross-attention maps + penultimate decoder layer | Epstein -- position/size in attention, appearance in activations |
| U-Net (SD/SDXL) | Deep ViT block outputs + cross-attention queries | "Not All Activations" -- systematic evaluation of all activation types |
| DiT (Flux/SD3) | Later MMAttn output space (last 10/18 layers) | ConceptAttention -- output projections yield sharp saliency |
| DiT (SD3) | Joint text-image self-attention space | Latent Space Disentanglement -- orthogonal semantic subspaces |

### 2. Methods for Extracting Concept Directions

| Method | Papers | Pros | Cons |
|---|---|---|---|
| **CLIP-guided optimization** | Kwon (Asyrp) | Works for arbitrary text-described concepts | Requires CLIP; optimization per edit |
| **PCA (unsupervised)** | Haas et al. | No external models; discovers dominant axes | Limited to dominant modes; requires interpretation |
| **Self-supervised reconstruction** | Li et al. (CVPR 2024) | No external models; arbitrary concepts | Requires paired prompts; 10K optimization steps |
| **LoRA fine-tuning** | Gandikota (Sliders) | Plug-and-play; composable; weight-space | Not activation-space; requires training |
| **Negative guidance teaching** | Gandikota (ESD) | Permanent erasure; no inference overhead | Destructive; cannot undo |
| **Linear probes (supervised)** | Baranchuk, Chen | Clean interpretation; causal intervention | Requires labeled data |
| **Optimal transport** | AcT (Apple) | Respects activation distributions | Requires contrastive examples |
| **Output-space projection** | ConceptAttention | Open-vocabulary; DiT-specific | Requires MM-DiT architecture |
| **SVD of latent codes** | 2025 SVD paper | No data collection needed | Newer, less validated |

### 3. Geometry of Activation Space: Consensus Findings

- **Linearity is the dominant finding**: Nearly all papers report that semantic directions in h-space/activation space are approximately linear
  - PCA directions are interpretable; random directions degrade (Haas)
  - Linear probes successfully decode scene properties (Baranchuk, Chen)
  - Editing via linear interpolation h + λ·v works smoothly (all h-space papers)
  - DiT semantic directions form orthogonal subspaces (Latent Space Disentanglement)

- **Layer dependence is strong**:
  - Bottleneck carries high-level semantics; decoder layers carry progressively more spatial detail
  - DiT later layers more interpretable than early layers
  - Different U-Net stages (down/bottleneck/up) have very different noise contamination levels

- **Timestep dependence is significant**:
  - Early reverse steps (high noise) = high-level semantics
  - Late reverse steps (low noise) = fine details/quality
  - For probing, intermediate timesteps (~500/1000) are optimal (ConceptAttention)
  - h-space directions are approximately consistent across timesteps (Li et al.)

- **Entanglement persists**:
  - Appearance and position are not fully disentangled (Epstein)
  - Naive concept directions can be entangled with other attributes (Sliders uses preservation concepts)
  - DiT architecture provides better disentanglement than U-Net (SD3 paper)

### 4. Open Questions and Gaps

1. **DiT interpretability is nascent**: ConceptAttention (2025) and the SD3 disentanglement paper are the main works. No systematic probing study equivalent to "Not All Activations" exists for DiT.
2. **No mechanistic interpretability**: Unlike LLM work (e.g., sparse autoencoders, circuit analysis), there is no circuit-level understanding of how diffusion models compose features.
3. **Representation engineering for diffusion**: AcT (Apple, ICLR 2025) is the closest to LLM-style activation engineering applied to image generation, but the field is far behind LLM interpretability.
4. **Nonlinearity**: While linear methods work surprisingly well, no paper has rigorously tested whether nonlinear probes/directions significantly outperform linear ones.
5. **Flow matching models**: The u-space work (AAAI 2024) is the only systematic study of editing in flow-matching transformer architectures; more work needed for Flux/SD3.
