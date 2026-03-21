# autoresearch — LASD: Linear Activation Steering for Diffusion

This is an autonomous research program for investigating whether Recursive Feature Machines (RFM) can extract linear concept directions from Diffusion Transformer activations for zero-cost steering.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `proposal.tex` — the research proposal with method description, experiment plan, and expected results.
   - `extract.py` — the RFM concept vector extraction pipeline. You modify this.
   - `steer.py` — the steering evaluation pipeline. You modify this.
   - `config.py` — fixed constants: model paths, prompt datasets, concept oracles, evaluation metrics. Do not modify.
   - `eval.py` — fixed evaluation harness: FID, concept accuracy, CLIP score, LPIPS diversity. Do not modify.
4. **Verify models exist**: Check that the DiT checkpoint (DiT-XL/2) and concept classifiers (aesthetic predictor, CLIP, NudeNet) are downloaded. If not, tell the human to run `python setup.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Architecture

The codebase has a clean separation:

```
extract.py    — YOU MODIFY: activation collection, RFM pipeline, concept vector extraction
steer.py      — YOU MODIFY: steering application, timestep strategies, spatial modes
config.py     — READ ONLY: model configs, dataset paths, concept definitions, hyperparameter ranges
eval.py       — READ ONLY: FID computation, concept accuracy, CLIP score, LPIPS diversity, cost measurement
```

**What you CAN do:**
- Modify `extract.py` — activation hooks, pooling strategies, RFM configuration, AGOP computation, extraction method.
- Modify `steer.py` — steering injection points, timestep strategies (A/B/C), spatial modes, epsilon schedules, layer selection, multi-concept composition.

**What you CANNOT do:**
- Modify `config.py` or `eval.py`. They contain the fixed evaluation harness and ground truth metrics.
- Install new packages. Only use what's in `pyproject.toml`.
- Modify the evaluation metrics. The functions in `eval.py` are the ground truth.

## The Research Stages

The project has a staged structure with go/no-go gates. You advance through stages sequentially. Each stage has a clear objective, success criterion, and failure action.

### Stage 0: NFA Validation
**Goal**: Does W^T W ∝ AGOP hold in DiTs?
**Run**: `python extract.py --mode nfa_validate --model dit-xl2 --timesteps 100,300,500,700,900`
**Metric**: `nfa_cosine` — cosine similarity between Neural Feature Matrix and AGOP at each (layer, timestep).
**Go**: nfa_cosine > 0.7 at majority of (layer, timestep) pairs.
**No-go**: nfa_cosine < 0.5 everywhere → pivot to empirical probing without NFA justification (skip to Stage 1 directly, note the negative result).

### Stage 1: Linearity Probing + Temporal Stability
**Goal**: Are concepts linearly decodable? Are concept directions stable across timesteps?
**Run**: `python extract.py --mode probe --model dit-xl2 --concepts aesthetic,style,color,object --n_samples 5000`
**Metrics**:
  - `probe_acc` — classification accuracy of RFM/linear/MLP probes per (concept, layer, timestep).
  - `stability` — temporal stability S(C, l) = mean pairwise cosine similarity of concept vectors across timesteps.
**Go**: probe_acc > 80% for at least 3/4 concepts at best (layer, timestep); stability > 0.5 for global concepts.
**No-go**: probe_acc < 60% everywhere → LASD cannot work; pivot to DiT representation analysis paper.

### Stage 2: Steering Evaluation
**Goal**: Do extracted concept vectors actually steer generation?
**Run**: `python steer.py --mode evaluate --model dit-xl2 --baselines cfg,fk4,meandiff,pca,logreg`
**Metrics**:
  - `concept_acc` — fraction of steered images classified as concept-positive.
  - `fid` — FID vs. unsteered reference.
  - `clip_score` — text-image alignment.
  - `lpips_div` — diversity (LPIPS variance across generations).
  - `cost_ratio` — FLOPs per steered generation / FLOPs per unsteered.
**Go**: concept_acc within 15% of FK(k=4) at cost_ratio ≈ 1.0.

### Stage 3: Ablations
**Goal**: Find optimal configuration.
**Experiments**: epsilon sweep, layer selection, timestep strategy (A vs B vs C), data efficiency (M), RFM iterations (R), extraction method comparison.
**Run**: `python steer.py --mode ablation --sweep <param>`

### Stage 4: Composition + Scale
**Goal**: Multi-concept steering, LASD+FK hybrid, cross-prompt generalization, scale to larger models.
**Run**: `python steer.py --mode compose --concepts "aesthetic+watercolor+warm"`

## Output format

Each experiment prints a summary:

```
---
concept_acc:      0.823
fid:              12.4
clip_score:       0.312
lpips_div:        0.456
cost_ratio:       1.002
probe_acc:        0.891
nfa_cosine:       0.943
stability:        0.782
peak_vram_mb:     24500.0
wall_seconds:     180.3
```

Not all fields appear in every run — only those relevant to the current stage. Extract key metrics:

```
grep "^concept_acc:\|^fid:\|^probe_acc:\|^nfa_cosine:\|^stability:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	stage	metric	value	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. stage: `S0`, `S1`, `S2`, `S3`, `S4`
3. primary metric name for this experiment (e.g. `nfa_cosine`, `probe_acc`, `concept_acc`, `fid`)
4. primary metric value (e.g. 0.943) — use 0.000 for crashes
5. peak memory in GB, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, `crash`, or `gate_pass` / `gate_fail`
7. short text description of what this experiment tried

Example:

```
commit	stage	metric	value	memory_gb	status	description
a1b2c3d	S0	nfa_cosine	0.943	24.0	gate_pass	NFA validation on DiT-XL/2 - holds at mid/late layers
b2c3d4e	S1	probe_acc	0.872	24.2	gate_pass	linear probes on aesthetic concept - strong linearity
c3d4e5f	S1	stability	0.781	24.0	keep	temporal stability for aesthetic - shared vector viable
d4e5f6g	S2	concept_acc	0.734	24.5	keep	RFM steering with strategy A eps=0.1
e5f6g7h	S2	concept_acc	0.756	24.5	keep	RFM steering with strategy B bins=4
f6g7h8i	S2	concept_acc	0.612	24.5	discard	mean-difference baseline - weaker than RFM
g7h8i9j	S2	fid	45.200	24.5	discard	eps=2.0 too large - artifacts
h8i9j0k	S3	concept_acc	0.000	0.0	crash	attention-weighted spatial mode OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar20`).

LOOP FOREVER:

1. Look at the current stage and what's been tried so far (read `results.tsv`).
2. Decide what to try next:
   - Within a stage: try the next logical experiment (vary one thing at a time).
   - At a go/no-go gate: evaluate the gate condition. If pass, advance to next stage. If fail, log `gate_fail` and follow the pivot instructions.
3. Modify `extract.py` or `steer.py` with the experimental change.
4. git commit with a descriptive message.
5. Run the experiment: `python <script>.py <args> > run.log 2>&1` (redirect everything — do NOT let output flood your context).
6. Read out results: `grep "^<relevant_metric>:" run.log`
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback. Fix if trivial; otherwise log crash and move on.
8. Record results in `results.tsv` (do NOT commit results.tsv — leave it untracked).
9. If the result is an improvement or a gate_pass, keep the commit (advance the branch).
10. If the result is worse or unhelpful, `git reset --hard HEAD~1` to revert.

## Experiment strategy

**Within each stage, follow this order:**

### Stage 0 experiments (expect ~2-3 runs):
1. NFA validation on DiT-XL/2 at 5 representative timesteps.
2. If marginal, try pre-adaLN vs. post-adaLN activation space.

### Stage 1 experiments (expect ~10-15 runs):
1. Linear probes for each concept at the best (layer, timestep) from NFA validation.
2. Full (layer × timestep) sweep for 2 key concepts (aesthetic, style).
3. Compare probes: RFM vs. linear vs. MLP (measure linearity gap).
4. Temporal stability analysis: pairwise cosine similarity heatmap.
5. Determine: Strategy A sufficient, or need Strategy B? What bin count?

### Stage 2 experiments (expect ~15-20 runs):
1. First steering attempt: best concept, best layer, Strategy A, eps=0.1.
2. Sweep eps: 0.05, 0.1, 0.2, 0.5, 1.0 — find the Pareto frontier.
3. Baselines: mean-difference, PCA, logistic regression weight — same eps.
4. FK steering (k=4) as upper bound.
5. CFG as practical baseline.
6. Second concept. Third concept. Build up steerability statistics.
7. Strategy B with best bin count from Stage 1.

### Stage 3 experiments (expect ~10-15 runs):
1. Layer selection: all vs. middle vs. single-best vs. last-10.
2. Data efficiency: M = 100, 200, 500, 1000, 5000.
3. RFM iterations: R = 1, 2, 3, 5.
4. Spatial mode: uniform vs. attention-weighted.

### Stage 4 experiments (expect ~5-10 runs):
1. Two-concept composition. Three-concept composition.
2. Negative steering (eps < 0) for concept suppression.
3. LASD + FK(k=2) hybrid vs. FK(k=8).
4. Cross-prompt generalization test.
5. If available: scale test on SD3 or Flux.

**Total: ~50-65 experiments across all stages.**

## Key principles

**One variable at a time.** Change one thing per experiment. If you change both the layer and the epsilon, you won't know which mattered.

**Compare to the right baseline.** Every steering result should be compared to (a) no steering and (b) the simplest extraction method (mean-difference). If RFM doesn't beat mean-difference, the contribution is weak.

**Log negative results.** A concept where steering fails is as informative as one where it succeeds. Record everything.

**Watch FID.** If FID degrades by >20% relative to unsteered, the steering is too aggressive regardless of concept accuracy. Back off epsilon.

**The Pareto frontier matters.** The key figure is concept_acc vs. FID at varying eps. Plot this mentally as you go. The best eps is where you get maximum concept improvement with minimal FID degradation.

**Simplicity criterion**: All else being equal, simpler is better. Strategy A (shared vector) beating Strategy B (time-binned) is a great outcome — it means the approach is even simpler than expected. A finding that mean-difference matches RFM is also valuable (it means you don't need the AGOP machinery). Don't over-complicate to squeeze marginal gains.

**Timeout**: Extraction experiments (Stage 0-1) should take ~5-15 minutes each. Steering evaluation (Stage 2-4) should take ~10-30 minutes each (depending on number of generated images). If a run exceeds 60 minutes, kill it and treat as a crash.

**Crashes**: If a run OOMs, try reducing batch size or number of samples before abandoning. If it's a code bug, fix and re-run. If the approach is fundamentally too expensive, log it and move on.

**NEVER STOP**: Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you finish all planned experiments in a stage, advance to the next stage. If you finish all stages, go back and try combinations of the best findings, or test on new concepts, or explore edge cases. The loop runs until the human interrupts you.

As a rough estimate: if each experiment averages ~15 minutes, you can run ~4/hour, ~48 over a 12-hour session. The human wakes up to a full results.tsv spanning Stages 0-3, with the key questions (does NFA hold? is linearity strong? does steering work? does RFM beat baselines?) all answered.
