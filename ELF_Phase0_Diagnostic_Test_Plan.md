# Phase 0 Diagnostic Plan: Is a Non-Memoryless (ASBM-Style) Coupling Justified for ELF?

**Purpose of this document.** This is a runnable specification for a coding agent. It defines the
preliminary tests to complete **before** committing cluster time (Magus) or money to training a
modified ELF-B. None of these tests require training the full ELF model. They run on a laptop
(CPU or Apple MPS) or free-tier Colab, using only a frozen pretrained T5-small encoder and small
samples of text. Total estimated wall-clock time: **2-4 days of intermittent local compute**,
most of it unattended.

**Why these specific tests and not others.** Each test is designed to falsify or support one
specific load-bearing claim required for the interpolation project (Idea 1 in the prior review) to
be worth pursuing. If a test fails its go-criterion, it tells you *specifically* what part of the
plan needs to change (different coupling method, different bottleneck, or abandon the idea), rather
than a vague "it didn't work." Tests are grouped into three phases; later phases assume earlier ones
passed.

---

## How to Use This Document (for the code agent)

1. Implement each test as a standalone script under `phase0/test_XX_<name>.py`.
2. Each test writes its numeric results to `phase0/results/test_XX.json` and any plots to
   `phase0/results/test_XX_*.png`.
3. After each test, check its result against the stated **Go / No-Go criterion** before proceeding
   to the next. Do not skip ahead — later tests assume earlier artifacts exist (e.g., Test 4 reuses
   the embedding sample generated in Test 1).
4. At the end, populate the **Decision Log** table (Section 5) with actual results and the resulting
   recommendation.
5. Suggested libraries: `transformers` (T5-small), `torch` (CPU/MPS backend), `numpy`, `scipy`,
   `POT` (Python Optimal Transport, `pip install pot`), `scikit-dimension` (`pip install scikit-dimension`),
   `matplotlib`. No GPU is required for anything in Phase 0.

---

## Phase 0A — Embedding Geometry Diagnostics
*Question this phase answers: does ELF's embedding space actually show the kind of nonlinear
structure that a non-memoryless coupling would exploit — or is the straight-line path already
close to the best available path?*

These tests use **only** a frozen, off-the-shelf T5-small encoder on real text. No ELF training
involved. This is the cheapest and most decision-relevant phase — if it fails, nothing downstream
matters.

### Test 1 — Data Preparation and Embedding Extraction
**Rationale.** Every later test needs a consistent, reusable sample of clean embeddings `x` that
matches ELF's actual setup (T5-small encoder, embedding dim 512, 128-d bottleneck projection).
Using ELF's exact encoder configuration is what makes the rest of Phase 0 relevant to your actual
project rather than a generic embedding-geometry study.
**Method.**
- Sample ~5,000-10,000 sentences from a public OpenWebText-like source (e.g., a small `openwebtext`
  or `wikitext` slice from Hugging Face `datasets`, capped at a few hundred MB).
- Encode with frozen T5-small (`t5-small` from Hugging Face) to get 512-d contextual token embeddings.
- Apply a random (or PCA-initialized) linear projection to 128-d to emulate ELF's bottleneck,
  since ELF operates in the projected space, not the raw 512-d space.
- Save `x_512.npy` and `x_128.npy` (token-level, with sequence/position metadata).
**Go/No-Go.** N/A (setup step). Sanity check: embedding norms and pairwise distance distribution
should look non-degenerate (not collapsed to a point, not exploding).
**Est. local time.** 1-2 hours (mostly download + encoding).

### Test 2 — Straight-Line vs. Approximate Optimal-Transport Coupling
**This is the single most decision-relevant test in the whole plan.**
**Rationale.** ELF's linear interpolant implicitly pairs each noise sample with a data sample
*independently at random* (a memoryless coupling) and connects them by a straight line. A
non-memoryless coupling (ASBM-style) instead tries to pair noise and data points so that the
*connecting paths are as short/direct as possible in aggregate* — i.e., an optimal-transport-like
coupling. If, empirically, the OT-optimal pairing on a same-sized sample looks basically the same
as random pairing (in terms of average path length / curvature), then there is nothing for a
non-memoryless coupling to exploit at this scale, and the entire premise of Idea 1 needs to be
reconsidered.
**Method.**
- Take a subsample of ~1,000-2,000 clean embeddings `x` (128-d, from Test 1) and generate matched
  Gaussian noise samples `ε` of the same size.
- Compute (a) the average squared path length under **random/independent pairing** (ELF's current
  scheme): pair each `x_i` with a random `ε_j`, average `||x_i - ε_j||²`.
- Compute (b) the average squared path length under the **entropic-OT-optimal pairing** using
  Sinkhorn (`ot.sinkhorn` from POT) between the same `x` and `ε` sets.
- Report the ratio (b)/(a). Also visualize a 2D PCA/t-SNE projection with a handful of matched
  pairs from each scheme drawn as line segments, to sanity-check the numbers aren't a projection
  artifact.
**Go/No-Go criterion.**
- **Go (proceed to Phase 0B):** OT-optimal average path length is meaningfully lower than random
  pairing — a useful threshold is **≥15% reduction** in average squared path length. This says the
  independent/memoryless coupling is leaving real transport cost on the table, consistent with
  ASBM's Proposition 3.1 critique.
- **No-Go:** Reduction is small (<5-8%). This means random pairing is already close to optimal at
  this embedding scale/dimension, and a learned non-memoryless coupling is unlikely to produce a
  meaningful quality improvement — the interpolation project should be shelved in favor of the
  SDE/sampler project.
- **Ambiguous (5-15%):** Worth a closer look at Test 3 and Test 4 before deciding.
**Est. local time.** 2-4 hours (Sinkhorn on ~1-2k points is feasible on CPU; avoid going above
~5k points without a GPU, cost grows roughly quadratically in sample size).

### Test 3 — Local Curvature / Manifold Linearity Estimate
**Rationale.** Test 2 tells you whether *global* pairing matters. This test asks a complementary
question: locally, does the data manifold curve away from straight lines connecting nearby points
(which would justify a *curved path*, independent of *which* points get paired), or is it locally
flat? This matters because ASBM-style coupling and "use a curved interpolant" are related but
distinct fixes, and the diagnosis should tell you which one (or both) is worth pursuing.
**Method.**
- For a sample of ~500 clean embeddings, find each point's k-nearest-neighbors (k=10) in the
  128-d space.
- Fit a local linear (PCA) approximation of the manifold in each neighborhood and measure the
  residual variance not captured by the top few local principal components (a standard local
  curvature/manifold-flatness proxy).
- Separately, estimate global **intrinsic dimensionality** of the embedding cloud (e.g., via
  `scikit-dimension`'s MLE or TwoNN estimator) and compare it to the 128-d bottleneck — if the
  intrinsic dimension is much lower than 128, this both supports the "low-dimensional manifold"
  hypothesis ELF's own authors cite (Li & He, 2025, ref [32]) and gives a concrete target
  dimensionality for any redesigned coupling.
**Go/No-Go criterion.**
- **Supports nonlinear-path direction:** high local residual variance (manifold is locally
  curved) AND intrinsic dimension substantially below 128 (e.g., <60% of ambient bottleneck dim).
- **Does not support it:** local residual variance is low (manifold is locally close to flat/linear)
  and intrinsic dimension is close to 128 — this would corroborate a No-Go from Test 2.
**Est. local time.** 1-2 hours.

### Test 4 — Token-Level vs. Sequence-Level Coupling Structure
**Rationale.** ELF denoises **per token**, not per sequence. ASBM's non-memoryless coupling
argument needs to hold at the granularity ELF actually operates at. It's possible that
sentence-level embeddings show exploitable structure while individual token embeddings (which
ELF actually flows) look closer to independent/exchangeable — in which case a per-token coupling
gains little even if Test 2 looks favorable at the sequence level. This is exactly the
"dimensionality scale" obstacle flagged in the original research agenda, made concrete and testable.
**Method.**
- Repeat Test 2's OT-vs-random comparison, but computed **per token position** (e.g., group tokens
  by relative sequence position or by fixed small groups) rather than pooling entire sequences into
  one embedding.
- Compare the OT-improvement ratio at the token level vs. the sequence level from Test 2.
**Go/No-Go criterion.**
- **Go:** token-level OT-improvement ratio is within roughly half of the sequence-level ratio
  (i.e., the effect isn't purely a sequence-aggregation artifact).
- **No-Go / needs redesign:** token-level improvement is negligible even though sequence-level
  improvement was large — this means the coupling gain evaporates at the granularity ELF actually
  uses, and any redesign would need to operate at a coarser granularity than per-token, which is a
  significant architecture change beyond the original idea's scope.
**Est. local time.** 2-3 hours (reuses Test 2's code with a grouping change).

---

## Phase 0B — ASBM Feasibility & Engineering Diagnostics
*Question this phase answers: even if Phase 0A says the geometry justifies a coupling redesign, can
ASBM's specific coupling-estimation machinery actually run at ELF's scale (per-token, up to
1024-token sequences, 128-d bottleneck) without exploding in compute or requiring an unknown prior
energy function you can't specify?*

**Only run this phase if Phase 0A returned Go or Ambiguous.**

### Test 5 — ASBM Coupling-Estimation Convergence and Compute Scaling
**Rationale.** This directly targets the two obstacles already identified for this direction:
dimensionality scale and an unknown prior energy function. Before training anything, you need to
know whether the coupling-estimation step itself (independent of the flow network) is tractable.
**Method.**
- Implement (or adapt from ASBM's reference code, if available) the coupling/bridge-matching
  estimation step in isolation, as a pure numerical procedure — not inside a training loop.
- Run it on synthetic Gaussian-mixture data at increasing dimensionality (e.g., 16, 32, 64, 128)
  and increasing batch/sample size (e.g., 256, 1024, 4096 points), recording wall-clock time and
  peak memory at each setting.
- Fit a rough scaling curve (does cost grow linearly, quadratically, or worse in dimension and
  sample size?) and extrapolate to ELF's actual per-batch scale (batch size 512, sequences up to
  1024 tokens × 128-d, per ELF's Appendix D.2 hyperparameters).
**Go/No-Go criterion.**
- **Go:** extrapolated cost at ELF's real batch scale is a small fraction (<20%) of the flow
  network's own forward/backward pass cost — meaning coupling estimation won't dominate the
  training budget.
- **No-Go:** cost scaling is clearly super-linear and the extrapolated cost at real scale would
  dominate or double training time — this means the coupling step needs to be approximated,
  sub-batched, or computed at a coarser granularity (e.g., per-sequence instead of per-token)
  before it's usable, which is a scoping decision that needs to happen now, not mid-training-run
  on the cluster.
**Est. local time.** 3-6 hours (synthetic data, no encoder needed, but multiple runs across settings).

### Test 6 — Prior Energy Function Sensitivity
**Rationale.** The original research agenda explicitly flags "unknown prior energy function" as an
obstacle. If the choice of prior energy materially changes the resulting coupling, you need a
principled way to select or learn it before committing to a full training run — otherwise the
project has an unresolved hyperparameter with no clear default.
**Method.**
- On the same synthetic or small real-embedding setup from Test 5, run the coupling estimation
  under 2-3 plausible prior energy choices (e.g., a quadratic/Gaussian prior, an energy derived
  from the empirical noise distribution, and a uniform/flat prior as a control).
- Compare the resulting couplings (e.g., via the same average-path-length metric as Test 2) across
  prior choices.
**Go/No-Go criterion.**
- **Go:** results are qualitatively similar across reasonable prior choices (low sensitivity) —
  meaning you can pick a defensible default and move on.
- **No-Go / needs more design work:** results vary substantially by prior choice — meaning prior
  selection becomes a first-class research question in itself and should be explicitly scoped
  (and time-budgeted) as part of the project plan, not treated as a minor implementation detail.
**Est. local time.** 2-3 hours (reuses Test 5's harness).

---

## Phase 0C — Training Pipeline Readiness Tests
*Question this phase answers: does your actual training code work correctly at toy scale, and does
introducing the new coupling destabilize training — cheaply, before you find out on the cluster?*

**Run in parallel with Phase 0A/0B; not gated on their results**, since this phase is about code
correctness, not the scientific question.

### Test 7 — Toy-Scale Baseline (Linear Interpolant) Reproduction
**Rationale.** You cannot trust a comparison between "baseline" and "modified coupling" unless the
baseline code itself is verified correct first. This is the standard reproducibility check that
should happen before any modification, and it also de-risks the cluster run by catching bugs in a
30-minute local run instead of a 20-hour cluster job.
**Method.**
- Implement a deliberately tiny ELF-like model (e.g., 2-4 transformer layers, hidden dim 128,
  bottleneck dim 32) trained on a few thousand short sequences (e.g., a subset of Test 1's data)
  for a short number of steps.
- Verify: (a) the MSE denoising loss decreases over training, (b) the CE decode-branch loss
  decreases over training, (c) x-prediction and the shared-weight decode branch both run without
  shape/dtype errors, (d) a forward sampling pass (Euler ODE, per ELF's Algorithm 2) produces
  non-degenerate token sequences (not all-padding or all-one-token collapse).
**Go/No-Go criterion.** All four checks pass. If any fail, fix before proceeding — this is a
pure code-correctness gate, not a scientific one.
**Est. local time.** 2-4 hours including debugging.

### Test 8 — Toy-Scale Coupling Stability Probe
**Rationale.** The biggest named risk for the full project is that the modified coupling
destabilizes training. This test creates a cheap, fast-iterating environment (same toy scale as
Test 7) to catch that failure mode early and iterate on fixes, rather than discovering it on a
20-hour cluster job with no time left to recover.
**Method.**
- Using the same toy setup as Test 7, swap in the coupling scheme validated in Phase 0B (or a
  simplified version of it) in place of the linear interpolant.
- Train for the same short number of steps and compare: loss curve stability (no NaNs, no
  divergence), gradient norm behavior, and whether the CE decode-branch loss still decreases
  (this specifically checks the x-prediction/weight-sharing compatibility risk flagged earlier —
  if decode-branch loss stalls or diverges while it didn't in Test 7, that's a direct signal the
  new coupling breaks the shared-weight mechanism ELF depends on).
**Go/No-Go criterion.**
- **Go:** training is stable (no divergence) and the decode-branch loss decreases at a similar
  rate to Test 7's baseline.
- **No-Go:** divergence, NaNs, or a decode-branch loss that fails to decrease — this is a concrete,
  early signal to fix the coupling/CFG-compatibility math (see the "hidden assumption" about
  v = x - ε in the original review) before scaling up, not after.
**Est. local time.** 3-5 hours including iteration on fixes if Go fails on the first attempt.

---

## Decision Log (fill in after running all tests)

| Test | Result (fill in) | Go / No-Go | Notes |
|---|---|---|---|
| 1. Data prep | | N/A | |
| 2. OT vs. random coupling (**key test**) | | | |
| 3. Local curvature / intrinsic dim | | | |
| 4. Token- vs. sequence-level structure | | | |
| 5. ASBM compute scaling | | | |
| 6. Prior energy sensitivity | | | |
| 7. Toy baseline reproduction | | | |
| 8. Toy coupling stability | | | |

**Overall recommendation logic:**
- If **Test 2 is No-Go**, stop here regardless of other results — the core premise doesn't hold at
  this scale, and cluster time should go to the SDE/sampler project instead.
- If **Test 2 is Go/Ambiguous** but **Test 5 or 6 is No-Go**, the idea survives in principle but
  needs a scoped redesign (coarser granularity, or an explicit prior-selection sub-study) before
  a full cluster run — budget an extra 1-2 weeks for that redesign rather than proceeding as planned.
- If **Test 8 is No-Go** after reasonable iteration, treat this as a hard blocker: do not proceed
  to cluster-scale training until a toy-scale run is stable, since an unstable toy run will not
  become stable simply by adding more compute.
- If everything is Go, you have an evidence-backed justification (not just a plausible-sounding
  motivation) for spending Magus time and money on the full interpolation project — and the
  toy-scale code from Phase 0C is a head start on the real training script.
