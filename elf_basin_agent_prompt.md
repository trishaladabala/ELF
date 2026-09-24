# Agent Implementation Prompt — Decoder-Basin Research on ELF

Paste everything below the line into Antigravity as the project brief. It is written
to be read by an agent that has a shell, a Python environment, and one NVIDIA RTX
A4000 (16 GB).

---

## ROLE

You are the implementation engineer for a machine-learning research project. You will
build a reproducible experimental codebase, not a product. Every result must be
regenerable from a config file and a seed. You will work in phases with explicit
go/no-go gates. **Do not skip a gate. Do not start a later phase before the gate for the
current phase has passed and been reported.**

## HARDWARE AND NON-NEGOTIABLE CONSTRAINTS

- One NVIDIA RTX A4000, 16 GB VRAM. Assume ~14 GB usable.
- Never attempt to train any diffusion language model from scratch. The base model was
  trained on TPU v5p×64 for 7.5 hours. The only training permitted in this project is
  adapter- or head-scale fine-tuning with the main network frozen.
- Default to bf16 for all forward passes. Use fp32 only for loss accumulation and for
  optimizer state on the small trainable modules.
- Use FlashAttention / `scaled_dot_product_attention` everywhere. Never materialize a
  full `[B, L, V]` fp32 logits tensor at L=1024; at V≈32k that is ~131 MB per sample
  before backward. Use top-k logits, chunked positions, or fused CE.
- Keep a running disk budget. State harvesting can trivially produce hundreds of GB if
  done naively. Store as fp16 memmap and subsample timesteps.
- Every long-running job must checkpoint and be resumable. Assume the machine can be
  interrupted.

## SCIENTIFIC BACKGROUND (read this before writing code)

Two papers define the project. Both PDFs are in `papers/`.

1. **ELF: Embedded Language Flows** (arXiv 2605.10938). A continuous diffusion language
   model. A frozen T5-small encoder maps tokens to 512-d contextual embeddings. A
   flow-matching denoiser (`ELF-B`: 12 layers, hidden 768, 12 heads, 105M params, with a
   128-d bottleneck) transports Gaussian noise to clean embeddings via
   `z_t = t·x + (1−t)·ε`. The same shared-weight network switches to `mode="decode"` at
   `t=1` and produces token logits through an unembedding matrix. Discretization happens
   exactly once, at the end. Conditioning (time, CFG scale, mode) is in-context: 4 time
   tokens, 4 CFG tokens, 4 mode tokens prepended to the sequence.

2. **Continuous Language Diffusion as a Decoder-Interface Problem** (arXiv 2606.08810).
   An analysis of ELF's released checkpoints. Its central object is the **decoder margin**

   ```
   m_i(h) = g_i(h) − max_{j≠i} g_j(h)
   ```

   and the **margin-τ basin** `B_{i,τ} = { h : m_i(h) ≥ τ }`. It shows that generation
   succeeds when trajectories enter a high-margin basin, that basin entry is *token-wise*
   and staggered across positions, and that MSE and perplexity can each validate the
   wrong object.

   Its **Theorem 1** states that a perturbation `δ` preserves the decoded token when
   `‖δ‖₂ < m(h)/L(h)`, where `L(h)` is the local Lipschitz constant of the pairwise
   margin functions. **The paper only ever optimized `m`. It never estimated or
   regularized `L`, and its scalar hinge-margin experiment (§5.13) failed after both 5k
   and 10k steps.** Its own stated diagnosis is that what matters is training the decoder
   on "the neighborhood the denoiser must actually visit," not maximizing a per-sample
   scalar gap.

   Its reverse-basin-navigation experiment further shows the basin is strongly
   **anisotropic**: random, PCA, sentiment, and decoder-gradient directions have very
   different effective sensitivities. Isotropic Gaussian perturbation is therefore the
   *least* informative probe direction.

**Our thesis.** Basin quality is governed by the ratio `m/L` measured along
trajectory-realistic directions, not by margin alone and not by embedding MSE. We test
this by (i) building the measurement harness, (ii) training a decoder-side adapter on
harvested trajectory states with an `m/L` objective, and (iii) if time permits, adding an
inference-time projection that exploits the frozen encoder ELF discards.

## WHAT WE ARE BUILDING (three tracks)

- **Track C — testbed.** Everything runs on the *conditional* checkpoint
  `ELF-B-de-en` at L=128 (64 condition + 64 target) as the primary testbed, with
  `ELF-B-owt` at L=1024 as the secondary. Rationale: L=128 is roughly an order of
  magnitude cheaper per step, BLEU is a metric reviewers trust, and the prior analysis
  paper audited *only* unconditional OWT, so conditional basins are unstudied.
- **Track A — method.** A trajectory-realistic, sensitivity-regularized decoder adapter.
  The denoiser trunk stays frozen. Only the decode path is adapted.
- **Track B — stretch.** Interface-anchored sampling: at mid-trajectory, re-encode the
  current token hypothesis with the frozen T5 encoder and anchor high-`m/L` positions to
  that on-manifold projection.

---

# PHASE 0 — Environment and reproduction gate

**Goal: prove we can load the released weights and match published numbers.** Nothing
else matters until this passes.

Tasks:

1. Create the repo skeleton:

   ```
   elf-basin/
     configs/           # YAML, one per experiment, seeded
     src/elfbasin/
       model/           # checkpoint loading, ELF wrapper
       sampling/        # instrumented ODE/SDE samplers
       harvest/         # trajectory state caching
       train/           # adapter training
       eval/            # metric suite
       analysis/        # plots, tables
     scripts/           # thin CLI entrypoints
     tests/
     papers/
     runs/              # outputs, gitignored
     RESULTS.md         # append-only experiment log
   ```

2. Set up the environment. Try, in this order:
   - `github.com/Ugness/ELF-pytorch` (unofficial PyTorch reproduction, has model
     definition, training, and eval).
   - `github.com/lillian039/ELF` (official, JAX/TPU) — use only as the reference for
     algorithm details and hyperparameters. Do not try to run JAX on this GPU.

   Checkpoints live on Hugging Face under the org `embedded-language-flows`. Pull
   `ELF-B-owt` and `ELF-B-de-en`. Verify with **strict key matching** against the model
   definition and report any missing or unexpected keys explicitly. If a key mismatch
   exists, stop and report; do not silently `strict=False`.

3. Write `src/elfbasin/model/elf.py`: a thin wrapper exposing
   - `denoise(z, t, sc, cfg, cond=None) -> x_hat`
   - `decode(x, cfg, cond=None) -> logits`  (this must use `mode="decode"`, `t=1`, and
     the real nonlinear shared-weight readout, **not** a bare unembedding matmul; a bare
     unembedding is a different object and will invalidate every margin measurement)
   - `encode(token_ids) -> x` (frozen T5-small encoder, plus ELF's latent normalization)

4. Reproduce two numbers:
   - **OWT:** `ELF-B-owt`, 32-step SDE, γ=1.5, self-conditioning CFG=3, logit-normal time
     schedule (P_mean=−1.5, P_std=0.8), 1000 samples. Target Gen. PPL ≈ 24 under
     GPT-2-Large, entropy ≈ 5.15. The unofficial PyTorch repro reports 25.61 / 5.20, so
     anything in 24–26 with entropy near 5.15–5.20 is acceptable.
   - **De-En:** `ELF-B-de-en`, 64-step ODE, self-conditioning CFG=1, input-condition
     CFG=2, WMT14 De-En test set. Target BLEU ≈ 26.4.

**GATE 0.** Report both numbers in `RESULTS.md` with wall-clock, batch size, peak VRAM,
and the exact config hash. If either number is off by more than ~10% relative, stop and
report the discrepancy with your diagnosis before proceeding.

---

# PHASE 1 — Instrumented sampler and measurement harness

**Goal: a sampler that records everything the analysis needs, at negligible extra cost.**

The official generation loop is scan-like and exposes nothing. Write our own explicit
loop that performs identical SDE/ODE updates (verify bit-comparable-to-tolerance output
against Phase 0 for the same seed) and additionally records, at every step `s` and every
position `i`:

- `m_i` — top1-minus-top2 margin of the decode-mode readout applied to `x̂_s`
- `top1_id`, `top2_id`, and top-k logits (k=8 is enough; do not store full vocab)
- decoder logit entropy
- self-conditioning delta `‖x̂_s − x̂_{s−1}‖` (per position and sequence-level)
- agreement of `top1_id` with the final decoded token (computed post-hoc)
- an `L_i` estimate — see below
- sequence-level: 10th-percentile margin, effective rank of `x̂_s` (entropy of the
  singular value spectrum)

**Estimating `L_i` (the local sensitivity).** Implement two estimators behind a common
interface and compare them:

- `jvp`: forward-mode Jacobian-vector product of the decode readout at `x̂_s` along a
  probe direction, taking the max over the top-k competing tokens of
  `|∇(g_i − g_j)·v|`. Average over a few probe directions.
- `fd`: finite differences, `(g(x̂ + h·v) − g(x̂ − h·v)) / 2h`, cheaper and adequate.

Probe directions `v` must be selectable: `{isotropic, trajectory, decoder_grad, pca}`.
The `trajectory` option is the important one and is defined in Phase 2.

Then define the **normalized margin** `ρ_i = m_i / L_i` and record it alongside `m_i`.

Also implement the metric suite in `src/elfbasin/eval/`:

- Gen. PPL under GPT-2-Large (geometric and arithmetic mean, both reported)
- unigram entropy, distinct-1/2, repeated-4-gram fraction
- JS divergence to an OWT reference token distribution
- MAUVE against OWT references
- BLEU (sacreBLEU) for De-En, ROUGE-1/2/L for XSum
- token agreement to a reference decode

**Every experiment in this project reports the full suite, never PPL alone.** A recent
paper (arXiv 2607.00588) shows ELF's low PPL can reflect a repetition attractor, so a
PPL-only claim is not defensible.

**GATE 1.** Reproduce, qualitatively, the phase structure from the analysis paper on
`ELF-B-owt` SDE32, 512 samples: (a) mean margin rises monotonically over denoising phase;
(b) self-conditioning disagreement peaks in a middle region; (c) per-token basin-entry
times are visibly staggered rather than simultaneous. Produce three plots. If you cannot
reproduce the staggering, the whole project premise is wrong; stop and report.

---

# PHASE 2 — Trajectory state harvesting

**Goal: a cached dataset of the states the denoiser actually visits.**

For a set of sampled trajectories, store to fp16 memmap:

- `x̂_s` at a subsampled set of steps (for a 32-step sampler, store steps
  {4, 8, 12, 16, 20, 24, 28, 31}; do not store all 32)
- the corresponding `z_s`
- the residual `r_s = x̂_s − x_ref`, where `x_ref` is the frozen-encoder re-encoding of
  the final decoded sequence (this is our proxy for the clean target on generated data)
- the native decoder's argmax token ids at the final step
- for De-En, additionally the ground-truth target tokens and their true encoder states

**Disk arithmetic — do this before launching.** De-En at L=128, d=512, fp16:
`4096 samples × 128 pos × 512 dim × 2 B ≈ 537 MB` per stored timestep. Eight timesteps is
~4.3 GB. OWT at L=1024 is 8× that per timestep, so for OWT subsample positions as well
(e.g. 256 random positions per sample) or reduce to 4 timesteps. Print the projected size
and refuse to run if it exceeds a configured cap.

The empirical distribution of `r_s` at each phase is the **`trajectory` probe direction**
used in Phase 1 and the perturbation distribution used in Phase 3. This is the concrete
operationalization of "the neighborhood the denoiser must actually visit."

**GATE 2.** Report, per phase: the norm distribution of `r_s`, its top principal
directions, and the cosine between `r_s` and (a) isotropic random directions and (b) the
decoder-gradient direction. If `r_s` is statistically indistinguishable from isotropic
noise, the anisotropy premise fails and Track A loses its motivation. Report this
honestly either way.

---

# PHASE 3 — Headroom / ceiling test (critical go-no-go)

**Goal: find out whether there is anything left for a better decoder to win.**

The analysis paper's Minimal Decoder Protocol found that a single linear readout on
generated final states reaches 97.9% agreement with the native decoder at 32k samples.
That means most bulk token decisions are easy and any decoder-side method has a small
target. We must size that target before we build a method.

Do three measurements:

1. **Oracle-decoder ceiling.** For harvested final states, if every position were decoded
   to the token that the *ground-truth-conditioned* encoder state decodes to, how much do
   Gen. PPL / BLEU improve? This is the upper bound on any decoder-side method.
2. **Early-state ceiling.** Same, but decoding `x̂_s` at step 20, 24, 28 instead of the
   final step. This bounds how many NFEs a better decoder could save.
3. **Error taxonomy.** Of positions where the native decode differs from the oracle,
   classify by frequency bucket, subword vs. whole-word, numeric, and by `ρ_i` decile.

**GATE 3 — decision point.**

- If the oracle-decoder ceiling on De-En is **≥ +1.0 BLEU** or the early-state ceiling
  shows ≥ 25% NFE savings at matched quality, **proceed to Phase 4 (Track A)**.
- If the ceiling is below that, **skip Phase 4 and go straight to Phase 5 (Track B)**, and
  report Phase 3 as a standalone negative-result contribution. Do not build a method into
  a ceiling that doesn't exist.

State the decision explicitly in `RESULTS.md` with the numbers behind it.

---

# PHASE 4 — Track A: sensitivity-regularized decoder adapter

**Goal: adapt only the decode path so that generated states decode robustly.**

Freeze everything except a small trainable module on the decode path (a LoRA on the
decode-mode projections plus the unembedding, or a small adapter block; keep trainable
parameter count under ~5M and log it).

Train on harvested states from Phase 2, not on clean encoder states. The objective:

```
L = CE( decode(x̂ + r), y )                                  # trajectory-realistic CE
  + λ_sens · E_v [ ‖ ∇_x (g_top1 − g_top2) · v ‖ ]           # local sensitivity penalty
  + λ_cons · KL( decode(x̂) ‖ decode(x̂ + r') )                # neighborhood consistency
```

where `r, r'` are drawn from the **empirical residual distribution at the matching
phase**, and `v` is drawn from the trajectory-direction distribution. The second term is
the untried half of Theorem 1: it lowers `L(h)` instead of raising `m(h)`.

Required ablation ladder, all with identical data, seed, step count, and trainable
parameter count:

| # | Variant | Purpose |
|---|---------|---------|
| 0 | frozen native decoder | baseline |
| 1 | adapter, clean-state CE | does adaptation alone help? |
| 2 | adapter, isotropic-Gaussian CE | reproduces ELF's existing recipe |
| 3 | adapter, scalar hinge margin on `m` | reproduces the published negative result |
| 4 | adapter, trajectory-residual CE | tests "right neighborhood" |
| 5 | 4 + sensitivity penalty | tests `m/L` (our method) |
| 6 | 4 + 5 + consistency KL | full |

Variant 3 must be included. Reproducing the prior paper's negative result and then
showing what fixes it is the argument structure of the paper.

Report for each: full metric suite, plus the mechanism metrics — margin and `ρ`
distributions over phase, token flip rate under trajectory-realistic perturbation, and
the phase at which the 10th-percentile margin crosses fixed thresholds.

**Memory notes.** Use `mode="decode"` forward with gradient checkpointing. Restrict the
CE and margin terms to top-k (k=64) competing tokens to keep logits small. On De-En at
L=128 you should fit batch 16–32; on OWT at L=1024 drop to batch 2–4 with accumulation.

**GATE 4.** Variant 5 or 6 must beat variant 0 on De-En BLEU **and** not degrade entropy,
distinct-2, or repetition. If PPL improves while entropy drops, that is the repetition
attractor and counts as a failure, not a success. Say so plainly if it happens.

---

# PHASE 5 — Track B: interface-anchored sampling (stretch)

**Goal: exploit the frozen encoder that ELF discards at inference.**

ELF states explicitly that the encoder is used only during training. But the state space
*is* the frozen T5 contextual embedding manifold, so the encoder is a free projection
operator onto that manifold, available at inference for ~35M params.

Algorithm, at sampling step `s`:

1. Compute `ρ_i` for every position from the decode-mode readout.
2. Maintain an anchor set `A`: positions whose `ρ_i` has exceeded threshold `τ` for `r`
   consecutive steps.
3. Every `K` steps, take the full argmax hypothesis `ŷ_s`, run the frozen T5 encoder on
   it to get `x̃_s`, and for `i ∈ A` blend
   `x̂_s[i] ← (1−λ)·x̂_s[i] + λ·x̃_s[i]`, then re-derive `z_s[i]` consistently.
4. Unanchored positions flow normally and now attend to clean, on-manifold context.
5. Exit when `A` covers the sequence.

**Run the decisive three-way pilot first, before building the full thing.** At step 20 of
32, for the top-decile-`ρ` positions, compare:

- (a) do nothing (baseline)
- (b) freeze the latent in place (the analysis paper reports this fails)
- (c) re-encode through T5 and anchor

If (c) is not clearly better than (b), stop Track B and report it. That single comparison
is the whole bet.

**Required control experiment.** Anchoring must be applied at the *wrong* phase (early,
pre-basin-entry) and shown to degrade quality. Without that control, anchoring is a
heuristic; with it, the claim is about *when* a continuous DLM should discretize.

Also test the predicted negative case if time allows: anchoring should *not* help a
learned-VAE latent system, because there is no frozen encoder defining the manifold.

---

# PHASE 6 — Write-up assets

Produce, as files:

- `RESULTS.md` — append-only, every run with config hash, seed, hardware, wall-clock
- A results table generator producing LaTeX for the ablation ladder
- Plots: phase diagram of margin and `ρ`; per-token basin-entry histogram; PPL–entropy
  frontier (every method must be shown as a *frontier*, not a single point, by sweeping
  CFG); NFE-vs-quality curve; error taxonomy
- `repro.sh` that regenerates every number from scratch

---

## RULES FOR YOU, THE AGENT

**Do:**
- Report VRAM and wall-clock for every run.
- Seed everything and record the seed. Run at least 3 seeds for any headline claim and
  report mean ± std.
- Write a unit test for the sampler equivalence check (Phase 1) and for the margin and
  `ρ` computations against a hand-computed toy case.
- Ask me before any change that alters the experimental protocol.
- When a gate fails, say so clearly and stop. A clean negative result reported early is
  worth more than a broken pipeline discovered late.

**Do not:**
- Do not train the denoiser trunk. Do not train anything from scratch.
- Do not use a bare unembedding matmul as "the decoder." The native decoder is a
  mode-conditioned nonlinear shared-weight readout.
- Do not use isotropic Gaussian perturbation as the default; it is the baseline we are
  arguing against, so it belongs in the ablation table, not in the method.
- Do not report Gen. PPL without entropy, distinct-2, and repetition alongside it.
- Do not `strict=False` your way past a checkpoint key mismatch.
- Do not silently tune a hyperparameter on the test set. Split generated samples into
  validation and test halves and select thresholds on validation only.
- Do not delete or overwrite entries in `RESULTS.md`.

## FIRST ACTION

Read both PDFs in `papers/`. Then produce a written implementation plan for Phase 0 only:
the exact packages, the checkpoint loading strategy, the two reproduction commands, and
your estimate of VRAM and runtime for each. Wait for my approval before writing code.
