# Local Work Plan: Fix the Score Derivation, Isolate the Diversity Bug, Rerun on Real Language

**Purpose.** Three things need to happen before Part B's pilot is trustworthy enough to justify
requesting Magus time: (1) verify the score identity against ground truth, independent of any
trained network, (2) isolate whether the γ-diversity collapse is a real property or a bug, and
(3) rerun the pilot on actual language data instead of random tokens. All three run on a MacBook
(CPU/MPS). Hand this document to a code agent as-is; it assumes the Part A/B codebase already
exists and should be extended, not rewritten.

**Estimated total time: 2-3 days**, matching the prior estimate. Do not book Magus time until
Section D (final decision) is written.

---

## Part C1 — Analytic Ground-Truth Check for the Score Identity (no network involved)

**Why this test design and no other.** The B3 pilot only validated that the SDE reduces to the
ODE as g→0, which is necessary but, as established, doesn't test whether the drift correction
term itself is correct — that requires an independent ground truth. The cleanest available ground
truth is a case where the score is known in closed form without training anything: a Gaussian
mixture. If `x` is drawn from a mixture of Gaussians with components `N(μ_k, Σ_k)` and weights
`w_k`, and `z_t = t·x + (1-t)·ε` with `ε ~ N(0, σ²I)` independent of `x`, then `z_t` conditioned
on component `k` is itself Gaussian: `N(t·μ_k, t²Σ_k + (1-t)²σ²I)`. That means `p_t(z)` is a
Gaussian mixture with a known closed form, and `∇log p_t(z)` can be computed exactly — no network,
no approximation, no training noise. This is the right test because it isolates the *formula*
from every other source of error (undertraining, network capacity, tokenization).

**Method.**
- Implement a small synthetic setup: 2-4 Gaussian components in, e.g., 8-16 dimensions (matching
  the toy model's bottleneck scale), fixed means/covariances/weights.
- Implement the exact analytic score `∇log p_t(z)` from the closed-form mixture density above,
  as a standalone function taking `(z, t)` and returning the true gradient — this is your ground
  truth, verify it independently via `scipy`/`numpy` autodiff or finite differences on the
  log-density itself (a second, independent check on the ground truth function before trusting it).
- Separately, implement the **exact posterior mean** `E[x|z_t]` for this same mixture in closed
  form (a standard Gaussian-mixture posterior calculation) — this is what a perfectly-trained
  x-prediction network would converge to. Use this in place of a trained network for this test,
  so the test isolates the score-identity formula, not network training quality.
- Plug this analytic `E[x|z_t]` into the score-identity formula as currently implemented in
  `test_B3_exact_sde.py`, and separately into the corrected candidate formula
  `∇log p_t(z) = (t·v(z,t) - z) / ((1-t)σ²)` where `v(z,t) = (x_pred - z)/(1-t)` per ELF's own
  x-to-v conversion (Algorithm 2).
- Compare both against the true analytic score across a grid of `(z, t)` values (e.g., `t` in
  `{0.1, 0.3, 0.5, 0.7, 0.9}`, `z` sampled from the actual marginal at each `t`), reporting mean
  and max absolute error for each formula.
**Go/No-Go criterion.**
- The corrected formula's error should be at or near floating-point precision (this is now an
  algebraic identity being checked, not an approximation).
- The original formula's error should be large and structured (not noise) — this is expected and
  is the point of the test: it should demonstrate, numerically, that the original formula is wrong,
  not just structurally different from ELF's approximation.
- If the corrected formula does *not* match the analytic score closely, the derivation needs
  another pass before proceeding — do not move to C2 until this passes.
**Output.** `phase0c/results/testC1_score_verification.json` + a plot of error vs. `t` for both
formulas.
**Est. local time.** 4-6 hours, most of it in getting the closed-form mixture posterior right;
this is worth doing carefully since everything downstream depends on it.

---

## Part C2 — Re-Verify the Full Sampler Pipeline With the Corrected Formula

**Rationale.** C1 validates the formula in isolation. This step re-integrates the corrected
formula into the actual sampler code path (which uses a trained network's x-prediction as a
proxy for `E[x|z]`, not the closed-form posterior) and re-runs the checks B3 already did, so
the corrected version has the same evidence bar the original claimed to have.
**Method.**
- Replace the score-identity formula in the sampler code with the corrected version from C1.
- Re-run the `g → 0` convergence check (should still pass — this was never in question).
- Additionally, on the **same synthetic Gaussian-mixture data from C1**, train an actual small
  toy network (reusing B1's architecture) to convergence, then compare its *implied* score
  (via the corrected formula, using the trained network's x-prediction) against the true analytic
  score from C1. This checks that the formula is correct *and* that a realistically-trained
  network's approximation error doesn't dominate the correction term.
**Go/No-Go criterion.** Trained-network-implied score should track the analytic score reasonably
closely in regions of high data density (some divergence in low-density regions is expected and
acceptable — flag it in the write-up rather than treating it as a failure).
**Est. local time.** 4-6 hours (mostly training the toy network on synthetic data, which is fast).

---

## Part C3 — Isolate the γ-Diversity Collapse

**Rationale.** Before trusting any diversity-based claim in a rerun of B4, determine whether
token diversity collapsing to exactly 1.0 as γ increases is a real property of noise reinjection
interacting with an undertrained/random-data model, or a bug — these have very different
implications and are cheap to distinguish.
**Method, in order (stop as soon as one step identifies the cause):**
- **C3a — Isolate the noise-reinjection function alone.** Test `sde_step`'s noise reinjection
  in isolation (no full sampling loop): feed it a fixed batch of synthetic `z` values across a
  range of `γ`, and directly measure the variance of the *output* `z_back` before it ever reaches
  the network. Confirm variance increases monotonically with `γ`, as the algebra
  (`alpha = 1 - γ·dt`, `z_back = alpha·z + (1-alpha)·ε`) implies it should. If this fails, the bug
  is in the reinjection function itself (e.g., a sign error, a clamp, or `dt` being computed with
  the wrong sign/scale).
- **C3b — Check the decode step's determinism.** If C3a passes (input diversity does increase
  with γ), check whether the **decode/argmax step** is where diversity is lost — e.g., confirm
  the unembedding logits are actually changing across samples at high γ (log logit variance
  across samples, not just the final argmax token), and confirm there's no accidental temperature
  or top-1 shortcut that would make the decode step insensitive to real differences in the
  underlying continuous embedding.
- **C3c — Check whether this is a genuine undertrained-model artifact.** If both C3a and C3b look
  correct in isolation, test whether the collapse still occurs on the C1/C2 synthetic Gaussian
  mixture setup (which has real structure, unlike B1's random tokens) with a properly trained
  toy network. If the collapse disappears on structured data, it was very likely a property of
  applying strong noise reinjection to a model with nothing meaningful to denoise toward (i.e.,
  an artifact of the random-token setup, consistent with the broader concern about B1's data).
**Go/No-Go criterion.** By the end of C3, you should have one specific, stated cause (reinjection
bug / decode-step bug / random-data artifact) rather than an open question. Do not proceed to the
B4 rerun in Part C5 until this is resolved, since diversity is one of the two metrics B4 relies on.
**Est. local time.** 3-5 hours.

---

## Part C4 — Retrain the Toy Model on Real Language (Not Random Tokens)

**Rationale.** B1 trained on 256-vocab random tokens, which cannot exercise any sampler's ability
to exploit real structure. This step redoes B1 on actual text so B4's rerun can distinguish
"no signal because nothing to find" from "no signal because the samplers don't differ."
**Method.**
- Reuse the wikitext-2 sample already collected in Phase 0 Test 1.
- Train a small subword tokenizer on this sample with a modest vocabulary (e.g., 1,000-2,000
  tokens via a byte-level BPE tokenizer) — large enough to carry real structure, small enough to
  stay fast on a laptop and comparable in scale to the toy model's capacity.
- Retrain the same toy architecture from B1 on this real, tokenized data for enough steps that
  CE loss plateaus **meaningfully below** `ln(vocab_size)` (the loss a model achieves by guessing
  uniformly at random) — this is the concrete evidence that the model has learned real structure,
  not just memorized noise statistics.
**Go/No-Go criterion.** Final CE loss should be clearly and reproducibly below the random-guessing
baseline (e.g., at least 15-20% lower in nats), not just numerically lower by a negligible margin.
**Output.** A new checkpoint, `mini_elf_real_checkpoint.pt`, replacing the random-token one for
all further comparisons.
**Est. local time.** 4-8 hours (tokenizer training is fast; model training may take longer than
B1's 102 seconds since real data has more to learn — budget for it, don't force an early stop).

---

## Part C5 — Rerun the Step-Budget Comparison (B4) With All Three Fixes

**Rationale.** This is the actual re-test of the original hypothesis, now with a corrected
formula (C1/C2), an understood diversity metric (C3), and real language data (C4) — the same
experiment as B4, but this time capable of actually answering the question it was designed to
answer.
**Method.** Identical protocol to the original B4: all samplers (ODE, ELF's SDE approximation at
2-3 γ values, the corrected exact SDE at 2-3 `g` values), both time schedules, step-budget sweep
(4/8/16/32/64 steps), using the C4 checkpoint. Report both the entropy/quality proxy and the
diversity metric (now validated in C3), with the same three-outcome interpretation framework as
the original plan (strong signal / modest signal / no signal).
**Go/No-Go / interpretation.** Same three outcomes as before, but this time the result is
actually diagnostic, since undertraining-on-random-data is no longer a confound.
**Est. local time.** 1 day.

---

## Part D — Final Decision

**Method.** Write a short final verdict covering: (a) whether the corrected score formula passed
C1/C2, (b) the resolved cause of the diversity collapse from C3, (c) the C5 outcome category, and
(d) an explicit go/no-go on requesting Magus time, using the same interpretation logic as the
original plan (strong or modest signal → proceed; no signal even on real data → consider a
cheaper first cluster run rather than the full protocol, per the original plan's Outcome 3
guidance).
**Output.** `phase0c/results/final_verdict.md`.
**Est. local time.** 1-2 hours.

---

## Sequencing Summary

| Day | Task |
|---|---|
| 1 | C1: analytic ground-truth score check (do not proceed if this fails) |
| 1-2 | C2: re-verify sampler pipeline with corrected formula |
| 2 | C3: isolate the diversity collapse (a→b→c, stop at first identified cause) |
| 2-3 | C4: retrain toy model on real wikitext-2 tokens |
| 3 | C5: rerun the B4 step-budget comparison with all fixes in place |
| 3 | Part D: write the final go/no-go verdict |

**Hard gate:** if C1 does not converge to a formula that matches the analytic ground truth, stop
and revisit the derivation by hand before spending time on C2-C5 — everything downstream assumes
C1 is resolved correctly.
