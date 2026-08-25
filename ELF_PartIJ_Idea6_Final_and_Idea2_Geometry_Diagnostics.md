# Final Idea 6 Consolidation + New Diagnostics: Idea 2 (Trajectory Geometry, ELF-Centric)

**Framing.** Part I is the last test for Idea 6 — after this, no more iteration on it; whatever it
shows is the answer you build on. Part J is a fresh diagnostic suite for Idea 2, chosen
specifically because it's cheap, reuses infrastructure you already have, and can be pointed
directly at explaining Idea 6's results rather than being a generic, easily-dismissed geometry
study. Together they're meant to produce one coherent "diagnosis → method" paper, not two
unrelated projects.

---

## Part I — Idea 6, Final Decisive Test (no further iteration after this)

**Why one test, not another sweep.** The last round's problem wasn't lack of signal, it was
scanning many comparisons and reporting whichever cleared p<0.05 — which inflates false positives
by construction. The fix isn't more scanning, it's committing to **one** pre-registered comparison
before looking at results, run with a design correction instead of a post-hoc one.

### I1 — Pre-Registration (write this before running anything)
**Method.** Before generating a single sample, write down and lock:
- **The one comparison that matters:** Corrected-SDE (best `g` from prior rounds, e.g. `g=0.5`)
  vs. ODE, under the **logit-normal schedule** (ELF's actual default, not uniform), at the step
  count where the earlier plot showed the largest visual gap under logit-normal (read this off
  the existing G1 plot now, before rerunning anything — do not let the new run's results
  influence which step count you pre-register).
- **The test:** a single two-sided paired bootstrap comparison (same generation seeds/prompts
  scored under both samplers) at a significance threshold of α=0.05, **with no other comparisons
  run alongside it that could be selectively reported.** If you want the other step
  counts/schedules for descriptive context, generate and report them, but only the pre-registered
  one counts as a confirmatory claim.
**Output.** `phase0i/results/testI1_preregistration.md`, written and saved **before** I2 runs.
**Est. local time.** 30 minutes.

### I2 — Run It, at Higher Power Than Before
**Method.**
- Increase sample size beyond G1's N=256 if compute allows (e.g., N=500) — this is still toy-scale
  and cheap, and directly addresses the wide/overlapping CIs seen under logit-normal last round.
- Generate, decode, score with the existing external-LM harness exactly as before — no changes to
  the pipeline itself, only to sample size and which single comparison is treated as confirmatory.
- Report the paired bootstrap CI for the pre-registered comparison, plus the other step
  counts/schedules labeled explicitly as **descriptive, not confirmatory**.
**Go/No-Go criterion (final — this is the answer).**
- **CI excludes zero, favoring Corrected-SDE:** confirmed, real, appropriately-powered result.
  Proceed to Magus for Idea 6, and use this as the headline empirical claim.
- **CI excludes zero, favoring ODE, or includes zero:** no real advantage under the schedule that
  actually matters. Idea 6 alone is not strong enough to be the paper's central claim — proceed
  only as a supporting/negative-result section (a legitimate "we tested the natural hypothesis and
  it doesn't hold under the realistic schedule" finding), not as the headline.
**Output.** `phase0i/results/testI2_final_result.json` + plot.
**Est. local time.** 4-6 hours.

**This is the last test for Idea 6.** Whatever I2 shows, write it up as-is and move on — don't run
a third round looking for a different step count or schedule that happens to work.

---

## Part J — Diagnostic Suite for Idea 2 (ELF-Centric Trajectory Geometry)

**Why this version of Idea 2, specifically.** The original scoring flagged a real risk: a generic
embedding-geometry study risks just re-describing T5's geometry, not ELF's, and risks duplicating
the paper's own Fig. 17. The version below avoids both by tying every measurement to the actual
sampler behavior already characterized in Idea 6's tests — the question isn't "what does the
embedding space look like," it's **"does trajectory geometry explain where and why the SDE
sampler helps or hurts, on your own trained model, using data you already generated."** That's
ELF-centric, non-redundant with Fig. 17 (which is a qualitative decoded-text snapshot, not a
geometric trajectory measurement), and directly strengthens Idea 6 rather than sitting beside it
as an unrelated appendix.

### J1 — Extract Per-Sample Trajectory Curvature From Existing Generations
**Rationale.** You don't need new generations for this — reuse the exact sampling runs from I2
(and, if useful, earlier G-series runs), which already recorded (or can be made to record) the
full sequence of intermediate `z_t` states during sampling, not just the final output.
**Method.**
- Instrument the sampler to log intermediate `z_t` at each step for a subsample of the generations
  already scored in I2 (re-run generation only if intermediate states weren't saved the first time
  — same seeds, so outputs will match).
- For each sampled trajectory, compute a **local curvature proxy**: e.g., the angle between
  successive step directions `(z_{t+1} - z_t)` and `(z_t - z_{t-1})`, averaged across steps in the
  trajectory — a straight-line (ODE-like) path has near-zero average turning angle; a trajectory
  that deviates more has higher values.
- Also compute your existing intrinsic-dimensionality / local-flatness metrics from the original
  Phase 0 Test 3 code, but applied to the **denoising trajectory manifold** specifically (the
  sequence of `z_t` across time for many samples), not the static data embedding cloud as before —
  this is the key reframing that makes it about ELF's dynamics, not just T5's static geometry.
**Output.** `phase0j/results/testJ1_trajectory_curvature.json`, one curvature/geometry value per
generated sample, keyed to match I2's existing per-sample quality scores.
**Est. local time.** 4-6 hours.

### J2 — The Actual Hypothesis Test: Does Curvature Predict Where SDE Beats ODE?
**Rationale.** This is the test that decides whether Idea 2 is a real explanatory finding or just
a nice-looking but empty analysis. The specific, falsifiable claim: samples/regions where
trajectories are more curved (i.e., further from the straight-line path ODE assumes) should be
exactly where stochastic (SDE) sampling shows the largest quality advantage over deterministic
(ODE) sampling, since that's the mechanism (correcting drift/error accumulation) the paper itself
proposes for why SDE helps.
**Method.**
- Using J1's per-sample curvature values and I2's per-sample quality scores (external PPL) for
  both ODE and SDE on the *same* underlying generation, compute the per-sample quality gap
  (`PPL_ODE - PPL_SDE`, i.e., how much SDE helped that specific sample).
- Regress or correlate this quality gap against the curvature metric from J1. Report a correlation
  coefficient and its significance (not just a scatter plot — this needs to be a stated,
  testable number).
**Go/No-Go criterion.**
- **Go (real finding):** a statistically meaningful positive correlation — higher curvature
  predicts a bigger SDE advantage. This is a genuine mechanistic explanation worth writing up
  as its own contribution, and it directly strengthens whatever I2 found (or, if I2 was
  null/negative, this could still be a standalone finding about *when* stochastic sampling would
  help, even if the aggregate effect in I2 was small).
- **No-Go:** no meaningful correlation. This means the geometry story doesn't actually explain
  the sampler behavior, and Idea 2 should be scaled back to a purely descriptive appendix
  (interesting context, not an explanatory claim) rather than a paper section built around a causal
  story.
**Output.** `phase0j/results/testJ2_curvature_quality_correlation.json` + scatter plot with fitted
trend and correlation statistic.
**Est. local time.** 3-5 hours.

### J3 — Rule Out the Obvious Confound (Sequence Position / Step Count)
**Rationale.** A correlation between curvature and SDE advantage could be spurious if both are
independently driven by something else — most obviously, step count or position within the
sequence (later positions have more accumulated context and may behave differently for unrelated
reasons). This needs to be checked before J2's correlation is trusted.
**Method.** Re-run J2's correlation while controlling for step count (e.g., compute the
correlation separately within each step-count bucket already collected in I2/G1, rather than
pooling across step counts) and, if feasible, position-in-sequence. Confirm the relationship holds
**within** buckets, not just in the pooled data.
**Go/No-Go criterion.** The core correlation from J2 should survive when computed within
step-count buckets. If it only appears in pooled data and vanishes within buckets, it was a
confound, not a real relationship — report this honestly rather than keeping the pooled number.
**Est. local time.** 2-3 hours.

### J4 — Idea 2 Verdict
**Method.** State explicitly: does trajectory curvature explain sampler behavior (J2/J3), and is
this novel relative to the paper's own Fig. 17 (yes, by construction, since Fig. 17 is qualitative
decoded text, not a quantitative geometric measurement tied to sampler comparison). Recommend one
of: **strong complementary section** (J2 survives J3, meaningful correlation) — write this up
alongside Idea 6 as the "diagnosis" half of a diagnosis-then-method paper; **weak/descriptive
only** (no reliable correlation) — keep a trimmed, honest geometry appendix, but don't build paper
framing around it; or **drop** if J1's curvature metric turns out to be degenerate/uninformative
across the board (e.g., near-zero variance, nothing to correlate against).
**Output.** `phase0j/results/testJ_idea2_verdict.md`.
**Est. local time.** 1-2 hours.

---

## Sequencing Summary

| Day | Task |
|---|---|
| 1 (morning) | I1: pre-register the single Idea 6 comparison |
| 1 (rest) | I2: run it — this is Idea 6's final answer, no further iteration |
| 2 | J1: extract trajectory curvature from existing/rerun generations |
| 2-3 | J2: test curvature vs. SDE-advantage correlation (the real hypothesis test) |
| 3 | J3: rule out step-count/position confounds |
| 3 (end) | J4: Idea 2 verdict, and a combined framing decision for the paper |

**What you should have at the end:** a single, honest, appropriately-powered answer on whether
Idea 6's sampler advantage is real under realistic settings, and — regardless of that answer — a
tested (not assumed) account of whether trajectory geometry explains sampler behavior on your own
model, which is the piece that turns this from a sampler tweak into a paper with an actual
mechanistic story.
