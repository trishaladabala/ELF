# R2 Sanity Check + Two New Training-Time Interventions (R3, R4)

**Framing.** Before adding new ideas, one gate: confirm R2's negative result is real and not an
unwired loss term. Then two new interventions, each targeting the mechanism one step closer to
the actual established finding (overconfidence) than R1/R2 did.

---

## Part T — R2 Implementation Sanity Gate (do this first, half a day)

**Method.**
1. Rerun the R2 fine-tune with the curvature penalty weight pushed to an extreme, unreasonable
   value (e.g., 10-50x the original `0.1`, so 1.0-5.0) purely as a diagnostic — not a candidate
   for the real method, just a test of whether the loss term can move the metric at all.
2. Separately, log the raw gradient norm contributed by the curvature loss term specifically
   (vs. the main MSE/CE loss) during a few training steps, to check whether it's numerically
   negligible by construction.
3. Confirm the curvature computation used in the loss is on tensors that are actually part of the
   active computation graph (check for accidental `.detach()`, `.item()`, or numpy round-trips
   between the trajectory extraction code, which was built for post-hoc analysis, and the training
   loop, which needs differentiable tensors).
**Interpretation.**
- **If curvature still barely moves even at extreme weight:** the R2 negative result from before
  is invalid as reported — there's an implementation bug, not a scientific finding. Fix the wiring
  and only then decide whether curvature is truly unregularizable.
- **If curvature moves sharply at extreme weight:** confirms the original negative result was
  real — the loss works, it just doesn't fix downstream quality. Keep R2's reported conclusion
  as-is.
**Output.** `phase0t/results/testT1_r2_sanity_gate.md`, stating explicitly which case applies.

---

## Part R3 — Direct Confidence Calibration on the Decode Branch

**Rationale.** R1 and R2 both intervened upstream of the actual established mechanism
(overconfidence). This targets it directly: if the SDE's problem is that it makes the model
overconfident in wrong outputs on non-degenerate paths, regularize confidence itself during
training rather than the geometry that happens to correlate with it.

**Method.**
1. Add label smoothing (a standard, well-understood intervention — start with a small smoothing
   factor, e.g. 0.05-0.1) to the decode-mode cross-entropy loss (`L_CE` in ELF's training
   objective) during a fine-tune of the C4 checkpoint, matched-control protocol identical to R1/R2
   (same steps, same data, only the loss changes).
2. As a second, alternative variant, try an explicit entropy-regularization term added to the
   decode-branch loss (penalizing low-entropy/overconfident predictions directly, rather than
   smoothing labels) — test both since they target the same goal via different mechanisms and one
   may work better than the other.
3. Evaluate exactly as before: ODE degeneracy rate, SDE quality gap on non-degenerate samples
   (paired bootstrap CI vs. matched control), and whether the SDE's characteristic entropy-lowering
   behavior (established back in the original diagnosis) is reduced.
**Go/No-Go criterion.** Same statistical bar as the R1 power check — a paired
difference-of-differences CI (regularized model's SDE-ODE gap vs. control's) that clearly excludes
zero, at a sample size large enough from the start to avoid repeating the R1 underpowered-then-flip
pattern (start at N≥512, not 256).
**Output.** `phase0r3/results/testR3_confidence_calibration.json` + comparison plot.
**Est. local time.** 1.5 days (two loss variants, matched control, adequately powered eval).

---

## Part R4 — Consistency-Style Trajectory Smoothing

**Rationale.** R2's negative result (pending Part T's sanity check) may also reflect that raw
turning-angle curvature is a weak or noisy training signal, not that trajectory smoothness is
fundamentally unhelpful. This tests a softer, more directly optimizable proxy for the same idea,
borrowed from consistency-model training.

**Method.**
1. Add a consistency loss during fine-tuning: sample two nearby time points `t` and `t'` on the
   same trajectory, and penalize the disagreement between the model's `x`-prediction at each
   (e.g., MSE between `x_pred(z_t, t)` and `x_pred(z_{t'}, t')` after accounting for their expected
   relationship under the flow) — this is a more standard, better-behaved training signal than a
   raw geometric angle.
2. Same matched-control protocol as R1/R2/R3: fine-tune from the C4 checkpoint, identical steps
   and data, only the auxiliary loss differs.
3. Evaluate identically: degeneracy rate, SDE quality gap vs. matched control with a properly
   powered paired CI (N≥512 from the start), and — as a diagnostic, not the main claim — whether
   this loss actually reduces the original turning-angle curvature metric more effectively than
   R2's direct penalty did (this would be informative regardless of whether it helps downstream
   quality, since it would tell you whether the consistency loss is a better lever on the same
   underlying geometry).
**Go/No-Go criterion.** Same as R3 — a properly powered CI that clearly excludes zero on the
downstream quality-gap improvement, not just a directional trend.
**Output.** `phase0r4/results/testR4_consistency_smoothing.json` + comparison plot.
**Est. local time.** 1.5-2 days.

---

## Sequencing and What to Do With the Results

| Day | Task |
|---|---|
| 0.5 | Part T: confirm R2's negative result is real, not a wiring bug |
| 1-2.5 | R3: confidence calibration (label smoothing + entropy regularization variants) |
| 2.5-4.5 | R4: consistency-style smoothing |

**This should be the last round of training-time intervention attempts before Magus**, regardless
of outcome. If R3 or R4 shows a real, properly-powered effect, you have your second paper
contribution (training-time prevention alongside the inference-time correction). If both are
negative too, that's not a wasted effort — a paper that shows three different, well-motivated
training-time approaches (density reweighting, geometric regularization, confidence calibration)
all fail to fix a trade-off that a simple inference-time correction handles is itself a real,
citable finding: it suggests the trade-off may be a more fundamental property of stochastic
sampling in this architecture than something trainable away, which is worth stating plainly in
the discussion section rather than treated as an absence of a result.
