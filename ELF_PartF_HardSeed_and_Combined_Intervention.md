# Local Test Battery: F1 (Hard-Seed Analysis) + F2 (Combined R3+R5 Pilot)

**Mac feasibility.** F1 is analysis-only on data you already have (R3/R5 degenerate indices,
existing curvature-extraction code) — no GPU/MPS needed, runs in minutes. F2 is a short toy-scale
fine-tune combining two already-validated recipes — same scale and cost as R1-R5, which you've
already run repeatedly on this machine. Both fully in scope for local work.

**Discipline note, stated plainly.** This is the last local round for the R-series/F-series line
of investigation, full stop. Whatever these two produce, write it up and move to Magus. Given the
history in this project (T2 finding a wiring issue, T3/T4 finding regime mismatches, T7 finding a
flipped explanation), verify each result once with the checksum/independence method already
established, but do not spawn a T8, T9, etc. chasing further explanation.

---

## Part F1 — Hard-Seed Curvature Analysis (analysis only, no training)

### F1a — Partition the Seed Population
**Method.** Using the exact degenerate-index lists already recorded from the T5/T6 runs (R3's 25
indices, R5's 25 indices, and — pull this too if not already saved — Control's degenerate indices
from the same N=256 run), construct three groups:
- **Always-fails:** the 15 indices present in both R3's and R5's degenerate sets.
- **Sometimes-fails:** the 10 indices unique to R3 plus the 10 unique to R5 (fixed by one
  intervention but not the other).
- **Never-fails (control):** indices where control itself did not degenerate, and neither did
  R3 or R5 fail on them (a clean baseline-easy group for comparison).
**Output.** `phase0f/results/testF1a_seed_groups.json` — three lists of sample indices.
**Est. time.** 30 minutes.

### F1b — Extract Curvature Per Seed Group
**Method.** Reuse the existing J1 trajectory-curvature extraction pipeline. For each seed in each
of the three groups from F1a, generate (or reuse already-saved) the ODE trajectory from that exact
initial noise vector on the **control** model (use one consistent model across groups so any
difference reflects the seed's intrinsic difficulty, not which model generated it), and compute
the turning-angle curvature metric already established.
**Output.** `phase0f/results/testF1b_curvature_by_group.json` — per-seed curvature values, grouped.
**Est. time.** 2-3 hours (mostly trajectory generation for ~256 seeds if not already cached).

### F1c — Statistical Comparison
**Method.**
1. Compare curvature distributions across the three groups (always-fails vs. sometimes-fails vs.
   never-fails) using a Kruskal-Wallis test (non-parametric, appropriate given likely non-normal
   curvature distributions), followed by pairwise comparisons if the omnibus test is significant.
2. As a complementary, alternative explanation to check (don't skip this — a competing hypothesis
   test is what makes the curvature finding credible rather than a single fishing expedition):
   also test whether the always-fails group differs from the others in simple noise-vector
   properties (e.g., L2 norm of the initial `ε`, distance from the embedding-space origin) —
   if hard seeds are explained just as well by noise magnitude as by curvature, say so plainly
   rather than favoring the curvature story by default.
**Go/No-Go interpretation.**
- **Go (supports the paper's mechanism):** always-fails seeds show significantly higher curvature
  than never-fails seeds, and this isn't equally well explained by noise-vector magnitude alone.
  This becomes a genuine third piece of evidence for the curvature-overconfidence mechanism.
- **No-Go:** no significant curvature difference across groups, or the noise-magnitude control
  explains the pattern just as well. Report this honestly — "hard seeds exist but aren't explained
  by our curvature metric" is still a fine, minor observation, just not a new headline finding.
**Output.** `phase0f/results/testF1c_statistical_test.md` with test statistics and p-values for
both the curvature comparison and the noise-magnitude control.
**Est. time.** 2-3 hours.

---

## Part F2 — Combined R3 + R5 Intervention Pilot (toy-scale training)

### F2a — Train the Combined Model
**Rationale.** R3 (label smoothing) and R5 (decoupled architecture) are mechanistically
independent — one changes a loss term, the other changes model structure — so there's no obvious
reason they can't be applied together. Worth one matched pilot before deciding whether this is
worth a real-scale Magus run.
**Method.** Using the same decoupled-architecture setup from R5, fine-tune with the label-smoothing
loss from R3 added on top (same smoothing factor as the original R3 run). Match steps, data, and
all other hyperparameters exactly to the R3/R5 control protocol — this needs to be a fair,
matched comparison against all three prior conditions (control, R3 alone, R5 alone), not just a
new standalone run.
**Output.** A new checkpoint, `combined_r3_r5.pt`.
**Est. time.** Same as a single R-series run, roughly half a day.

### F2b — Evaluate Under the Established In-Distribution Protocol
**Method.** Evaluate exactly as T5 did — the regime that actually reproduces degeneracy
(`logit_normal`, 8 steps, `L=16`, matching fine-tuning length) — at the same N=256, with Wilson
CIs, and using the corrected token-ID-based degeneracy detector (not the 4-gram string version,
which T5's own writeup noted breaks at this short length).
**Report:** degeneracy rate for control, R3 alone, R5 alone (reuse existing numbers, don't rerun),
and the new combined model, all four side by side.
**Verify before trusting the result:** run the same checksum/independence check from T6 on the
combined checkpoint, confirming it's genuinely distinct from R3 and R5 individually (not an
accidental reload of one or the other).
**Est. time.** 3-4 hours.

### F2c — Does Combining Actually Shrink the Always-Fails Set?
**Method.** Using the same seed-index tracking as F1a, check whether the combined model's
degenerate indices overlap less with the original "always-fails" 15-seed set than either R3 or R5
did alone. This is the most informative version of the result — not just "is the rate lower" but
"does combining actually rescue some of the previously-unfixable hard seeds."
**Go/No-Go criterion.**
- **Go:** combined degeneracy rate is significantly lower than *both* R3 alone and R5 alone
  (not just lower than control — that bar is too easy to clear given both already beat control
  individually), via a paired bootstrap or proportion test between combined and the better of the
  two individual results. This is a real, positive, combinable result worth porting to Magus as
  the R-series' headline finding.
- **Partial:** combined beats one of the two individually but not both, or beats control clearly
  but the improvement over the better single intervention isn't statistically distinguishable —
  report as "combining doesn't clearly outperform the better individual method, but doesn't hurt
  either," which is still useful context, just not a strong new claim.
- **No-Go:** combined performs no better than the better of R3/R5 alone, or worse (possible if the
  two interventions interact badly, e.g. label smoothing interacting poorly with the decoupled
  decode branch's own loss). Report as-is; this is informative either way.
**Output.** `phase0f/results/testF2c_combined_verdict.md`.
**Est. time.** 2 hours.

---

## Sequencing and Final Stop

| Time | Task |
|---|---|
| 0.5 day | F1a + F1b: partition seeds, extract curvature |
| 0.5 day | F1c: statistical test, with the noise-magnitude control |
| 0.5 day | F2a: train the combined model |
| 0.5 day | F2b + F2c: evaluate, verify independence, check hard-seed rescue |

**Total: ~2 days.** After this, whatever F1 and F2 show, they go into the paper as final,
reported findings — this is explicitly the end of local iteration on the R/F-series. The next
step after this battery, regardless of outcome, is finalizing the pre-registration document and
moving the primary Claims A/B/C (plus F1/F2 if they land as real positive results) to Magus.
