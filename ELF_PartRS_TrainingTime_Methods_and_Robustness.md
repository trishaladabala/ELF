# Pre-Magus Diagnostics: Two Training-Time Method Ideas + Robustness Additions

**Framing.** R1 and R2 below are the only additions in this document meant to change the paper's
*claim*, not just its length — they turn your inference-only correction into a two-sided
(training + inference) method contribution, which is the kind of upgrade that matters for
main-track odds. Part S (non-retraining additions) is explicitly labeled as rigor, not bulk —
each item there should only be included if it genuinely strengthens an existing claim, not
because it adds a table. If an item in Part S doesn't change what a reader concludes, cut it.

**Sequencing note.** Do R1 and R2 as toy-scale pilots on the MacBook first (both reuse your
existing C4 real-language checkpoint's training recipe). Only port a version to Magus (as a
lightweight fine-tune of the real ELF-B checkpoint, not a from-scratch retrain) if the toy pilot
shows a real effect — same discipline as everything before this.

---

## Part R — Training-Time Method Additions (pilot before Magus)

### R1 — Training-Density Fix for the Near-Boundary Instability

**Rationale.** G2 already showed a real (though not fully clean — slope -1.20 vs. theoretical -2,
plateauing) reduction in near-`t=1` score error from oversampling that region during training.
This pilot turns that into an actual, evaluable method rather than a single diagnostic plot.

**Method.**
1. Using the C4 training recipe, train two toy checkpoints identically except for the
   time-sampling distribution: (a) ELF's default logit-normal schedule (`Pmean=-1.5, Pstd=0.8`),
   (b) the same schedule mixed with an explicit oversampling component near `t∈[0.9,1.0]`
   (reuse G2's exact mixture, don't redesign it).
2. For each checkpoint, run the full existing evaluation suite: ODE/SDE degeneracy rate
   (with Wilson CIs), non-degenerate quality comparison (log-PPL, paired bootstrap), and the
   curvature-overconfidence correlation.
3. **The key comparison:** does checkpoint (b) reduce ODE's own degeneracy rate and/or narrow
   the SDE's quality penalty on non-degenerate samples, *without* needing the corrected-SDE
   sampler at all? That's the claim that would matter — a training fix that reduces the need for
   the inference-time correction, rather than just marginally improving score-error plots.
**Go/No-Go criterion.**
- **Go:** checkpoint (b)'s ODE degeneracy rate and/or non-degenerate quality gap is meaningfully
  better than checkpoint (a)'s, with CIs that don't just overlap noise. This is your second method
  contribution — write it up as a complementary training-time fix.
- **No-Go:** no meaningful difference between (a) and (b) on the actual downstream metrics (even
  if the raw score-error plot looks similar to G2). This means the training-density fix helps the
  error metric but doesn't propagate to generation quality — still worth one honest sentence in
  the paper, but don't build a section around it.
**Output.** `phase0r/results/testR1_training_density_fix.json` + comparison plot.
**Est. local time.** 1 day (two toy training runs plus the existing eval suite).

### R2 — Curvature-Regularized Fine-Tuning

**Rationale.** This is the more ambitious of the two, and the one most likely to actually move
main-track odds if it works, because it directly demonstrates that your diagnosis (Idea 2) is
*actionable*, not just descriptive.

**Method.**
1. Add an auxiliary loss term to the C4 training recipe: penalize the same local-curvature proxy
   already used in J1 (turning angle between successive `z_t` steps) during training, added to
   the existing MSE/CE losses with a small weight (start conservative, e.g. tune on a coarse grid
   of 2-3 weights rather than one guess).
2. Fine-tune (not train from scratch) a copy of the C4 checkpoint with this added loss for a
   modest number of additional steps, and separately fine-tune an identical control for the same
   number of steps with no curvature penalty (this control matters — you need to isolate the
   effect of the penalty, not just "more training").
3. Evaluate both: does the curvature-regularized model show (a) lower average trajectory
   curvature at inference time (sanity check the intervention worked), (b) reduced ODE degeneracy
   rate, (c) a smaller SDE quality-penalty gap on non-degenerate samples, all relative to the
   no-penalty control, not just relative to the original unregularized checkpoint.
**Go/No-Go criterion.**
- **Go:** the curvature penalty measurably reduces trajectory curvature AND this translates into
  either lower ODE degeneracy or a smaller SDE quality penalty, beating the matched-training-time
  control. This is a real prevention-based method, complementary to R1 and to the SDE correction —
  three coordinated contributions (diagnose, correct at inference, prevent at training) is a
  genuinely strong paper shape.
- **Partial:** curvature drops but downstream metrics don't move — report honestly as "curvature
  is a symptom you can suppress without fixing the underlying quality trade-off," which is itself
  an interesting negative finding about the mechanism, not a wasted pilot.
- **No-Go:** no curvature reduction, or quality degrades elsewhere (the penalty could hurt
  legitimate content diversity, not just pathological curvature — check for this explicitly by
  monitoring the entropy/diversity metrics already established, not just the target metrics).
**Output.** `phase0r/results/testR2_curvature_regularization.json` + before/after comparison.
**Est. local time.** 1.5-2 days (includes the weight-tuning grid and the matched control run).

---

## Part S — Non-Retraining Robustness Additions (rigor, not padding — include selectively)

Each of these reuses infrastructure you already have. Include only the ones that would change a
skeptical reviewer's confidence in your existing claims — not all of them are equally worth doing.

- **Multi-seed variance at real scale, matching the paper's own convention (Table 6, 6 seeds).**
  Not optional — this is expected rigor for any claim built on the real ELF-B checkpoints, not an
  add-on. Do this regardless of anything else in this list.
- **Curvature-metric robustness check:** rerun the J2 correlation using a second, independent
  curvature proxy (e.g., discrete second-derivative magnitude along the trajectory, not just
  turning angle) and confirm the correlation direction/significance is not an artifact of one
  specific metric definition. Cheap, and directly answers an obvious reviewer question.
- **CFG-scale × sampler interaction:** the paper's own Fig. 4 already sweeps CFG scale; extend
  your degeneracy/quality-gap comparison across 2-3 CFG values instead of one fixed value, to
  check whether the SDE-vs-ODE trade-off is stable across the guidance settings a practitioner
  would actually use, or specific to one CFG value. Reuses your existing harness entirely.
- **Failure-mode taxonomy:** a cheap qualitative pass categorizing what kinds of degeneracy occur
  (exact-token repetition vs. templated/n-gram repetition vs. incoherent-but-non-repetitive
  garbage) across ODE vs. SDE outputs already generated. This adds real explanatory content (and
  a genuinely useful table/figure) without any new generation.
- **What NOT to add just for length:** re-running the same comparison at more step counts than
  already covered, or re-deriving statistics you've already established are stable (e.g., don't
  re-audit the log-PPL vs. raw-PPL question again — that's settled). If an addition doesn't change
  a claim's credibility, it's page count, not evidence, and reviewers can tell the difference.

---

## Sequencing Summary

| Day | Task |
|---|---|
| 1 | R1: training-density fix toy pilot |
| 2-3 | R2: curvature-regularized fine-tuning toy pilot (includes weight-tuning + control) |
| 3-4 | Part S items you decide are load-bearing (multi-seed is mandatory; others as time allows) |
| After | Only port R1/R2 to Magus as lightweight ELF-B fine-tunes if their toy pilots showed a real, non-noise effect — otherwise report them as toy-scale-only footnotes and keep Magus budget focused on the core Claims A/B/C replication |

**The honest bar for "does this help main-track odds":** R1 and R2 succeeding would give you a
paper with a diagnosis, an inference-time correction, and a training-time prevention — three
coordinated contributions instead of one. That structural upgrade is what actually moves a venue
tier, not the number of tables in the appendix.
