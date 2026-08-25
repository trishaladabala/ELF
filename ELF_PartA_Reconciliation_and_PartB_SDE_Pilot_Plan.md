# Local Work Plan: Test 2/6 Reconciliation + SDE/Sampler Pilot

**Purpose.** Two workstreams, both runnable on a MacBook (CPU or Apple MPS), no cluster
required. Hand this whole document to a code agent as-is.

- **Part A (half-day to one day, do first):** reconcile the Test 2 vs. Test 6 discrepancy from
  the Phase 0 report, to close out the coupling question with confidence either way.
- **Part B (primary effort, ~2-3 weeks):** a local pilot of the SDE/sampler project — the
  same "de-risk before spending real compute" logic that Phase 0 applied to the coupling idea,
  now applied to the sampler idea before it goes to the Magus cluster. This reuses the toy model
  already built and validated in Phase 0 Tests 7-8.

Do not let Part A block Part B — they can run in parallel if convenient, but Part A is short
enough that sequencing it first is simpler and costs nothing.

**Assumed artifacts already on disk** (from the prior Phase 0 run): `test_01_data_prep.py`
outputs (`x_512.npy`, `x_128.npy`), `test_02_ot_coupling.py`, `test_06_prior_sensitivity.py`,
and the toy model/training loop from `test_07_toy_baseline.py` / `test_08_coupling_stability.py`.
Reuse this code rather than rewriting it — the point of Part A specifically is to isolate *why*
two runs disagreed, which requires holding as much of the existing code path fixed as possible.

---

## Part A — Reconciling Test 2 (2.11%) vs. Test 6 (8.2%–30.9%)

**Why this matters.** These two tests measured "OT improvement over random pairing" and got
answers that differ by more than 10x. Before treating the coupling idea as closed, it's worth
one focused day to determine whether that's a real effect masked by inconsistent measurement
setup, or a methodology artifact. This is not re-opening the coupling question — it's making
sure the No-Go is measured correctly before it gets cited as a finding.

### A1 — Audit the Two Test Scripts for Silent Differences
**Method.** Before running anything new, diff `test_02_ot_coupling.py` and
`test_06_prior_sensitivity.py` line-by-line and produce a short table of every parameter that
differs between them: sample size `n`, Sinkhorn entropic regularization `ε`, max iterations,
convergence tolerance, cost matrix normalization (squared Euclidean vs. Euclidean, raw vs.
mean-centered), random seed, and which embedding sample (full 128-d PCA set vs. a subsample)
each one draws from.
**Output.** `phase0/results/testA1_diff_table.json` — one row per parameter, values from each
script side by side.
**Est. time.** 30-45 minutes. This alone will likely explain most of the gap.

### A2 — Controlled Re-Run: Hold Everything Fixed Except the Prior/Cost Formulation
**Rationale.** The only variable that should differ between the reconciliation runs is the
thing actually being tested (prior/cost formulation) — not sample size, not `ε`, not seed.
**Method.**
- Fix sample size `n` = whatever Test 2 originally used (log it explicitly).
- Fix Sinkhorn `ε` and max iterations to Test 2's original values.
- Run the OT-vs-random comparison three times, varying only the cost/prior formulation:
  (1) Test 2's original ("gaussian"-equivalent) formulation, (2) Test 6's "empirical" formulation,
  (3) Test 6's "uniform" formulation.
- Use the same random seed across all three so the underlying `x`/`ε` samples are identical;
  only the coupling cost changes.
**Output.** `phase0/results/testA2_controlled_comparison.json` with the three reduction
percentages side by side, plus the delta from each of the original Test 2 (2.11%) and Test 6
(8.2/30.9%) numbers.
**Est. time.** 1-2 hours.

### A3 — Sinkhorn Convergence Diagnostics
**Rationale.** At `ε = 0.05` against a cost scale of ~17 (per the reported random-pairing cost
of 137.44 over ~8 points, i.e., per-point cost order ~17), entropic regularization is small
enough relative to the cost scale that under-convergence is a real possibility, not just a
formality to check off.
**Method.**
- For each of the three formulations in A2, log: number of Sinkhorn iterations to convergence,
  final marginal constraint violation (L1 distance between achieved and target marginals), and
  whether the solver hit the iteration cap without converging.
- Additionally, rerun the "empirical" formulation (the one that gave 30.9%) at two smaller `ε`
  values (e.g., 0.02 and 0.01) and two larger `ε` values (e.g., 0.1 and 0.2), holding everything
  else fixed, to see whether the 30.9% number is stable across `ε` or is itself an artifact of a
  particular regularization strength.
**Go/No-Go / interpretation criterion.**
- If marginal violation is small (e.g., <1e-3) and stable across `ε` for all three formulations,
  the numbers are trustworthy as computed, and whichever value A2 produces under matched
  conditions is the real answer.
- If the "empirical" formulation shows large marginal violation or is unstable across `ε`
  (result swings by >10 percentage points as `ε` changes), treat the 30.9% figure as unreliable
  and defer to the smaller, more stable numbers — this would confirm the original No-Go.
**Est. time.** 2-3 hours.

### A4 — Write the Reconciliation Verdict
**Method.** Produce a one-page summary: what caused the discrepancy (from A1), what the
matched-condition number actually is (from A2), and whether it's numerically trustworthy
(from A3). State a single reconciled percentage with a confidence level, and an explicit
recommendation: **confirmed No-Go**, **reopen for a closer look later (not now)**, or
**inconclusive, needs a larger sample size to resolve**.
**Output.** `phase0/results/testA_reconciliation_verdict.md`.
**Est. time.** 30-60 minutes.

**Stop condition for Part A.** Once A4 is written, stop — do not chase this further today
regardless of the outcome. If it reopens the coupling question, that goes on a list for later,
not into today's plan. Move to Part B.

---

## Part B — SDE/Sampler Local Pilot (before requesting Magus time)

**Why this phase exists.** The same logic that justified Phase 0 for the coupling idea applies
here: before spending cluster budget and calendar time on a full-scale ELF-B reproduction with a
new sampler, verify locally that the code is correct, the exact-SDE derivation actually reduces
to the ODE case, and the qualitative pattern you're hoping to reproduce (SDE beating ODE more in
the few-step regime) shows up even at toy scale. This reuses the validated toy model from Phase 0
Tests 7-8, so most of the training-loop code does not need to be rewritten.

### B1 — Extend the Toy Model to a Full Sampling-Ready Checkpoint
**Rationale.** Tests 7-8 validated that the toy training loop is correct (losses decrease, no
errors) but only ran 500 steps as a smoke test. To evaluate samplers meaningfully, you need a
toy model that has actually converged reasonably well, not just "not broken."
**Method.**
- Reuse the exact architecture and training loop from `test_07_toy_baseline.py` (2-layer,
  64-hidden toy ELF, x-prediction, shared-weight decode branch, linear rectified-flow
  interpolant — matching ELF's real training recipe, just scaled down).
- Train for longer (e.g., 5,000-10,000 steps instead of 500) on a slightly larger toy dataset
  (reuse the wikitext-2 sample from Phase 0 Test 1, or expand it modestly) until both the L2 and
  CE loss curves visibly plateau, not just trend downward.
- Save the resulting checkpoint — this is now your reusable "mini-ELF" for all of Part B.
**Go/No-Go.** Both losses should plateau (flatten within the last ~20% of training) rather than
still visibly decreasing — a still-decreasing model isn't a fair testbed for comparing samplers,
since sampler quality differences will be swamped by undertraining.
**Est. local time.** 3-6 hours (mostly unattended training on CPU/MPS).

### B2 — Implement ELF's Existing Sampler Suite Exactly as Specified
**Rationale.** You need faithful, bug-free baselines before a new sampler can be meaningfully
compared against them. This is a code-correctness step, not a research step.
**Method.** Implement, against the B1 checkpoint:
- The deterministic ODE Euler sampler (ELF's Algorithm 2).
- ELF's noise-reinjection SDE-inspired sampler (Algorithm 6), with the tunable noise
  re-injection scale `γ`.
- The logit-normal time schedule ELF uses by default (vs. a uniform-schedule control), matching
  Appendix C.6's setup.
**Sanity checks.**
- At `γ = 0`, the SDE-inspired sampler's output should be numerically identical (or extremely
  close, modulo floating-point noise) to the plain ODE sampler — this directly verifies your
  implementation matches the paper's stated behavior ("When γ = 0, no stochastic perturbation is
  applied, and the update reduces to deterministic ODE sampling").
**Go/No-Go.** The γ=0 equivalence check must pass exactly before proceeding — if it doesn't,
there's a bug in the sampler implementation that will invalidate every downstream comparison.
**Est. local time.** 4-8 hours including debugging.

### B3 — Derive and Implement the Exact Probability-Flow SDE
**Rationale.** This is the actual research contribution: ELF's own paper describes its SDE
variant as an approximation ("we adopt a simpler approximation... noting that it primarily
captures the per-step stochastic behavior"). This step produces a more principled alternative
and checks it against that approximation at toy scale before scaling up.
**Method.**
- Derive the reverse-time SDE associated with ELF's linear-interpolant flow, following the
  stochastic-interpolant formalism ELF already cites (Albergo & Vanden-Eijnden framework,
  refs [2], [43] in the paper).
- Implement it as a fourth sampler alongside the three from B2.
**Sanity checks.**
- As the derived SDE's noise term is scaled toward zero, its output should converge toward the
  plain ODE sampler's output — same logic as the γ=0 check in B2, applied to the new derivation.
- Numerically verify the derivation doesn't silently assume properties (e.g., isotropic noise,
  a specific marginal at t=1) that don't hold for ELF's actual corrupted-embedding setup at the
  final decode step.
**Go/No-Go.** The zero-noise convergence check must pass before any comparison is meaningful.
**Est. local time.** 1-2 days — this is the most mathematically involved step in the whole plan;
budget real time for it rather than rushing.

### B4 — Toy-Scale Step-Budget Comparison (the actual pilot result)
**Rationale.** This is the toy-scale rehearsal of the exact experiment you'd eventually run on
Magus (mirroring ELF's own Fig. 5c/7a protocol: sampler quality vs. step budget). If the
qualitative pattern doesn't show up even at toy scale, that's useful information before
committing cluster time to it.
**Method.**
- Using the B1 checkpoint, evaluate all four samplers (ODE, ELF's SDE approximation at 2-3 γ
  values, your derived exact SDE, each with both logit-normal and uniform time schedules) across
  a step-budget sweep (e.g., 4, 8, 16, 32, 64 steps — scaled down from ELF's 8-1024 given the
  toy model's smaller capacity).
- Use whatever quality proxy is available at toy scale (e.g., reconstruction loss against held-out
  toy sequences, or a small local LM's perplexity on generated toy outputs if one is easy to set
  up) as a stand-in for ELF's Gen. PPL metric — note explicitly in the results that this is a
  proxy, not the real metric, since the toy model and toy LM aren't the real evaluation setup.
**Interpretation, not strict Go/No-Go.** Look for the qualitative pattern ELF reports: stochastic
sampling should show a larger advantage over ODE at low step counts, with the gap narrowing as
step count increases. Three outcomes and what each means:
- **Pattern holds, and the exact SDE beats ELF's approximation:** strong signal to proceed to
  Magus with this project as the primary bet.
- **Pattern holds, but the exact SDE and the approximation perform similarly:** proceeding is
  still reasonable (a well-scoped Findings-tier contribution, as previously assessed), but temper
  expectations of a large quantitative win.
- **Pattern doesn't hold at all at toy scale:** don't treat this as a definitive kill — toy models
  are small enough that few-step degradation dynamics may not transfer — but it's a reason to
  budget a smaller/cheaper first cluster run (e.g., ELF-B at reduced token budget) rather than
  committing to the full protocol immediately.
**Est. local time.** 1-2 days (compute is cheap at toy scale; most of this is orchestration and
plotting).

### B5 — Write the Local Pilot Summary and Cluster-Scale Transition Notes
**Method.** Produce a short report covering: (a) the B4 results and interpretation, (b) a
checklist of what changes when moving from the toy model to real ELF-B (model size, real T5-small
encoder instead of the toy setup, real OWT data instead of wikitext-2, real batch size/step
counts per Appendix D.2's hyperparameters), and (c) an estimate of how much of the B2/B3 sampler
code can be reused as-is versus needs modification for the real model's dimensionality.
**Output.** `phase0b/results/sampler_pilot_summary.md`.
**Est. local time.** 2-3 hours.

---

## Sequencing Summary

| Day | Task |
|---|---|
| 1 (morning) | Part A (A1-A4): reconciliation sprint |
| 1 (afternoon) – Day 2 | B1: extend and fully train the toy model |
| Day 2-3 | B2: implement and sanity-check ELF's existing samplers |
| Day 3-5 | B3: derive and implement the exact SDE, verify zero-noise convergence |
| Day 5-7 | B4: step-budget comparison across all samplers |
| Day 7 | B5: write-up and cluster-transition notes |

**Total: about one week of local work**, most of it unattended training/sampling time rather
than active coding, before deciding whether — and how — to request Magus time for the real
SDE/sampler project.
