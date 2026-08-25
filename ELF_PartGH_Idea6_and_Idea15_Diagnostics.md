# Pre-Commitment Diagnostics: Idea 6 (Principled SDE Sampler) + Idea 15 (Confidence-Gated Post-Hoc Refinement)

**Purpose and honesty note.** These plans are designed to make any eventual failure a *real*
finding rather than a preventable mistake — that is the strongest guarantee a diagnostic plan can
offer. It cannot guarantee the main project succeeds. Every prior round of this process caught a
real issue (wrong formula, confounded metric, underpowered comparison); treat that as the base
rate for what rigorous diagnostics are supposed to surface, not as bad luck to be engineered away
entirely this time.

**Scope.** Part G closes the two specific gaps flagged in the last review of the SDE/sampler
work (Idea 6): underpowered external-PPL comparison, and the unresolved confound in the `t→1`
instability finding. Part H is a full new diagnostic pilot for Idea 15, since it has not been
tested at all yet. Both run on a MacBook, reusing existing checkpoints and harnesses where
possible.

**Estimated total time: 3-4 days** (Part G: ~1 day, reusing existing infrastructure; Part H:
~2-3 days, building new infrastructure). Do not request Magus time, or commit to either idea as
the primary project, until both parts have written verdicts.

---

## Part G — Closing the Idea 6 Gaps (Statistical Power + the `t→1` Confound)

### G1 — Properly Powered External-PPL Comparison
**Why this specific fix.** The last E1 result had error bars close to the size of the effect and
PPL values (thousands) far outside any regime where the differences are interpretable — that's a
sample-size and scale problem, not a "the effect isn't there" problem. This test doesn't change
the method, it fixes the statistics.
**Method.**
- Reuse the exact E1 harness (detokenizer, `distilgpt2` scorer) and C4 checkpoint — no
  retraining needed.
- Increase the number of generated sequences per (sampler × step-count) cell substantially over
  the original run (e.g., from whatever the original small batch was to at least 200-500
  sequences per cell) — toy-scale generation is cheap, so this is a compute-trivial fix.
- Compute a paired bootstrap confidence interval (resample sequences with replacement, recompute
  the mean PPL difference between each sampler pair, repeat ~1,000 times) rather than reporting
  point estimates. Report the CI for the specific comparisons that matter (exact-SDE vs. ODE,
  exact-SDE vs. ELF's approximation) at each step count.
- Separately, flag and exclude (or clearly report separately) any generated sequences that are
  degenerate (e.g., all-repeated-token or empty) before computing PPL, since a handful of
  degenerate sequences can dominate a mean PPL and explain runaway error bars on their own —
  check whether this, rather than sample size alone, explains the earlier spike/near-zero swings.
**Go/No-Go criterion.** A comparison only counts as evidence for or against a sampler once its
bootstrap CI excludes zero difference. If CIs remain wide and overlapping even at higher sample
size, that is itself the answer: no detectable signal at toy scale, and Idea 6 should proceed (if
at all) as a cluster-scale question, not a toy-scale-validated one.
**Output.** `phase0g/results/testG1_powered_ppl_comparison.json` + a plot with bootstrap CIs
shown explicitly (not just point estimates) for each sampler at each step count.
**Est. local time.** 4-6 hours (compute is cheap; most of the time is in the bootstrap/exclusion
logic being done correctly).

### G2 — Disambiguating the `t→1` Confound (Amplification vs. Undertraining)
**Why this specific fix.** The last E2 result (fitted slope −1.50 vs. theoretical −2, with visible
flattening near `t=0.99`) is consistent with two different explanations that have different
implications, and the test as run couldn't tell them apart: either the `1/(1-t)²` amplification
factor is fundamentally noisy near the boundary regardless of training, or the network was simply
undertrained in that region because ELF's own logit-normal time schedule (`Pmean=-1.5, Pstd=0.8`)
puts very little training density there. This test separates the two.
**Method.**
- Take the C4 real-language checkpoint's training recipe and create a second variant: retrain
  (or continue training) an identical architecture with a modified time-sampling distribution
  that oversamples `t` near 1 (e.g., mix the standard logit-normal schedule with a component
  concentrated in `t ∈ [0.9, 1.0]`), so this region gets meaningfully more training exposure than
  the original run.
- Rerun the exact E2 score-error-vs-`(1-t)` measurement on this new checkpoint, using the same
  grid of `t` values as before, and compare the fitted slope and the near-boundary error magnitude
  directly against the original C4-checkpoint result.
**Interpretation, not strict Go/No-Go.**
- **If the error near `t=1` drops substantially and/or the fitted slope moves closer to the
  theoretical −2** with more training density there, that confirms undertraining was a real
  contributor — meaning the instability is partly fixable via training-schedule design, which is
  actually good news for Idea 6 (it's an engineering lever, not a fundamental dead end).
- **If the error near `t=1` is essentially unchanged despite the extra training**, that's stronger
  evidence the instability is fundamental to the exact-SDE formulation itself, and the honest
  framing for Idea 6 becomes "characterizing a real, unavoidable tradeoff" rather than "a fixable
  training issue" — still a legitimate finding, but changes what the eventual paper can claim.
**Output.** `phase0g/results/testG2_t1_confound_disambiguation.json` + an overlay plot of both
checkpoints' error-vs-`(1-t)` curves.
**Est. local time.** 4-8 hours (mostly the additional training run, still toy-scale and fast).

### G3 — Idea 6 Verdict
**Method.** Combine G1 and G2 into a single, explicit statement: does a statistically defensible
quality advantage exist for the exact SDE at any step count, and is the near-boundary instability
fixable or fundamental. State one of: **confirmed quality signal → proceed to Magus with the
original scope**; **no quality signal, but a defensible characterization/tradeoff paper → proceed
to Magus with the narrower framing**; **no signal and no clean characterization → do not proceed
with Idea 6 as primary**.
**Output.** `phase0g/results/testG_idea6_verdict.md`.
**Est. local time.** 1-2 hours.

---

## Part H — Full Diagnostic Pilot for Idea 15 (Confidence-Gated Post-Hoc Refinement)

This idea has had zero empirical testing so far — treat it with the same skepticism the SDE work
received, not less, just because the motivating examples (the "turningping" and "Kendall Kendall"
artifacts) are real and specific. A concrete-sounding motivation is not the same as a validated
mechanism.

### H1 — Distribution-Shift Probe: Which "Revision" Input Does the Frozen Decoder Actually Tolerate?
**Why this is first and non-negotiable.** The entire idea depends on feeding the frozen decode-mode
network an input it was never trained on (some stand-in for "this token needs revision"). If the
network's output degrades unpredictably under all viable stand-ins, the "no retraining required"
framing is false and the project needs to be re-scoped before anything else is tested.
**Method.**
- Using the C4 real-language checkpoint's decode-mode branch, take a batch of tokens the model
  already decodes correctly with high confidence.
- Apply three candidate "revision" transformations to their embeddings before re-running decode
  mode: (a) a hard zero vector, (b) fresh Gaussian noise at the training noise scale, (c) ELF's
  own re-corruption process (`z̃ = p·x + (1-p)·ε` from Appendix B.1, at a higher noise level `p`
  than a token gets by default) — this is the candidate the review flagged as most likely to work.
- For each, measure: does the re-decoded token stay the same or change, and does the network's
  output confidence (max softmax probability) collapse to near-uniform, stay high but wrong, or
  behave sensibly (i.e., produce a plausible in-distribution guess rather than garbage)?
**Go/No-Go criterion.**
- **Go:** at least one candidate (expected: the re-corruption approach) produces sensible,
  non-degenerate output when applied to tokens that don't need revision (near-identity behavior)
  and visibly different output when applied to genuinely wrong tokens — this is the minimum bar
  for the whole idea to be viable at all.
- **No-Go:** all three candidates produce degenerate or unpredictable output regardless of input —
  this would mean the frozen network genuinely cannot support this kind of post-hoc revision, and
  the idea needs to be rescoped as a training-time change (closer to the more expensive Idea 5)
  rather than an inference-only trick.
**Est. local time.** 4-6 hours. **Do not proceed past this test if it fails.**

### H2 — Confidence-Signal Validity: Does Entropy Actually Catch the Errors You Care About?
**Why this test design.** The review specifically flagged that repetition-type errors ("Kendall
Kendall") can be locally confident on each individual token while being globally wrong — meaning
the exact detector this idea relies on might systematically miss the exact error class that
motivated it. This needs to be tested directly, not assumed.
**Method.**
- Build a small labeled synthetic error set on top of C4-checkpoint-generated (or real wikitext-2)
  toy sequences: inject at least two distinct error types at known positions — (a) a single-token
  substitution (a clearly wrong word swapped in), and (b) a repetition error (a token or short
  phrase duplicated, mirroring the "Kendall Kendall" pattern).
- For each injected error, compute the per-token entropy/max-probability confidence signal at
  the error position and at surrounding non-error positions.
- Report precision/recall of an entropy-threshold-based detector against the known injected error
  locations, **broken out separately by error type** — this is the critical design choice, since
  averaging across types would hide exactly the failure mode in question.
**Go/No-Go criterion.**
- **Go:** the detector reliably flags substitution-type errors (expected) — but explicitly check
  and report whether it also catches repetition-type errors, since this may legitimately be a
  **No-Go for that specific error class** while still being viable for others.
- **If repetition errors are missed (expected, per the review's own concern):** this doesn't kill
  the idea, but it means the paper's scope must be explicit about which error classes the method
  addresses, and it may be worth testing one cheap alternative detector (e.g., n-gram repetition
  detection as a second, complementary signal) rather than relying on entropy alone.
**Output.** `phase0h/results/testH2_confidence_signal_validity.json` broken out by error type.
**Est. local time.** 4-6 hours.

### H3 — Refinement Loop Convergence and Stability
**Why this matters.** An iterative correction process that doesn't provably converge (or that
oscillates/diverges) isn't usable regardless of whether the underlying detector and revision
mechanism are individually sound.
**Method.**
- Using the validated revision mechanism from H1 and the detector from H2, run multiple sequential
  refinement passes (e.g., up to 5) on a batch of sequences with injected errors from H2.
- Track, pass by pass: number of remaining flagged-low-confidence positions, and (using the
  known injected error locations) whether actual errors are being corrected, left unchanged, or
  — a specific failure mode to check for — newly *introduced* at positions that were previously
  correct (a real risk: re-corrupting and re-decoding a correct token could accidentally break it).
**Go/No-Go criterion.**
- **Go:** error count decreases monotonically or plateaus across passes, and new-error
  introduction rate is low and does not grow with more passes.
- **No-Go:** oscillation (errors fixed then reintroduced elsewhere), divergence, or a new-error
  rate that erodes most of the gained corrections — this would mean the loop needs a stopping
  rule or dampening mechanism before it's usable, which is an added design step, not a dead end.
**Est. local time.** 3-5 hours.

### H4 — Naive-Masking vs. Confidence-Gated Re-Corruption: Head-to-Head Ablation
**Why this comparison, specifically.** The review's central technical claim was that literal
mask-token injection would likely underperform confidence-gated re-corruption, because ELF was
never trained on discrete masks. This should be verified empirically rather than taken as given,
since it directly determines which mechanism to build the real pilot around.
**Method.** Run H1-H3's full pipeline twice: once using a hard zero-vector "mask" as the revision
input (the naive approach), once using ELF's own re-corruption process (the review's recommended
approach). Compare final error-correction rates and new-error-introduction rates between the two.
**Go/No-Go / interpretation.** If re-corruption doesn't clearly outperform naive masking, that's
worth knowing before committing to the more complex framing in any eventual writeup — it would
simplify the method, not weaken the paper.
**Est. local time.** 2-3 hours (mostly reusing H1-H3's code with the input swapped).

### H5 — End-to-End Net Effect, Properly Powered
**Why this is the actual pilot result.** This is the real test of the idea: does the full
refinement pipeline improve *naturally generated* toy-model output (not synthetically injected
errors), measured with the same statistical rigor established in Part G.
**Method.**
- Generate a batch of naturally-occurring toy-model outputs (no injected errors) using the C4
  checkpoint, at a reasonably large sample size (matching G1's fix: hundreds of sequences, not a
  handful).
- Apply the validated refinement loop (from H1/H2/H3, using the winning mechanism from H4).
- Score before/after with the external-LM PPL harness (reused from E1/G1), using the same paired
  bootstrap CI approach as G1 — this is a **paired** comparison (same underlying generations,
  before vs. after refinement), which gives more statistical power than an unpaired comparison
  would, and is the right test design for this specific question.
**Go/No-Go criterion.** A genuine effect requires the bootstrap CI for the before/after PPL
difference to exclude zero. Anything else is not evidence of a real improvement, regardless of
how the point estimate looks.
**Output.** `phase0h/results/testH5_net_effect.json` + before/after paired comparison plot with CI.
**Est. local time.** 4-6 hours.

### H6 — Idea 15 Verdict
**Method.** Combine H1-H5 into one explicit statement: is the mechanism viable at all (H1), which
error classes does it actually address (H2), is the loop stable (H3), which revision mechanism
should be used (H4), and is there a statistically real net improvement (H5). State one of:
**confirmed Go** (proceed to a real-scale pilot on frozen ELF-B once available); **partial Go**
(viable but scoped to specific error types, or needs a stopping-rule fix from H3); **No-Go**
(distribution-shift or detector-validity failure means this needs to become a training-time idea,
closer to Idea 5, not an inference-only trick).
**Output.** `phase0h/results/testH_idea15_verdict.md`.
**Est. local time.** 1-2 hours.

---

## Sequencing Summary

| Day | Task |
|---|---|
| 1 | G1 (powered PPL comparison) + G2 (t→1 confound disambiguation), can run in parallel |
| 1 (end) | G3: Idea 6 verdict |
| 2 | H1 (distribution-shift probe — hard gate) → H2 (confidence-signal validity) |
| 2-3 | H3 (convergence/stability) → H4 (naive vs. re-corruption ablation) |
| 3 | H5 (properly-powered net-effect test) |
| 3 (end) | H6: Idea 15 verdict |
| 4 | Combine both verdicts into a single go/no-go on which idea (if either) to bring to Magus |

**Hard gate:** H1 must pass before H2-H5 are run — if the frozen decoder cannot tolerate any
revision input in a sensible way, the entire mechanism is invalid regardless of how good the
error detector or refinement loop design is.

**What "no doubt of failure" actually looks like at the end of this.** Not zero risk — a
main-project commitment where every remaining risk is a genuine open scientific question (does
this scale from 125K to 100M parameters, does real OWT behave like wikitext-2) rather than a
question you could have answered on a laptop this week.
