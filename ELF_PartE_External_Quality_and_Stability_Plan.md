# Local Work Plan: External Quality Metric + Near-Decode Stability Characterization

**Purpose.** Two things, both doable on a MacBook, both build directly on C1-C5's existing
artifacts (no retraining from scratch needed): (1) replace the self-referential decoder-entropy
"quality" metric in C5 with an external, independent judge, since C3 already proved decoder
entropy can look good purely because of collapse; (2) characterize the near-`t=1` score-error
growth that C2 surfaced, since it's plausibly the actual reason ELF's authors chose an
approximation over an exact SDE, and turns this into a more defensible finding either way.

**Estimated total time: 1-1.5 days.** Part E1 is a hard gate for a Magus decision; Part E2 is
valuable but not gating — it can run in parallel or after, and doesn't need to finish before you
decide on cluster time.

---

## Part E1 — External-LM Quality Metric (gates the Magus decision)

### E1.1 — Detokenization Pipeline
**Rationale.** C4's BPE tokenizer already maps real wikitext-2 text to the 1024-token vocabulary
the toy model was trained on. Every sampler's output in C5 is a sequence of token IDs in that
same vocabulary — decoding them back to text is a direct inverse of a step you already built,
not new infrastructure.
**Method.**
- Reuse the exact tokenizer object saved during C4 (do not retrain or re-fit it — using a
  different tokenizer instance would invalidate the vocabulary alignment).
- Write a `decode_tokens_to_text(token_ids) -> str` function and sanity-check it round-trips
  correctly on a few real wikitext-2 sentences encoded then decoded back through C4's tokenizer.
**Go/No-Go.** Round-trip check must reproduce the original text (or something very close, allowing
for BPE's own lossy edge cases) before proceeding — if detokenization is subtly wrong, every
downstream LM score is meaningless.
**Est. local time.** 1-2 hours.

### E1.2 — External-LM Scoring Harness
**Rationale.** This is the toy-scale analog of what ELF's own paper actually measures (Gen. PPL
under GPT-2 Large) — an external, independent model judging plausibility, rather than the
sampler's own decoder grading its own homework.
**Method.**
- Load a small pretrained causal LM locally (`distilgpt2` is a reasonable choice: runs on CPU,
  handles short sequences without issue, no GPU required).
- Implement `score_text(text) -> perplexity` using the external LM's own negative log-likelihood,
  exactly mirroring the standard Gen. PPL definition (perplexity of generated text under an
  independent pretrained model) rather than inventing a new scoring convention.
- Validate the harness on two sanity cases before trusting it on real samples: (a) a batch of
  actual wikitext-2 sentences should score with low/reasonable perplexity, (b) a batch of random
  token sequences decoded to text should score with much higher perplexity. If (a) and (b) don't
  clearly separate, the harness itself has a bug — fix before proceeding.
**Go/No-Go.** The two sanity cases in the validation step must separate clearly (e.g., by at least
an order of magnitude in perplexity) before this harness is trusted for anything else.
**Est. local time.** 2-4 hours (mostly around getting tokenization boundaries between the toy
BPE vocab and distilgpt2's own tokenizer handled correctly — these are different tokenizers, so
scoring happens on the decoded *text*, not on token IDs directly).

### E1.3 — Rerun the C5 Comparison With External-LM Perplexity as the Quality Axis
**Rationale.** This is the actual re-test: same samplers, same step-budget sweep, same C4
checkpoint, but now with a quality metric that cannot be gamed by decoder collapse the way
self-entropy can.
**Method.**
- Regenerate samples exactly as in C5 (all five sampler configurations × both time schedules ×
  the same step-budget sweep: 4/8/16/32/64 steps).
- For each generated batch, decode to text (E1.1) and score with the external LM (E1.2).
- Report external-LM perplexity **alongside** the diversity metric already validated in C3 — do
  not report perplexity in isolation, since a low-perplexity collapse to a single common phrase is
  still possible and diversity is the metric that would catch it.
- Explicitly flag which sampler configurations were flagged as likely-collapsed by C3's mechanism
  (low diversity, e.g., ELF-SDE γ=1.0) and check whether they now also show poor (high) external
  perplexity, which would confirm the two metrics agree once the confound is removed, or continue
  to disagree, which would need a closer look.
**Interpretation.** Compare the resulting pattern against C5's original entropy-based ranking:
- **If the "SDE beats ODE at low step counts" pattern survives** using external perplexity and
  reasonable diversity together, that's the strongest evidence yet for the project and a
  legitimate basis to request Magus time.
- **If the pattern reverses or disappears** once judged externally, that confirms the original
  C5 ranking was measuring self-confidence/collapse rather than real generation quality, and the
  project should be re-scoped around the stability characterization in Part E2 rather than a
  quality claim.
- **If the two metrics still disagree in a way that isn't explained by collapse**, that's worth
  a specific follow-up before trusting either number.
**Output.** `phase0e/results/testE1_external_quality_step_budget.json` + a plot in the same format
as the original C5 figure, but with external-LM perplexity replacing decoder entropy on the
quality axis.
**Est. local time.** 4-6 hours (mostly generation + scoring compute, which is cheap at toy scale).

---

## Part E2 — Characterizing the Near-`t=1` Stability Tradeoff (not gating, but valuable)

**Rationale.** C2 already showed the trained network's implied score error grows sharply as
`t → 1`, consistent with the `1/(1-t)²` amplification factor in the exact score formula. If this
holds up as a real, quantifiable pattern — not just two anecdotal points — it reframes the
project's likely contribution from "our SDE wins" to "here is the actual tradeoff between the
exact formulation and ELF's heuristic, and here is why the heuristic exists." That is a more
defensible and more interesting claim to bring to a paper, and it's worth spending a few hours to
establish it properly rather than leaving it as a single suggestive plot.
**Method.**
- Using the same C2 setup (trained toy network's implied score vs. the analytic-posterior implied
  score), extend the existing `t` grid to sample more densely near the boundary (e.g., `t` in
  `{0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99}`, rather than stopping at 0.9 as in the original C2 run).
- Fit the observed network-based error as a function of `(1-t)`, and report the fitted exponent
  alongside the theoretically predicted `1/(1-t)²` scaling — do they match, or does the network's
  error grow faster/slower than the analytic amplification factor alone would predict (which
  would suggest an additional source of error beyond the known amplification)?
- Separately, check whether ELF's own noise-reinjection heuristic (Alg. 6, already implemented in
  B2) shows a comparable blow-up near `t=1`, or whether it's specifically designed to avoid this
  regime — Alg. 6's time-shifting behavior (`t_back = alpha·t`) may be doing exactly that by
  construction, which would be worth stating explicitly if confirmed.
**Interpretation, not a strict go/no-go.** This is a characterization, not a pass/fail test. The
useful outcomes are: (a) confirming the `1/(1-t)²` scaling holds and explaining why ELF's
approximation avoids the region where it would bite, or (b) finding the error growth doesn't
match the predicted exponent, which would point to a remaining implementation issue worth another
look. Either outcome is publishable framing, which is the point of doing this properly rather than
leaving it as a single plot with two data points near the boundary.
**Output.** `phase0e/results/testE2_boundary_stability_characterization.json` + a plot of score
error vs. `(1-t)` on a log-log scale (a straight line with slope ≈ -2 would confirm the predicted
scaling), with ELF's heuristic sampler's behavior overlaid for comparison.
**Est. local time.** 4-6 hours.

---

## Part F — Final Decision, Updated

**Method.** Write a short final verdict incorporating E1's result as the primary quality signal
(superseding C5's entropy-based reading) and E2's characterization as supporting context. State
one of three outcomes explicitly:
- **Confirmed Go:** external-LM-validated pattern still shows a genuine SDE-vs-ODE advantage at
  low step counts, with diversity that isn't collapse-driven — proceed to Magus with the original
  scope.
- **Reframed Go:** the quality advantage doesn't clearly survive external scoring, but E2's
  stability characterization is solid and interesting on its own — proceed to Magus with a
  narrower, more honest framing (tradeoff/characterization paper rather than a "wins" claim).
- **No-Go:** neither the quality pattern nor the stability characterization holds up — redirect
  effort to the trajectory-geometry companion study instead, which remains available as the
  lowest-risk fallback.
**Output.** `phase0e/results/final_verdict_v2.md`.
**Est. local time.** 1-2 hours.

---

## Sequencing Summary

| Time | Task |
|---|---|
| Hour 0-2 | E1.1: detokenization pipeline + round-trip check (hard gate) |
| Hour 2-6 | E1.2: external-LM scoring harness + sanity-check separation (hard gate) |
| Hour 6-12 | E1.3: rerun C5 comparison with external quality metric |
| In parallel or after | E2: near-`t=1` stability characterization |
| Final | Part F: updated go/no-go verdict |

**Hard gates:** do not proceed past E1.1 or E1.2 if their respective sanity checks fail — a broken
detokenizer or an LM harness that can't separate real text from gibberish will produce numbers
that look precise but mean nothing, which is a worse failure mode than not having the metric at
all, since it would look like resolution rather than remaining uncertainty.
