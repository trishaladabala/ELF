# Part C/D — Final Verdict: Fix, Revalidate, Decide

## (a) Score Formula Verification (C1/C2)

**✅ PASS.** The corrected score formula `(t·x_pred - z) / ((1-t)²·σ²)` matches the analytic ground truth at machine precision.
The original B3 formula had structured errors of 41.3 at t=0.9 (vs 0.0 for the corrected formula).

**C2:** g→0 convergence verified with both analytic and trained network. Network x-prediction loss: 0.7999.

## (b) Diversity Collapse (C3)

**Root cause: random-data artifact.** The decoder head trained on random tokens produces near-uniform logits (top-1 probability ≈ 1/V). At higher γ, noise reinjection without meaningful denoising drives all samples to the same arbitrary argmax. The noise-reinjection function itself is correct (C3a ✅).

## (c) C5 Outcome

**🟡 MODEST SIGNAL.** SDE shows advantage at low steps, but the corrected formula doesn't significantly outperform ELF's approximation.

## (d) Go/No-Go on Magus Time

**C4 confirmed:** Model learned real structure (CE=1.7438, 74.8% below random).

### **CONDITIONAL GO.** Proceed with a reduced Magus budget.

The signal is modest at toy scale, which is expected given the small model. Budget a single training run rather than the full sweep.
