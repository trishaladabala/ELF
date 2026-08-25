# Pre-Registration: Real-Scale Replication (ELF-B)

**Date:** 2026-07-20 (Claims A–C), 2026-08-21 (Claim D added)  
**Context:** This document pre-registers the exact hypotheses, pass/fail thresholds, and publication strategies for the real-scale replication on the 105M-parameter ELF-B checkpoint. It incorporates critical scale-trend data from the Phase 0K local de-risking experiments and the Phase 0F Hard-Seed Geometric Analysis.

---

## 1. Primary Claims & Numeric Pass/Fail Bars

We are testing four core findings from the toy-scale Phase 0 diagnostics. Based on intermediate-scale tests (~1M and ~5M parameters), there is a significant risk that the toy findings were artifacts of under-parameterization. The numeric bars below define exactly what constitutes a "replication" at the 105M scale.

### Claim A: ODE Degeneracy Rate
- **Toy Finding:** The pure ODE sampler suffers a 21.5% catastrophic repetition collapse (degeneracy), which the Corrected-SDE perfectly eliminates (0.0%).
- **Scale Trend Warning:** At 5M parameters, the ODE degeneracy naturally fell to 0.0% without needing the SDE.
- **"Replicates" Bar:** The ELF-B ODE sampler exhibits a catastrophic degeneracy rate meaningfully above zero (e.g., **> 1.0%**), and the Corrected-SDE significantly reduces it.
- **"Does Not Replicate" Bar:** The ELF-B ODE sampler naturally avoids degeneracy (**< 1.0%** rate), proving the instability was merely an artifact of small model capacity.

### Claim B: SDE Quality Penalty on Normal Paths
- **Toy Finding:** On the subset of sequences where the ODE does *not* degenerate, the SDE is actively harmful, increasing perplexity by ~35%. The SDE trades normal-path quality for catastrophic-failure prevention.
- **"Replicates" Bar:** On the jointly non-degenerate subset of sequences, the SDE's Generative PPL is significantly *worse* than the ODE's. The paired bootstrap CI for the log-PPL difference (ODE - SDE) must be **strictly negative** (meaning SDE log-PPL > ODE log-PPL).
- **"Does Not Replicate" Bar:** The SDE matches or improves upon the ODE's Generative PPL on normal generation paths.

### Claim C: Curvature-Driven Overconfidence
- **Toy Finding:** Highly curved regions of the embedding manifold cause the SDE's Brownian noise to induce overconfidence (lower entropy), explaining the quality penalty.
- **Scale Trend Warning:** The strong positive correlation (r ≈ +0.38 at 275K params) weakened to +0.13 at 1M params and inverted to **-0.08** at 5M params.
- **"Replicates" Bar:** At ELF-B scale, there remains a statistically significant positive correlation between ODE trajectory curvature and SDE entropy advantage across tokens (**Spearman $r \ge +0.10$, $p < 0.01$**).
- **"Does Not Replicate" Bar:** The correlation is flat or negative, confirming the 5M-parameter trend that this geometric mechanism does not govern real-scale behavior.

### Claim D: Geometric Signature of Hard Seeds
- **Toy Finding:** Seeds that consistently produce degenerate ODE outputs ("Always-Fails") exhibit significantly *lower* trajectory curvature than seeds that never degenerate ("Never-Fails"). The effect is massive: Cohen's $d = 1.60$, rank-biserial $r = 0.918$, $p = 1.58 \times 10^{-9}$ (Mann-Whitney U, $N_{\text{always}}=15$, $N_{\text{never}}=200$).
- **Alternate Group Check:** The effect is not an artifact of the 15 most extreme seeds. The "Sometimes-Fails" group ($N=20$) shows a nearly identical separation from Never-Fails: Cohen's $d = 1.51$, rank-biserial $r = 0.878$, $p = 4.97 \times 10^{-11}$. Pooling all degenerate seeds ($N=35$) yields $d = 1.61$, $r = 0.895$, $p = 1.55 \times 10^{-17}$.
- **Confound Control:** Initial noise magnitude (L2 norm of $z_0$) shows *no* significant difference across groups ($p = 0.52$), ruling out the trivial explanation that degenerate seeds simply start from more extreme noise.
- **"Replicates" Bar:** At ELF-B scale, ODE-degenerate seeds show significantly lower trajectory curvature than non-degenerate seeds. Required: **Mann-Whitney $p < 0.01$** and **Cohen's $d \ge 0.50$** (a medium effect — conservatively discounting a ~3× reduction from the toy-scale $d = 1.60$, to account for the possibility that scaling compresses the curvature distribution).
- **"Does Not Replicate" Bar:** The curvature distributions of degenerate and non-degenerate ELF-B seeds are indistinguishable ($p > 0.05$ or $d < 0.30$), implying that the geometric mechanism governing failure at toy scale does not transfer.
- **Dependency:** Claim D is only testable if Claim A replicates (i.e., the ELF-B ODE sampler still produces a non-trivial number of degenerate seeds to partition on). If Claim A does not replicate, Claim D is automatically voided.

---

## 2. Fallback Paper-Shape Decision Tree

Because the pre-Magus local de-risking heavily implies that Claims A and C will *not* replicate at 105M parameters, we must explicitly pre-register the publication strategy for all possible outcomes.

| Outcome Scenario | Publication Strategy & Venue Tier |
|------------------|-----------------------------------|
| **(1) A + B + C + D all replicate**<br>*(ODE collapses, SDE fixes but costs quality, curvature drives overconfidence, hard seeds are geometrically distinct)* | **Main-Track ML Conference.** The full geometric story. We present a deep diagnostic paper of the continuous-time geometry of text, introduce the Corrected-SDE as a vital stability mechanism, and show that failure is predictable from trajectory shape. |
| **(2) A + B + D replicate, C does not**<br>*(ODE collapses, SDE fixes but costs quality, hard seeds are geometrically distinct, but curvature doesn't explain overconfidence)* | **Main-Track ML Conference.** The paper pivots to a method + failure-analysis contribution: the Corrected-SDE fixes boundary collapse, and we can predict *which* seeds will fail from their geometric fingerprint, even if the token-level overconfidence mechanism doesn't hold at scale. |
| **(3) Only A + B replicate**<br>*(ODE collapses, SDE fixes it but costs quality, no geometric signal at scale)* | **Main-Track ML Conference.** Method-only paper (Idea 6): proving boundary instability of logit-normal schedules and deriving the Corrected-SDE to fix it. Geometry is reported as a toy-scale-only phenomenon. |
| **(4) Nothing replicates cleanly**<br>*(ODE is perfectly stable, SDE offers no benefit, no geometric signal)* | **Workshop or Findings Track (Negative Result).** *"Scaling Solves Sampling Instabilities in Continuous-Time Language Diffusion."* A warning that toy-scale diagnostics are heavily confounded by under-parameterization. |
| **(5) [MOST LIKELY] Replicates ONLY on Unconditional Generation**<br>*(Phenomena hold for OWT but vanish on conditional WMT14/XSum)* | **Main-Track ML Conference.** The paper scopes its claim explicitly to unconditional, high-freedom generation. Conditional tasks demonstrate the generalization boundary. |

---

## 3. Supporting Material: Training-Time Interventions (Non-Blocking)

The following toy-scale results are included as supporting material in the final paper. They do **not** require Magus replication to be reported — they bound the limits of training-time corrections and motivate the necessity of inference-time SDE correction.

### R3/R5 Partial Fix
- **Label Smoothing (R3):** Halved ODE degeneracy from ~20% to ~10% at toy scale. Mechanism: softened boundary confidence prevents hard collapse for marginal seeds.
- **Decoupled Architecture (R5):** Independently halved ODE degeneracy from ~20% to ~11%. Mechanism: separating flow-matching and token-prediction representations prevents the decoder from inheriting the encoder's boundary instability.

### Non-Synergy Finding
- **Combined R3 + R5:** Yielded 12.89% degeneracy — *worse* than R3 alone (10.55%) and statistically indistinguishable from R5 alone (11.33%). The interventions interact non-additively; Label Smoothing may dampen the discrete confidence signals the Decoupled decoder relies on.

### Failed Interventions
- **Density Re-weighting (R1):** Biasing training toward $t \to 1$ worsened degeneracy to ~28%. Forcing the model to look at the boundary destabilizes the learned flow field.
- **Curvature Regularization (R2):** Directly penalizing curved trajectories also worsened degeneracy to ~28%. Artificially straightening paths disrupts the natural geometry the model needs to navigate.

### Implication
Training-time fixes can reduce but not eliminate deterministic degeneracy. A persistent ~10% "hard-seed" floor exists that no tested intervention can breach. This directly motivates the Corrected-SDE as a necessary inference-time backstop, and makes Claim D's geometric characterization of the hard seeds scientifically valuable: if we can predict failure from geometry, we can at minimum apply the SDE selectively.

---

## 4. Status of the Adaptive Sampler Upgrade

**Decision: EXCLUDED from primary run.**

The Curvature-Driven Adaptive Sampler prototype (Idea 1 fold-in) was tested at toy scale (Phase 0K, Part 3). While it reduced degeneracy from 18% to 8%, it failed to match the fixed Corrected-SDE's complete elimination (0.0%). 

Because the primary purpose of the SDE in this architecture is to serve as an absolute backstop against catastrophic failure, an adaptive variant that still allows 8% of sequences to collapse is not a viable replacement. 

**This upgrade will NOT block or delay the core replication.** It may be explored as a secondary footnote only if extra cluster time remains after the primary ELF-B results are finalized.
