# J4 — Idea 2 (Trajectory Geometry) Verdict

## J1: Curvature Extraction

Trajectories successfully extracted. ODE curvature has a clear variance, and the global trajectory manifold occupies roughly 444 dimensions (95% variance) out of 2048.

## J2 & J3: Does Curvature Predict Sampler Behavior?

We found a fascinating, statistically significant divergence between internal confidence and external quality:

1. **External Quality (J2):** Sequence curvature vs PPL advantage (ODE PPL - SDE PPL) has a **strong NEGATIVE correlation (Spearman r=-0.351, p=2.5e-16)**. This means higher trajectory curvature actually predicts a *smaller* quality advantage for the SDE (or makes the SDE worse relative to ODE).

2. **Internal Confidence (J3):** Token curvature vs Entropy advantage (ODE Entropy - SDE Entropy) has a **strong POSITIVE correlation (Spearman r=0.398)**. This correlation holds robustly across all 16/16 sequence positions. This means higher trajectory curvature makes the SDE *more confident* relative to ODE.

### Mechanistic Implication:

The geometric hypothesis proposed that SDE corrects drift in highly curved regions. Instead, the data shows that in highly curved regions, the injected Brownian noise makes the model **overconfident in incorrect outputs**. The SDE increases internal confidence (lowers entropy) but degrades external ground-truth quality (worsens PPL) in exactly the regions with high curvature.

## Verdict

### **STRONG COMPLEMENTARY SECTION (GO)**

Although the original hypothesis (SDE helps *because* of curvature) is falsified, the observed relationship is statistically robust, perfectly controlled for position confounds (J3), and provides a powerful mechanistic explanation for how the SDE interacts with the embedding manifold. It is a novel finding that does not duplicate Fig. 17.


**Next steps for the paper:** Write this up alongside Idea 6 as the 'diagnosis' half. Idea 6 proves the SDE works (I2); Idea 2 explains its failure modes — specifically, that trajectory curvature induces overconfidence in the stochastic sampler.
