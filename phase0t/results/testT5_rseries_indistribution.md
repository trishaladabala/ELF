# Part T5: R-Series Re-evaluation In-Distribution

**Goal:** Re-run the R-series degeneracy comparison at `L=16` (the sequence length the checkpoints were actually fine-tuned for), but under the unstable `logit_normal`/8-step schedule.

## Results (N=256)
| Intervention | ODE Degeneracy | 95% CI | vs Control |
| :--- | :--- | :--- | :--- |
| Control | 20.31% | [15.84%, 25.66%] | N/A |
| R1 (Density Reweighting) | 35.16% | [29.57%, 41.19%] | No Significant Reduction |
| R3 (Label Smoothing) | 9.77% | [6.70%, 14.02%] | **REDUCED** |
| R3 (Entropy Regularization) | 17.97% | [13.75%, 23.13%] | No Significant Reduction |
| R5 (Decoupled Architecture) | 9.77% | [6.70%, 14.02%] | **REDUCED** |

## Conclusion
Evaluating the models in-distribution (L=16) under the unstable regime (`logit_normal`, 8 steps) successfully reproduces the baseline ODE degeneracy (Control: 20.31%, which strongly matches Phase 0's original ~21.5% finding).

With a sensitive and well-calibrated baseline, we can finally evaluate the interventions:
1. **R1 (Density Reweighting):** Actually *worsened* degeneracy (35.16%). Oversampling the early low-SNR timesteps may have harmed the model's boundary conditioning.
2. **R3 (Label Smoothing) & R5 (Decoupled Architecture):** Both successfully and significantly halved ODE degeneracy down to ~9.8%. 

**Final Verdict:** R3 and R5 do provide partial relief for ODE boundary collapse, confirming that extreme confidence/architectural entanglement contribute to the issue. However, neither intervention eliminated the collapse (both still exhibit ~10% degeneracy). Thus, the core conclusion remains intact: training-time interventions cannot fully replace the inference-time SDE correction for preventing degeneracy.