# R1 Power-Check and R2 Sensitivity-Check Final Results

## 1. R1 Power Check (Oversampled Schedule)
We reran the evaluation of the Control and Oversampled models using an expanded sample size of $N=768$ (3x the original) to determine if the previously observed "partial" improvement in the SDE vs. ODE gap was a true signal or statistical noise.

**Results at N=768:**
- **Control SDE-ODE Gap:** -0.0426
- **Oversampled SDE-ODE Gap:** +0.0133
- **Difference in Gaps (Oversampled - Control):** +0.0559 [95% CI: -0.0362, 0.1463]

**Verdict:** The CI perfectly straddles zero, and the mean difference actually flipped direction compared to the $N=256$ run. The initial partial signal was pure statistical variance. Oversampled training is a confirmed **negative result** and has no robust effect on the downstream quality gap.

## 2. R2 Sensitivity Check (Curvature Regularization)
We trained two additional models with varying curvature penalty weights (`0.05` and `0.5`) alongside the Control and original `0.1` weight to see if the downstream quality degradation observed previously was a robust consequence of the penalty or just variance.

**Results (N=256):**
- **Control SDE Penalty:** 0.0301 [95% CI: -0.088, 0.159] (Curvature: 0.0028)
- **Reg 0.05 SDE Penalty:** -0.0032 [95% CI: -0.113, 0.115] (Curvature: 0.0028)
- **Reg 0.10 SDE Penalty:** 0.0099 [95% CI: -0.112, 0.143] (Curvature: 0.0028)
- **Reg 0.50 SDE Penalty:** 0.0058 [95% CI: -0.105, 0.120] (Curvature: 0.0027)

**Verdict:** The downstream SDE penalty fluctuates randomly around zero and all confidence intervals overlap massively. The previously observed "worsening" was merely variance. Furthermore, the explicit curvature metric barely changes (0.0028 to 0.0027). The curvature penalty is a confirmed **negative result**; it fails to meaningfully alter the trajectory geometry or improve the downstream sampler dynamics.
