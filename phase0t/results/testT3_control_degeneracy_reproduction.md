# Part T3: Control Degeneracy Reproduction Audit

**Goal:** Confirm whether the R1-R5 "control" configuration actually reproduced the established 21.5% ODE degeneracy baseline, determining if the entire R-series tested the right phenomenon.

## 1. Config Audit
The original 21.5% ODE degeneracy baseline at toy scale (from `testI2_final_result.py`) was generated under a specific regime. The R1-R5 training and evaluation scripts systematically drifted from this regime.

| Parameter | Original 21.5% Baseline (`testI2`) | R1-R5 Control Runs (`testR1_eval` etc) | Impact |
| :--- | :--- | :--- | :--- |
| **Time Schedule** | `logit_normal` | `uniform` | `uniform` is inherently more stable and produces less degeneracy at the boundary. |
| **Step Count** | `8` steps | `32` steps | Higher step counts significantly reduce ODE integration errors that trigger catastrophic collapse. |
| **Sequence Length** | `L = 128` | `L = 16` | Sequences of length 16 are incredibly short; exact 4-gram repetitions are mathematically far less likely to form and compound than at length 128. |

## 2. Rerun at High Power
To completely rule out the possibility that the earlier N=768 run showing 0.0% degeneracy was a statistical anomaly (a Wilson-interval artifact of a rare event), we reran generation strictly on the R5 control checkpoint at **N=2048**.

- **ODE Degeneracy:** 0.00% (0 out of 2048 samples)
- **95% CI:** (0.0000%, 0.1872%)

## 3. Conclusion
**Does the control reproduce meaningful ODE degeneracy?**
**NO.** The CI upper bound is 0.18%, which is two orders of magnitude lower than the established ~21.5% / 14.6% baselines.

> [!WARNING]
> Because the R1-R5 series used a highly stable configuration (`uniform`, 32 steps, `L=16`) that natively has **0.0% degeneracy**, it failed to recreate the core pathology we were attempting to fix.
> 
> **Explicit Paper Caveat Required:** The R1-R5 interventions must be explicitly caveated in the paper as: *"tested on a configuration where baseline degeneracy did not reproduce; conclusions about training-time fixes for degeneracy specifically are unconfirmed."*
> 
> The findings regarding the SDE vs. ODE quality gap on non-degenerate samples (which was actively measured) remain valid, but any claims about whether these training-time interventions *fix degeneracy* cannot be supported by this data.
