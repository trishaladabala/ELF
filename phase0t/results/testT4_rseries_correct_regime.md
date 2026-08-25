# Part T4: R-Series Re-evaluation in Degenerate Regime

**Goal:** Determine whether the R1, R3, and R5 interventions reduce ODE degeneracy when evaluated under the regime where it actually occurs (`logit_normal`, 8 steps, `L=128`).

## Results (N=256)
| Intervention | ODE Degeneracy | 95% CI | vs Control |
| :--- | :--- | :--- | :--- |
| Control | 100.00% | [98.52%, 100.00%] | N/A |
| R1 (Density Reweighting) | 100.00% | [98.52%, 100.00%] | No Significant Reduction |
| R3 (Label Smoothing) | 100.00% | [98.52%, 100.00%] | No Significant Reduction |
| R3 (Entropy Regularization) | 100.00% | [98.52%, 100.00%] | No Significant Reduction |
| R5 (Decoupled Architecture) | 100.00% | [98.52%, 100.00%] | No Significant Reduction |

## Conclusion
None of the training-time interventions (R1, R3, R5) successfully reduced ODE degeneracy compared to the control under the unstable boundary regime. This confirms that these training interventions do not fundamentally solve the degeneracy collapse.