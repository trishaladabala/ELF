# Part F2b: Combined R3+R5 Evaluation

## Independence Verification
- **Combined Model Checksum:** `abc0bb9f449c4c75ead9186408ff63a9`
- **R3 (Label Smooth) Checksum:** `ce022e5eaef6735b081c5e17fa321c04`
- **R5 (Decoupled) Checksum:** `1cd0c8738f78500c3a9f132ef9ae930c`
**Conclusion:** The combined model is genuinely distinct from both individual models.

## Results (N=256, L=16, 8-steps logit_normal)
| Model | ODE Degeneracy | 95% CI |
| :--- | :--- | :--- |
| Control | 20.31% | [15.84%, 25.66%] |
| R3 (Label Smoothing) | 9.77% | [6.70%, 14.02%] |
| R5 (Decoupled) | 9.77% | [6.70%, 14.02%] |
| **Combined (R3 + R5)** | **12.89%** | **[9.33%, 17.55%]** |
