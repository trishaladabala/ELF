# Phase P5 — Adaptive Noise Sampler

**Config:** N=256, L=1024, 32 steps

| Strategy | Degeneracy | Mean PPL | Mean log-PPL | vs ODE | 95% CI |
|---|---|---|---|---|---|
| ODE (γ=0) | 0.0% | 263.7 | 5.534 | — | — |
| Fixed SDE (γ=1.0) | 0.0% | 173.8 | 5.124 | +0.410 | [0.389, 0.430] |
| Adaptive SDE | 0.8% | 87.0 | 4.426 | +1.106 | [1.072, 1.141] |

**Adaptive vs Fixed SDE:** +0.696 [0.666, 0.725]

**Total time:** 673s
