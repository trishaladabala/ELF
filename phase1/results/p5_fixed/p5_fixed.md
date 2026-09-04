# Phase P5 (FIXED) — Adaptive Sampler with Official SDE

**Pipeline:** Official `_ode_step`/`_sde_step` (matches P1)
**Config:** N=512, L=1024, 32 steps

## Results

| Strategy | Deg | Mean PPL | log-PPL | vs ODE | 95% CI |
|---|---|---|---|---|---|
| ODE (γ=0) | 0.0% | 48.4 | 3.825 | — | — |
| Fixed SDE (γ=1.0) | 0.0% | 30.5 | 3.366 | +0.458 | [0.431, 0.485] |
| Fixed SDE (γ=1.5) | 0.4% | 27.8 | 3.112 | +0.715 | [0.654, 0.777] |
| Adaptive SDE | 2.3% | 11.2 | 1.837 | +1.988 | [1.897, 2.075] |

## Comparisons

- **Fixed SDE (γ=1.0):** +0.458 [0.431, 0.485]
- **Fixed SDE (γ=1.5):** +0.715 [0.654, 0.777]
- **Adaptive SDE:** +1.988 [1.897, 2.075]
- **Adaptive vs Fixed SDE (γ=1.0):** +1.528 [1.438, 1.617]
- **Adaptive vs Fixed SDE (γ=1.5):** +1.280 [1.183, 1.378]

## Adaptive γ Analysis

- Overall mean γ: 2.886 ± 0.343
- Mean within-step γ std: 0.0091
- Per-step γ: [1.0, 2.99, 3.0, 3.0, 3.0, 3.0, 2.91, 2.99, 3.0, 3.0, 3.0, 3.0, 3.0, 2.99, 2.99, 2.96, 2.95, 2.97, 2.96, 2.84, 2.93, 2.92, 2.9, 2.82, 2.91, 2.93, 2.94, 2.88, 2.92, 2.93, 2.8, 2.92]
