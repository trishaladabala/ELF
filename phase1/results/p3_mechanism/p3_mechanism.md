# Phase P3 — Token-Level Mechanistic Analysis

**Date:** 2026-08-27 14:50

## Entropy-Stratified Analysis

Does SDE entropy increase more for low-ODE-entropy tokens (= regularization)?

| Bin | ODE Entropy | SDE Entropy | Diff (SDE−ODE) | N tokens |
|---|---|---|---|---|
| 0 | 0.0000 | 1.3637 | +1.3637 | 104,141 |
| 1 | 0.0000 | 1.3229 | +1.3229 | 104,141 |
| 2 | 0.0000 | 1.3086 | +1.3086 | 104,141 |
| 3 | 0.0000 | 1.3053 | +1.3053 | 104,140 |
| 4 | 0.0000 | 1.3056 | +1.3056 | 104,141 |
| 5 | 0.0000 | 1.2979 | +1.2979 | 104,141 |
| 6 | 0.0000 | 1.3099 | +1.3099 | 104,140 |
| 7 | 0.0000 | 1.3167 | +1.3167 | 104,141 |
| 8 | 0.0000 | 1.3430 | +1.3430 | 104,141 |
| 9 | 0.2502 | 1.5702 | +1.3200 | 104,141 |

Spearman r = -0.018 (p = 9.602e-01)

**Regularization hypothesis:** ⚠️ WEAK

## Per-Position Analysis

| Position | SDE−ODE Entropy |
|---|---|
| Q1 (0-255) | +1.32687 |
| Q2 (256-511) | +1.32195 |
| Q3 (512-767) | +1.30004 |
| Q4 (768-1023) | +1.32852 |

Spearman r = 0.027
