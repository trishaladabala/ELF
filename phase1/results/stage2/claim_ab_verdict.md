# Stage 2 — Claims A+B Verdict

**Date:** 2026-08-27 13:29
**Config:** N=1024, L=1024, 32 steps, γ=1.5

## Claim A: ODE Degeneracy

| Sampler | Degeneracy | 95% CI |
|---|---|---|
| ODE | 0.0% (0/1024) | [0.0%, 0.4%] |
| SDE | 0.7% (7/1024) | [0.3%, 1.4%] |

**Verdict:** ❌ DOES NOT REPLICATE (bar: >1%)

## Claim B: SDE Quality Penalty

- Jointly non-degenerate: 1017/1024
- Mean log-PPL diff (ODE−SDE): 0.7253 [0.6837, 0.7678]
- **Verdict:** ❌ DOES NOT REPLICATE

**Total time:** 1111s
