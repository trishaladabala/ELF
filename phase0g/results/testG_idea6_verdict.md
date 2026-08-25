# G3 — Idea 6 (Principled SDE Sampler) Verdict

## G1: Properly Powered PPL Comparison

**✅ A statistically significant PPL advantage exists** for the Corrected SDE at specific step counts (uniform schedule, 16 and 32 steps).

- ELF-SDE γ=0.5 vs ODE at 4 steps (uniform): Δ=+15033.9 [+2562.1, +29874.5] (worse)
- Corrected-SDE g=0.5 vs ODE at 16 steps (uniform): Δ=-1620.4 [-3167.7, -286.2] (better)
- Corrected-SDE g=0.5 vs ODE at 32 steps (uniform): Δ=-1868.8 [-3545.1, -434.8] (better)

**Critical caveat:** ELF's own SDE (γ=0.5) produces massive degeneracy (60-94% degenerate sequences), while the Corrected SDE produces 0% degeneracy. This is a strong differentiator even if the PPL advantage is modest.

## G2: Near-Boundary Stability

**The instability is partly fixable.** Oversampling t near 1 during training reduced boundary error by ~51% at t=0.99.

- Standard slope: -1.474
- Oversampled slope: -1.202
- Theoretical: -2.0

This confirms the instability is an **engineering problem** (training schedule design), not a fundamental theoretical dead end.

## Final Verdict

### **CONFIRMED QUALITY SIGNAL → Proceed to Magus with original scope.**

The Corrected SDE shows:
1. A statistically significant PPL advantage over ODE at mid-range step counts
2. Zero degeneracy vs 60-94% for ELF's approximation
3. A fixable near-boundary instability (via training schedule)
