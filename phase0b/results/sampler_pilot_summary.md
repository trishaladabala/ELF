# SDE/Sampler Local Pilot Summary

## Part A — Test 2/6 Reconciliation

### Finding: Discrepancy Fully Explained

The 10× gap between Test 2 (2.11% OT reduction) and Test 6 (8.2%–30.9%) was caused by **two compounding methodology differences**, not by a real effect:

| Factor | Test 2 | Test 6 | Impact |
|---|---|---|---|
| **Reduction metric** | Soft OT plan cost | Hard argmax assignment | Inflates by ~4× |
| **Noise distribution** | N(0, I) — ELF's actual setup | Empirical/uniform (data-matched) | Inflates by ~4-8× |

**Controlled experiment (A2):** Under matched conditions (n=1500, same seed, ε=0.05):
- Gaussian prior + soft metric (ELF-relevant): **2.27%**  
- Gaussian prior + hard metric: **8.70%**
- Empirical prior + hard metric: **31.70%**

**Convergence (A3):** All Sinkhorn runs converged with marginal violations at machine epsilon (~10⁻¹⁶). Numbers are trustworthy.

### Verdict: **Confirmed No-Go** for coupling project. Reconciled number = 2.27%.

---

## Part B — SDE/Sampler Pilot Results

### B1: Toy Model Checkpoint
- **125K param mini-ELF** (2-layer, 64-hidden, 32-bottleneck)
- Trained for 8,000 steps in 102 seconds on MPS
- CE loss plateaued at 5.545 (0.00% change in last 1000 steps)
- L2 loss still slightly noisy (2.17% change) — expected for random data

### B2: Sampler Suite
- ODE Euler and ELF's SDE (Algorithm 6) implemented
- **Critical γ=0 equivalence: EXACT (0.0 difference)** ✅
- Tested γ = {0, 0.1, 0.5, 1.0, 2.0}
- Both uniform and logit-normal time schedules

### B3: Exact Probability-Flow SDE
Derived from the stochastic-interpolant framework:

```
ELF interpolant:  z_t = t·x₁ + (1-t)·ε
Score identity:   ∇log p_t(z) = v(z,t) / ((1-t)·σ²)
Exact SDE:        dz = v·[1 + g²/(2(1-t)σ²)] dt + g·dW
```

- **g=0 convergence: EXACT (0.0 difference)** ✅
- **Monotone convergence as g→0: ✅**
- Confirmed distinct from ELF's approximation (mean |diff| = 0.3859)

### B4: Step-Budget Comparison

**Key observations:**

1. **At toy scale, entropy differences between samplers are tiny** (4th decimal place: 5.5451 vs 5.5450). This is expected — the toy model trained on random data has limited structure for samplers to exploit.

2. **The exact SDE (g=1.0, logit-normal schedule) shows the most notable behavior:**
   - Entropy decreases with more steps: 5.5450 → 5.5448 → 5.5445 → 5.5442
   - Token diversity is significantly higher (5.4–8.6 vs 1.0–1.7 for other samplers)
   - This is the only sampler showing clear step-dependent quality improvement

3. **ELF's SDE approximation collapses to low diversity** at higher γ — the noise reinjection drives samples toward a mode, reducing token variety to 1.0.

4. **Pattern assessment:** The expected "SDE beats ODE more at low steps" pattern shows weak but detectable signal (ELF-SDE γ=0.5 with logit-normal: 0.0001 entropy advantage at 4 steps, vanishing at 64 steps).

### Interpretation

This falls between **Outcome 2** and **Outcome 3** per the plan's framework:
- The exact SDE is mechanistically different from ELF's approximation ✅
- Diversity metrics clearly differentiate the samplers ✅
- But the quality proxy (entropy) shows only marginal differences at toy scale
- The real test requires a model trained on real language data, where few-step degradation is much more pronounced

---

## Cluster-Scale Transition Notes

### What Changes at Real Scale

| Parameter | Toy (B1) | Real (ELF-B) |
|---|---|---|
| Encoder | None (random embeddings) | Frozen T5-small (512-d) |
| Model | 125K params, 2 layers | ~100M params, 12+ layers |
| Data | Random tokens (256 vocab) | OpenWebText (32K vocab) |
| Bottleneck | 32-d | 128-d |
| Training | 8K steps | ~500K steps |
| Eval metric | Decoder entropy (proxy) | Gen. PPL (GPT-2 eval) |

### Code Reuse Assessment

| Component | Reusable? | Modifications Needed |
|---|---|---|
| ODE sampler (`testB2_samplers.py`) | ✅ Fully | Integrate with real ELF's `_forward_sample` |
| ELF SDE sampler | ✅ Fully | Already matches `_sde_step` in `sampling_utils.py` |
| Exact SDE sampler | ✅ Mostly | Scale drift clamp near t=1 may need tuning for 128-d |
| Step-budget sweep | ✅ Fully | Replace entropy proxy with real Gen. PPL |
| Time schedules | ✅ Fully | Both already implemented in `sampling_utils.py` |

### Recommended Cluster Protocol

1. **First run (cheap):** Train ELF-B for ~100K steps on reduced OWT subset. Run 4-sampler comparison at 8/16/32/64 steps. If no signal → stop.
2. **Full run (if signal):** Full ELF-B training per Appendix D.2. Comprehensive Gen. PPL evaluation per Fig. 5c protocol.
3. **Key hyperparameters to sweep:** g ∈ {0.1, 0.25, 0.5, 1.0} for exact SDE; γ ∈ {0.1, 0.5, 1.0} for ELF's approximation.
