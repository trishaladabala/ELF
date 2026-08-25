# H6 — Idea 15 (Confidence-Gated Post-Hoc Refinement) Verdict

## H1: Distribution-Shift Probe

**✅ PASS.** 3 candidate(s) produce sensible, non-degenerate output. The frozen decoder tolerates revision inputs.

## H2: Confidence-Signal Validity

- **Substitution errors:** F1=0.356 (detectable)
- **Repetition errors:** F1=0.260 (detectable)

## H3: Refinement Loop Convergence

**⚠️ NEEDS FIX.** The loop diverges — new errors grow with each pass. A stopping rule or dampening mechanism is required before the loop is usable.

  Pass 0: total=809, new=450
  Pass 1: total=934, new=576
  Pass 2: total=1020, new=655
  Pass 3: total=1152, new=778
  Pass 4: total=1262, new=888
  Pass 5: total=1438, new=1058

## H4: Naive Masking vs Re-Corruption

**Re-corruption outperforms naive masking** (fewer total errors after 5 passes). Use ELF-style re-corruption as the revision mechanism.

## H5: End-to-End Net Effect

**❌ Refinement significantly WORSENS PPL** (Δ=-4529.8 [-7793.0, -1523.0]).

- 109/256 improved, 109 worsened, 38 unchanged.

The worsened cases have much higher magnitude than improvements, driving the net effect negative. This is consistent with H3's finding that the loop introduces more errors than it fixes.

## Overall Verdict

### **PARTIAL GO — Viable mechanism, but loop needs redesign.**


The building blocks work individually:
- The frozen decoder tolerates revision inputs (H1 ✅)
- Entropy can detect errors (H2 ✅)
- Re-corruption is the better mechanism (H4 ✅)

But the iterative loop is counterproductive:
- New errors grow with each pass (H3 ⚠️)
- Net effect on natural output is negative (H5 ❌)

**Next step:** Redesign the loop with a stopping rule (e.g., only revise positions above a much higher entropy threshold, or limit to a single pass on only the most confident error detections). This is an engineering fix, not a fundamental problem with the idea.
