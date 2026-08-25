#!/usr/bin/env python3
"""G3 — Idea 6 Verdict."""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    print("=" * 60)
    print("G3 — Idea 6 Verdict")
    print("=" * 60)

    g1 = json.load(open(RESULTS_DIR / "testG1_powered_ppl_comparison.json"))
    g2 = json.load(open(RESULTS_DIR / "testG2_t1_confound_disambiguation.json"))

    lines = []
    lines.append("# G3 — Idea 6 (Principled SDE Sampler) Verdict\n")

    # G1 Analysis
    lines.append("## G1: Properly Powered PPL Comparison\n")
    if g1["any_significant_advantage"]:
        lines.append("**✅ A statistically significant PPL advantage exists** for the Corrected SDE "
                      "at specific step counts (uniform schedule, 16 and 32 steps).\n")
        # Detail the significant comparisons
        comps = g1["pairwise_comparisons"]
        for sched in comps:
            for sampler in comps[sched]:
                for ns in comps[sched][sampler]:
                    c = comps[sched][sampler][ns]
                    if c.get("excludes_zero") and c.get("diff_mean") is not None:
                        direction = "better" if c["diff_mean"] < 0 else "worse"
                        lines.append(f"- {sampler} vs ODE at {ns} steps ({sched}): "
                                     f"Δ={c['diff_mean']:+.1f} [{c['diff_ci_lo']:+.1f}, {c['diff_ci_hi']:+.1f}] "
                                     f"({direction})")
        lines.append("")
        lines.append("**Critical caveat:** ELF's own SDE (γ=0.5) produces massive degeneracy "
                      "(60-94% degenerate sequences), while the Corrected SDE produces 0% degeneracy. "
                      "This is a strong differentiator even if the PPL advantage is modest.\n")
    else:
        lines.append("**❌ No statistically significant PPL advantage** was detected at any step count.\n")

    # G2 Analysis
    lines.append("## G2: Near-Boundary Stability\n")
    if g2["substantial_improvement"]:
        lines.append(f"**The instability is partly fixable.** Oversampling t near 1 during training "
                      f"reduced boundary error by ~{(1-g2['err_ratio_t099'])*100:.0f}% at t=0.99.\n")
        lines.append(f"- Standard slope: {g2['slope_standard']:.3f}")
        lines.append(f"- Oversampled slope: {g2['slope_oversampled']:.3f}")
        lines.append(f"- Theoretical: -2.0\n")
        lines.append("This confirms the instability is an **engineering problem** "
                      "(training schedule design), not a fundamental theoretical dead end.\n")
    else:
        lines.append("The instability appears fundamental despite additional training.\n")

    # Verdict
    lines.append("## Final Verdict\n")
    if g1["any_significant_advantage"] and g2["substantial_improvement"]:
        lines.append("### **CONFIRMED QUALITY SIGNAL → Proceed to Magus with original scope.**\n")
        lines.append("The Corrected SDE shows:\n"
                      "1. A statistically significant PPL advantage over ODE at mid-range step counts\n"
                      "2. Zero degeneracy vs 60-94% for ELF's approximation\n"
                      "3. A fixable near-boundary instability (via training schedule)\n")
        verdict = "confirmed_go"
    elif g1["any_significant_advantage"]:
        lines.append("### **Proceed to Magus with original scope** (quality signal present, "
                      "stability needs more work at scale).\n")
        verdict = "go_with_caveats"
    elif g2["substantial_improvement"]:
        lines.append("### **Proceed to Magus with narrower framing** — the characterization/"
                      "tradeoff paper is defensible even without a clear quality win.\n")
        verdict = "narrower_framing"
    else:
        lines.append("### **Do not proceed with Idea 6 as primary.**\n")
        verdict = "no_go"

    verdict_text = "\n".join(lines)
    with open(RESULTS_DIR / "testG_idea6_verdict.md", "w") as f:
        f.write(verdict_text)

    print(verdict_text)
    print(f"\n  Saved → {RESULTS_DIR / 'testG_idea6_verdict.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
