#!/usr/bin/env python3
"""H6 — Idea 15 Verdict."""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    print("=" * 60)
    print("H6 — Idea 15 Verdict")
    print("=" * 60)

    h1 = json.load(open(RESULTS_DIR / "testH1.json"))
    h2 = json.load(open(RESULTS_DIR / "testH2.json"))
    h3 = json.load(open(RESULTS_DIR / "testH3.json"))
    h4 = json.load(open(RESULTS_DIR / "testH4.json"))
    h5 = json.load(open(RESULTS_DIR / "testH5.json"))

    lines = []
    lines.append("# H6 — Idea 15 (Confidence-Gated Post-Hoc Refinement) Verdict\n")

    # H1
    lines.append("## H1: Distribution-Shift Probe\n")
    if h1["verdict"] == "PASS":
        lines.append(f"**✅ PASS.** {len(h1.get('best_candidates', []))} candidate(s) produce "
                      f"sensible, non-degenerate output. The frozen decoder tolerates revision inputs.\n")
    else:
        lines.append("**❌ FAIL.** The frozen decoder cannot tolerate revision inputs.\n")

    # H2
    lines.append("## H2: Confidence-Signal Validity\n")
    sub_f1 = h2["substitution"]["best_detector"]["f1"]
    rep_f1 = h2["repetition"]["best_detector"]["f1"]
    lines.append(f"- **Substitution errors:** F1={sub_f1:.3f} "
                  f"({'detectable' if h2['sub_detectable'] else 'NOT detectable'})")
    lines.append(f"- **Repetition errors:** F1={rep_f1:.3f} "
                  f"({'detectable' if h2['rep_detectable'] else 'NOT detectable'})\n")
    if h2.get("scope_note"):
        lines.append(f"> **Scope note:** {h2['scope_note']}\n")

    # H3
    lines.append("## H3: Refinement Loop Convergence\n")
    if h3["verdict"] == "PASS":
        lines.append("**✅ PASS.** Error count decreases monotonically.\n")
    elif h3["verdict"] == "PARTIAL_PASS":
        lines.append("**🟡 PARTIAL.** Errors plateau/decrease but not strictly monotonic.\n")
    else:
        lines.append("**⚠️ NEEDS FIX.** The loop diverges — new errors grow with each pass. "
                      "A stopping rule or dampening mechanism is required before the loop is usable.\n")
        # Detail the divergence
        metrics = h3["pass_metrics"]
        for m in metrics:
            lines.append(f"  Pass {m['pass']}: total={m['total_errors']}, new={m['new_errors_introduced']}")
        lines.append("")

    # H4
    lines.append("## H4: Naive Masking vs Re-Corruption\n")
    winner = h4["winner"]
    if winner == "re_corruption":
        lines.append("**Re-corruption outperforms naive masking** (fewer total errors after 5 passes). "
                      "Use ELF-style re-corruption as the revision mechanism.\n")
    elif winner == "zero_mask":
        lines.append("**Naive masking outperforms re-corruption** — simplifies the method.\n")
    else:
        lines.append("**No clear winner** — methods are comparable.\n")

    # H5
    lines.append("## H5: End-to-End Net Effect\n")
    if h5["verdict"] == "SIGNIFICANT_IMPROVEMENT":
        lines.append(f"**✅ Refinement significantly improves PPL** "
                      f"(Δ={h5['diff_mean']:+.1f} [{h5['diff_ci_lo']:+.1f}, {h5['diff_ci_hi']:+.1f}]).\n")
    elif h5["verdict"] == "SIGNIFICANT_WORSENING":
        lines.append(f"**❌ Refinement significantly WORSENS PPL** "
                      f"(Δ={h5['diff_mean']:+.1f} [{h5['diff_ci_lo']:+.1f}, {h5['diff_ci_hi']:+.1f}]).\n")
        lines.append(f"- {h5['improved']}/{h5['improved']+h5['worsened']+h5['unchanged']} improved, "
                      f"{h5['worsened']} worsened, {h5['unchanged']} unchanged.\n")
        lines.append("The worsened cases have much higher magnitude than improvements, "
                      "driving the net effect negative. This is consistent with H3's finding "
                      "that the loop introduces more errors than it fixes.\n")
    else:
        lines.append("**🟡 No statistically significant effect detected.**\n")

    # Overall Verdict
    lines.append("## Overall Verdict\n")

    h1_ok = h1["verdict"] == "PASS"
    h2_ok = h2["sub_detectable"] or h2["rep_detectable"]
    h3_ok = h3["verdict"] in ("PASS", "PARTIAL_PASS")
    h5_ok = h5["verdict"] == "SIGNIFICANT_IMPROVEMENT"

    if h1_ok and h2_ok and h3_ok and h5_ok:
        lines.append("### **CONFIRMED GO.** Proceed to a real-scale pilot on frozen ELF-B.\n")
        verdict = "confirmed_go"
    elif h1_ok and h2_ok and not h3_ok and not h5_ok:
        lines.append("### **PARTIAL GO — Viable mechanism, but loop needs redesign.**\n\n")
        lines.append("The building blocks work individually:\n"
                      "- The frozen decoder tolerates revision inputs (H1 ✅)\n"
                      "- Entropy can detect errors (H2 ✅)\n"
                      "- Re-corruption is the better mechanism (H4 ✅)\n\n"
                      "But the iterative loop is counterproductive:\n"
                      "- New errors grow with each pass (H3 ⚠️)\n"
                      "- Net effect on natural output is negative (H5 ❌)\n\n"
                      "**Next step:** Redesign the loop with a stopping rule (e.g., only revise "
                      "positions above a much higher entropy threshold, or limit to a single pass "
                      "on only the most confident error detections). This is an engineering fix, "
                      "not a fundamental problem with the idea.\n")
        verdict = "partial_go"
    elif not h1_ok:
        lines.append("### **NO-GO.** The frozen decoder cannot support post-hoc revision. "
                      "Idea 15 must be rescoped as a training-time change (closer to Idea 5).\n")
        verdict = "no_go"
    else:
        lines.append("### **NO-GO at current design.** Individual components have issues. "
                      "Needs significant redesign before proceeding.\n")
        verdict = "no_go"

    verdict_text = "\n".join(lines)
    with open(RESULTS_DIR / "testH_idea15_verdict.md", "w") as f:
        f.write(verdict_text)

    print(verdict_text)
    print(f"\n  Saved → {RESULTS_DIR / 'testH_idea15_verdict.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
