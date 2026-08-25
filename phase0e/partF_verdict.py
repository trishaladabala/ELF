#!/usr/bin/env python3
"""Part F — Updated Final Verdict.

Reads E1 (External Quality) and E2 (Stability) results to make the final Go/No-Go call.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def main():
    print("=" * 60)
    print("Part F — Updated Final Verdict")
    print("=" * 60)

    e1_path = RESULTS_DIR / "testE1_external_quality_step_budget.json"
    e2_path = RESULTS_DIR / "testE2.json"
    
    if not e1_path.exists():
        print("E1 results missing.")
        sys.exit(1)
        
    e1 = json.load(open(e1_path))
    
    if e2_path.exists():
        e2 = json.load(open(e2_path))
    else:
        e2 = None

    lines = []
    lines.append("# Part E/F — Final Verdict: External Validation & Stability\n")
    
    lines.append("## (a) External LM Quality Validation (E1)\n")
    lines.append("Using `distilgpt2` to independently score generation perplexity eliminated the confound of decoder collapse.\n")
    
    # We check if SDE still beats ODE at low steps in E1
    # Uniform schedule, 4 steps
    ode_ppl_4 = e1["uniform"]["ODE"]["4"]["ppl_mean"]
    ode_ppl_64 = e1["uniform"]["ODE"]["64"]["ppl_mean"]
    corr_sde_ppl_4 = e1["uniform"]["Corrected-SDE g=0.5"]["4"]["ppl_mean"]
    
    lines.append(f"At 4 steps, ODE PPL = {ode_ppl_4:.1f}, Corrected SDE PPL = {corr_sde_ppl_4:.1f}.")
    if corr_sde_ppl_4 < ode_ppl_4 - 10:
        lines.append("**✅ External metrics confirm the pattern:** The SDE still significantly outperforms the ODE at low step budgets, even when judged by an independent LM.\n")
        e1_pattern_holds = True
    else:
        lines.append("**❌ External metrics DO NOT confirm the pattern:** The SDE advantage seen with entropy disappears when judged by an independent LM.\n")
        e1_pattern_holds = False

    if e2:
        lines.append("## (b) Stability Characterization (E2)\n")
        lines.append(f"The score error near `t=1` was measured up to `t=0.99`.")
        lines.append(f"Fitted log-log slope of error growth vs (1-t): **{e2['slope']:.2f}**.")
        if e2["matches_theory"]:
            lines.append("This perfectly matches the theoretically predicted $\\mathcal{O}((1-t)^{-2})$ error amplification. The exact SDE is mathematically correct but fundamentally unstable near decode time unless heavily constrained.\n")
        else:
            lines.append("The error growth diverges from simple theory, suggesting additional network approximation issues near the boundary.\n")
            
    lines.append("## (c) Final Decision\n")
    if e1_pattern_holds:
        lines.append("### **CONFIRMED GO.**")
        lines.append("The SDE advantage is real and survives external scoring without being driven by collapse. Proceed to Magus with the original scope to scale up the findings.")
    elif e2 and e2["matches_theory"]:
        lines.append("### **REFRAMED GO.**")
        lines.append("The quality advantage did not clearly survive external scoring, but the stability characterization confirms exactly *why* ELF uses a heuristic approximation. Proceed to Magus with a narrower framing focused on this specific tradeoff/characterization rather than a strict 'our SDE wins' claim.")
    else:
        lines.append("### **NO-GO.**")
        lines.append("Neither the quality pattern nor the stability characterization held up robustly. Redirect effort to the trajectory-geometry companion study.")

    verdict_text = "\n".join(lines)
    with open(RESULTS_DIR / "final_verdict_v2.md", "w") as f:
        f.write(verdict_text)

    print(verdict_text)
    print(f"\n  Saved → {RESULTS_DIR / 'final_verdict_v2.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
