# Part E/F — Final Verdict: External Validation & Stability

## (a) External LM Quality Validation (E1)

Using `distilgpt2` to independently score generation perplexity eliminated the confound of decoder collapse.

At 4 steps, ODE PPL = 6394.3, Corrected SDE PPL = 8234.2.
**❌ External metrics DO NOT confirm the pattern:** The SDE advantage seen with entropy disappears when judged by an independent LM.

## (b) Stability Characterization (E2)

The score error near `t=1` was measured up to `t=0.99`.
Fitted log-log slope of error growth vs (1-t): **-1.50**.
The error growth diverges from simple theory, suggesting additional network approximation issues near the boundary.

## (c) Final Decision

### **NO-GO.**
Neither the quality pattern nor the stability characterization held up robustly. Redirect effort to the trajectory-geometry companion study.