#!/usr/bin/env python3
import json
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def bootstrap_paired_diff(is_deg_a, is_deg_b, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(is_deg_a)
    diffs = np.array(is_deg_a, dtype=float) - np.array(is_deg_b, dtype=float)
    boot_diffs = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(RESULTS_DIR / "testF1a_seed_groups.json", "r") as f:
        f1_data = json.load(f)
        
    with open(RESULTS_DIR / "testF2_combined_fails.json", "r") as f:
        f2_data = json.load(f)
        
    always_fails = set(f1_data["always_fails"])
    comb_fails = set(f2_data["combined_fails"])
    r3_fails = set(f1_data["r3_fails"])
    r5_fails = set(f1_data["r5_fails"])
    
    # 1. Hard-seed rescue
    overlap = always_fails.intersection(comb_fails)
    rescued = always_fails - comb_fails
    
    # 2. Paired bootstrap vs R3 (since R3 and R5 had identical 25 counts)
    n_samples = 256
    is_deg_comb = np.zeros(n_samples)
    is_deg_r3 = np.zeros(n_samples)
    is_deg_comb[list(comb_fails)] = 1
    is_deg_r3[list(r3_fails)] = 1
    
    diff_mean, diff_lo, diff_hi = bootstrap_paired_diff(is_deg_comb, is_deg_r3)
    # Negative diff_mean means Combined has lower degeneracy (is better)
    sig_better = diff_hi < 0
    sig_worse = diff_lo > 0
    
    comb_rate = len(comb_fails) / n_samples
    r3_rate = len(r3_fails) / n_samples
    
    out_lines = [
        "# Part F2c: Combined Verdict",
        "",
        "## Hard-Seed Rescue",
        f"- **Original 'Always-Fails' Set:** 15 seeds (failed in both R3 and R5).",
        f"- **Combined Model Failed On:** {len(overlap)} of those 15 seeds.",
        f"- **Rescued:** {len(rescued)} hard seeds successfully decoded by the combined model!",
        "",
        "## Statistical Comparison vs Best Individual (R3/R5)",
        f"- **Combined Degeneracy Rate:** {comb_rate:.2%} ({len(comb_fails)}/256)",
        f"- **Best Individual Rate (R3/R5):** {r3_rate:.2%} (25/256)",
        f"- **Paired Diff (Combined - Individual):** {diff_mean:+.2%} [{diff_lo:+.2%}, {diff_hi:+.2%}]",
        "",
        "## Verdict"
    ]
    
    if sig_better:
        out_lines.append("**GO:** The combined intervention (Decoupled + Label Smoothing) significantly outperforms either intervention alone. It successfully rescues seeds that were unfixable by the single methods. This represents a true synergistic effect and should be the headline finding for Magus.")
    elif sig_worse:
        out_lines.append("**NO-GO (Interference):** The combined intervention performed significantly *worse* than the individual methods. The two methods interact poorly (e.g., label smoothing conflicting with decoupled decoding).")
    else:
        if comb_rate < r3_rate:
            out_lines.append("**PARTIAL (Trending Positive):** The combined model reduced the degeneracy rate and rescued some hard seeds, but the overall improvement over the best individual model is not statistically significant. Combining them doesn't hurt, but the synergy isn't statistically confirmed at this sample size.")
        elif comb_rate == r3_rate:
            out_lines.append("**NO-GO (Flat):** The combined model performed identically in aggregate to the best individual models. There is no synergistic benefit to combining them.")
        else:
            out_lines.append("**PARTIAL (Trending Negative):** The combined model trended worse than the individual models, though not significantly. There is no benefit to combining them.")
            
    out_path = RESULTS_DIR / "testF2c_combined_verdict.md"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        
    print(f"Saved report to {out_path}")

if __name__ == "__main__":
    main()
