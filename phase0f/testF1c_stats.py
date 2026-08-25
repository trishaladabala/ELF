#!/usr/bin/env python3
import json
import sys
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    in_path = RESULTS_DIR / "testF1b_curvature_by_group.json"
    with open(in_path, "r") as f:
        groups = json.load(f)
        
    always_c = np.array(groups["always_fails"]["curvatures"])
    some_c = np.array(groups["sometimes_fails"]["curvatures"])
    never_c = np.array(groups["never_fails"]["curvatures"])
    
    always_m = np.array(groups["always_fails"]["noise_mags"])
    some_m = np.array(groups["sometimes_fails"]["noise_mags"])
    never_m = np.array(groups["never_fails"]["noise_mags"])
    
    # Kruskal-Wallis tests
    h_stat_c, p_val_c = stats.kruskal(always_c, some_c, never_c)
    h_stat_m, p_val_m = stats.kruskal(always_m, some_m, never_m)
    
    # Mann-Whitney U tests for pairwise (Always vs Never)
    u_stat_c, p_val_c_pair = stats.mannwhitneyu(always_c, never_c, alternative='two-sided')
    u_stat_m, p_val_m_pair = stats.mannwhitneyu(always_m, never_m, alternative='two-sided')
    
    # Summary stats
    def summarize(arr):
        if len(arr) == 0: return 0.0, 0.0
        return np.mean(arr), np.median(arr)
        
    mean_c_a, med_c_a = summarize(always_c)
    mean_c_s, med_c_s = summarize(some_c)
    mean_c_n, med_c_n = summarize(never_c)
    
    mean_m_a, med_m_a = summarize(always_m)
    mean_m_s, med_m_s = summarize(some_m)
    mean_m_n, med_m_n = summarize(never_m)
    
    out_lines = [
        "# Part F1c: Hard-Seed Statistical Analysis",
        "",
        "## Curvature Comparison (Turning Angle)",
        f"- **Always-Fails (N={len(always_c)}):** Mean = {mean_c_a:.4f}, Median = {med_c_a:.4f}",
        f"- **Sometimes-Fails (N={len(some_c)}):** Mean = {mean_c_s:.4f}, Median = {med_c_s:.4f}",
        f"- **Never-Fails (N={len(never_c)}):** Mean = {mean_c_n:.4f}, Median = {med_c_n:.4f}",
        "",
        f"**Kruskal-Wallis H-test (Across all 3):** H={h_stat_c:.4f}, p={p_val_c:.2e}",
        f"**Mann-Whitney U-test (Always > Never):** U={u_stat_c:.1f}, p={p_val_c_pair:.2e}",
        "",
        "## Initial Noise Magnitude Comparison (L2 Norm)",
        f"- **Always-Fails (N={len(always_m)}):** Mean = {mean_m_a:.4f}, Median = {med_m_a:.4f}",
        f"- **Sometimes-Fails (N={len(some_m)}):** Mean = {mean_m_s:.4f}, Median = {med_m_s:.4f}",
        f"- **Never-Fails (N={len(never_m)}):** Mean = {mean_m_n:.4f}, Median = {med_m_n:.4f}",
        "",
        f"**Kruskal-Wallis H-test (Across all 3):** H={h_stat_m:.4f}, p={p_val_m:.2e}",
        f"**Mann-Whitney U-test (Always vs Never):** U={u_stat_m:.1f}, p={p_val_m_pair:.2e}",
        "",
        "## Interpretation",
    ]
    
    sig_curvature = p_val_c_pair < 0.05
    sig_noise = p_val_m_pair < 0.05
    
    if sig_curvature and not sig_noise:
        direction = "higher" if mean_c_a > mean_c_n else "lower"
        out_lines.append(f"**GO:** Always-fails seeds show significantly {direction} curvature than never-fails seeds, and this is NOT explained by a difference in initial noise magnitude. This provides strong, independent support for the trajectory geometry playing a key role in failure (hard seeds have {direction} curvature).")
    elif sig_curvature and sig_noise:
        out_lines.append("**NO-GO (Confounded):** While always-fails seeds show significantly different curvature, they ALSO differ significantly in raw initial noise magnitude. The hard seeds might just be extreme noise vectors, making curvature a correlated symptom rather than the definitive cause.")
    elif not sig_curvature:
        out_lines.append("**NO-GO (Negative):** There is no significant difference in curvature between the hard seeds and the easy seeds. The curvature metric does not explain why these 15 seeds consistently fail.")
        
    out_path = RESULTS_DIR / "testF1c_statistical_test.md"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        
    print(f"Saved to {out_path}")
    
if __name__ == "__main__":
    main()
