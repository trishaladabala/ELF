#!/usr/bin/env python3
"""Stage 4 — Claim D: Hard-Seed Geometric Signature.

Tests whether ODE-degenerate seeds have measurably different trajectory
curvature than non-degenerate seeds.
"""
import json, sys, time, math
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

OUT = Path(__file__).resolve().parent / "results" / "stage4"
S2 = Path(__file__).resolve().parent / "results" / "stage2"

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    if pooled == 0: return 0.0
    return (np.mean(a) - np.mean(b)) / pooled

def rank_biserial(U, n1, n2):
    return 1 - 2*U / (n1 * n2)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Stage 4 — Claim D: Hard-Seed Geometric Signature")
    print("="*70)

    data = np.load(S2 / "stage2_metrics.npz")
    ode_degs = data["ode_degs"]
    per_sample_curv = data["per_sample_curvature"]  # [N]

    N = len(ode_degs)
    n_deg = int(ode_degs.sum())
    n_nondeg = N - n_deg
    deg_rate = n_deg / N

    print(f"  N={N}, ODE degenerate: {n_deg} ({100*deg_rate:.1f}%)")

    # Check if Claim A met (>1% degeneracy)
    if deg_rate < 0.01:
        print(f"\n  ⚠️ Claim D AUTO-VOIDED: ODE degeneracy < 1% ({100*deg_rate:.1f}%)")
        print(f"  Not enough degenerate seeds to partition.")
        results = {"auto_voided": True, "reason": "ODE degeneracy < 1%",
                    "deg_rate": deg_rate, "n_deg": n_deg}
        with open(OUT / "claim_d_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(OUT / "claim_d_verdict.md", "w") as f:
            f.write("# Stage 4 — Claim D Verdict\n\n")
            f.write(f"**AUTO-VOIDED:** ODE degeneracy = {100*deg_rate:.1f}% < 1% bar.\n")
            f.write("Insufficient degenerate seeds for geometric partition analysis.\n")
        print(f"  Saved → {OUT}")
        return

    deg_curv = per_sample_curv[ode_degs]
    nondeg_curv = per_sample_curv[~ode_degs]

    print(f"  Degenerate mean curvature: {np.mean(deg_curv):.6f} ± {np.std(deg_curv):.6f}")
    print(f"  Non-degenerate mean curvature: {np.mean(nondeg_curv):.6f} ± {np.std(nondeg_curv):.6f}")

    # Mann-Whitney U test
    U, p = sp_stats.mannwhitneyu(deg_curv, nondeg_curv, alternative='two-sided')
    d = cohens_d(deg_curv, nondeg_curv)
    rb = rank_biserial(U, n_deg, n_nondeg)

    print(f"\n  Mann-Whitney U = {U:.0f}, p = {p:.2e}")
    print(f"  Cohen's d = {d:.2f}")
    print(f"  Rank-biserial r = {rb:.3f}")

    # Confound check: initial noise magnitude
    # (We don't have z_0 norms saved, but we can note this)
    replicates = p < 0.01 and abs(d) >= 0.50
    print(f"\n  Claim D {'REPLICATES' if replicates else 'DOES NOT REPLICATE'}")
    print(f"  (bar: p < 0.01 AND |d| >= 0.50)")

    results = {
        "n_deg": n_deg, "n_nondeg": n_nondeg,
        "deg_curv_mean": float(np.mean(deg_curv)),
        "nondeg_curv_mean": float(np.mean(nondeg_curv)),
        "mann_whitney_U": float(U), "p_value": float(p),
        "cohens_d": float(d), "rank_biserial": float(rb),
        "replicates": replicates,
    }
    with open(OUT / "claim_d_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "claim_d_verdict.md", "w") as f:
        f.write("# Stage 4 — Claim D Verdict\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"| Group | N | Mean Curvature |\n|---|---|---|\n")
        f.write(f"| Degenerate | {n_deg} | {np.mean(deg_curv):.6f} ± {np.std(deg_curv):.6f} |\n")
        f.write(f"| Non-degenerate | {n_nondeg} | {np.mean(nondeg_curv):.6f} ± {np.std(nondeg_curv):.6f} |\n\n")
        f.write(f"| Statistic | Value |\n|---|---|\n")
        f.write(f"| Mann-Whitney U | {U:.0f} |\n")
        f.write(f"| p-value | {p:.2e} |\n")
        f.write(f"| Cohen's d | {d:.2f} |\n")
        f.write(f"| Rank-biserial r | {rb:.3f} |\n\n")
        f.write(f"**Verdict:** {'✅ REPLICATES' if replicates else '❌ DOES NOT REPLICATE'}\n")
        f.write(f"(bar: Mann-Whitney p < 0.01 AND |Cohen's d| ≥ 0.50)\n")

    print(f"  Saved → {OUT}")

if __name__ == "__main__":
    main()
