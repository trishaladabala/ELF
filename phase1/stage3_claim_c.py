#!/usr/bin/env python3
"""Stage 3 — Claim C: Curvature-Driven Overconfidence.

Loads per-token curvature and entropy from Stage 2, computes Spearman
correlation between ODE curvature and SDE entropy advantage.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

OUT = Path(__file__).resolve().parent / "results" / "stage3"
S2 = Path(__file__).resolve().parent / "results" / "stage2"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Stage 3 — Claim C: Curvature-Driven Overconfidence")
    print("="*70)

    data = np.load(S2 / "stage2_metrics.npz")
    curv = data["curvature"]       # [N, L]
    ode_ent = data["ode_entropy"]  # [N, L]
    sde_ent = data["sde_entropy"]  # [N, L]
    ode_degs = data["ode_degs"]
    sde_degs = data["sde_degs"]

    N, L = curv.shape
    joint_ok = (~ode_degs) & (~sde_degs)
    n_ok = int(joint_ok.sum())
    print(f"  N={N}, L={L}, jointly non-degenerate: {n_ok}")

    # Entropy advantage: SDE entropy - ODE entropy (positive = SDE more confident = lower entropy)
    # Actually: ODE entropy - SDE entropy (positive = SDE has lower entropy = more confident)
    ent_adv = ode_ent - sde_ent  # positive = SDE more confident

    # Flatten across non-degenerate samples and all token positions
    curv_flat = curv[joint_ok].flatten()
    ent_flat = ent_adv[joint_ok].flatten()

    # Remove NaN/inf
    valid = np.isfinite(curv_flat) & np.isfinite(ent_flat) & (curv_flat > 0)
    curv_v = curv_flat[valid]
    ent_v = ent_flat[valid]
    print(f"  Valid token pairs: {len(curv_v):,}")

    # Spearman correlation
    r, p = sp_stats.spearmanr(curv_v, ent_v)
    print(f"\n  Spearman r = {r:.4f}, p = {p:.2e}")

    # Per-sequence correlation (more robust)
    seq_rs = []
    for i in np.where(joint_ok)[0]:
        c = curv[i]; e = ent_adv[i]
        v = np.isfinite(c) & np.isfinite(e) & (c > 0)
        if v.sum() > 10:
            sr, _ = sp_stats.spearmanr(c[v], e[v])
            if np.isfinite(sr):
                seq_rs.append(sr)
    mean_seq_r = float(np.mean(seq_rs)) if seq_rs else 0
    print(f"  Mean per-sequence Spearman r = {mean_seq_r:.4f} (over {len(seq_rs)} seqs)")

    # Verdict
    replicates = r >= 0.10 and p < 0.01
    print(f"\n  Claim C {'REPLICATES' if replicates else 'DOES NOT REPLICATE'}")
    print(f"  (bar: r >= +0.10 and p < 0.01)")

    results = {
        "spearman_r_flat": float(r), "spearman_p_flat": float(p),
        "n_valid_pairs": int(valid.sum()),
        "mean_per_seq_r": mean_seq_r, "n_seqs": len(seq_rs),
        "replicates": bool(replicates),
    }
    with open(OUT / "claim_c_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "claim_c_verdict.md", "w") as f:
        f.write("# Stage 3 — Claim C Verdict\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Spearman r (flat) | {r:.4f} |\n")
        f.write(f"| p-value | {p:.2e} |\n")
        f.write(f"| Mean per-seq r | {mean_seq_r:.4f} |\n")
        f.write(f"| N valid pairs | {int(valid.sum()):,} |\n\n")
        f.write(f"**Verdict:** {'✅ REPLICATES' if replicates else '❌ DOES NOT REPLICATE'}\n")
        f.write(f"(bar: Spearman r ≥ +0.10, p < 0.01)\n\n")
        if not replicates:
            f.write("> Scale trend predicted this: r=+0.38 at 275K → +0.13 at 1M → −0.08 at 5M.\n")

    print(f"  Saved → {OUT}")

if __name__ == "__main__":
    main()
