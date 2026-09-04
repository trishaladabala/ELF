#!/usr/bin/env python3
"""Phase P3 — Token-Level Mechanistic Analysis.

Uses Stage 2 data (1024 samples with per-token ODE/SDE entropy) to test:
1. Entropy-stratified analysis: does SDE help more at low-entropy (high-confidence) tokens?
2. Per-position analysis: does SDE help more at later positions?
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

OUT = Path(__file__).resolve().parent / "results" / "p3_mechanism"
S2 = Path(__file__).resolve().parent / "results" / "stage2"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Phase P3 — Token-Level Mechanistic Analysis")
    print("="*70)

    data = np.load(S2 / "stage2_metrics.npz")
    ode_ent = data["ode_entropy"]  # [N, L]
    sde_ent = data["sde_entropy"]  # [N, L]
    ode_degs = data["ode_degs"]
    sde_degs = data["sde_degs"]

    N, L = ode_ent.shape
    joint_ok = (~ode_degs) & (~sde_degs)
    n_ok = int(joint_ok.sum())
    print(f"  N={N}, L={L}, jointly non-degenerate: {n_ok}")

    ode_e = ode_ent[joint_ok]  # [n_ok, L]
    sde_e = sde_ent[joint_ok]  # [n_ok, L]

    # ── Analysis 1: Entropy-stratified SDE advantage ──
    # SDE advantage = ODE_entropy - SDE_entropy (positive = SDE more confident = better quality)
    # But what we really want: does SDE lower PPL more for tokens where ODE was over-confident (low entropy)?
    # Re-interpret: SDE advantage in terms of quality = SDE produces more calibrated entropy
    # For paper: bin by ODE entropy, show how SDE entropy differs

    ode_flat = ode_e.flatten()
    sde_flat = sde_e.flatten()
    diff_flat = sde_flat - ode_flat  # positive = SDE has higher entropy

    # 10 bins by ODE entropy
    n_bins = 10
    pcts = np.percentile(ode_flat, np.linspace(0, 100, n_bins + 1))
    bins = []
    for i in range(n_bins):
        lo, hi = pcts[i], pcts[i+1]
        mask = (ode_flat >= lo) & (ode_flat < hi) if i < n_bins - 1 else (ode_flat >= lo)
        if mask.sum() > 0:
            bins.append({
                "bin": i, "ode_ent_lo": float(lo), "ode_ent_hi": float(hi),
                "n_tokens": int(mask.sum()),
                "mean_ode_ent": float(ode_flat[mask].mean()),
                "mean_sde_ent": float(sde_flat[mask].mean()),
                "mean_diff": float(diff_flat[mask].mean()),
                "std_diff": float(diff_flat[mask].std()),
            })

    print("\n  ── Entropy-Stratified Analysis ──")
    print(f"  {'Bin':>4s}  {'ODE_ent':>8s}  {'SDE_ent':>8s}  {'Diff':>8s}  {'N_tok':>8s}")
    for b in bins:
        print(f"  {b['bin']:4d}  {b['mean_ode_ent']:8.4f}  {b['mean_sde_ent']:8.4f}  "
              f"{b['mean_diff']:+8.4f}  {b['n_tokens']:8d}")

    # Key insight: does SDE entropy increase more for low-ODE-entropy tokens?
    # If diff is more positive for low ODE entropy → SDE is regularizing overconfident tokens
    diffs_by_bin = [b["mean_diff"] for b in bins]
    ode_ents_by_bin = [b["mean_ode_ent"] for b in bins]
    r_stratified, p_stratified = sp_stats.spearmanr(ode_ents_by_bin, diffs_by_bin)
    print(f"\n  Spearman r(ODE_entropy, SDE−ODE_diff) across bins = {r_stratified:.3f}, p = {p_stratified:.3e}")
    print(f"  Interpretation: {'Negative r → SDE increases entropy more for low-entropy (overconfident) tokens = REGULARIZATION' if r_stratified < 0 else 'Positive r → SDE increases entropy more for HIGH-entropy tokens = NOT regularization'}")

    # ── Analysis 2: Per-position analysis ──
    # Does SDE advantage change with position (early vs late tokens)?
    mean_diff_by_pos = (sde_e - ode_e).mean(axis=0)  # [L]
    positions = np.arange(L)

    # Split into quartiles
    q_size = L // 4
    pos_quartiles = []
    for qi, label in enumerate(["Q1 (0-255)", "Q2 (256-511)", "Q3 (512-767)", "Q4 (768-1023)"]):
        start = qi * q_size
        end = start + q_size
        q_diff = mean_diff_by_pos[start:end]
        pos_quartiles.append({
            "quartile": label,
            "mean_diff": float(q_diff.mean()),
            "std_diff": float(q_diff.std()),
        })

    print("\n  ── Per-Position Analysis ──")
    for q in pos_quartiles:
        print(f"  {q['quartile']}: mean SDE−ODE entropy = {q['mean_diff']:+.5f}")

    r_pos, p_pos = sp_stats.spearmanr(positions, mean_diff_by_pos)
    print(f"\n  Spearman r(position, SDE−ODE_diff) = {r_pos:.3f}, p = {p_pos:.3e}")
    print(f"  {'SDE effect grows with position → error correction' if r_pos > 0.1 else 'SDE effect is position-independent → global regularization'}")

    # ── Analysis 3: Token-level quality proxy ──
    # For each token, measure "improvement" as |ODE_ent - natural| vs |SDE_ent - natural|
    # where "natural" ≈ 5.2 bits (typical English text entropy)
    NATURAL_ENT = 5.2  # bits, approx for natural English text
    ode_deviation = np.abs(ode_flat - NATURAL_ENT)
    sde_deviation = np.abs(sde_flat - NATURAL_ENT)
    improvement = ode_deviation - sde_deviation  # positive = SDE closer to natural

    imp_by_ode_ent = []
    for b in bins:
        lo, hi = b["ode_ent_lo"], b["ode_ent_hi"]
        idx = b["bin"]
        mask = (ode_flat >= lo) & (ode_flat < hi) if idx < n_bins - 1 else (ode_flat >= lo)
        if mask.sum() > 0:
            imp_by_ode_ent.append({
                "bin": idx,
                "mean_ode_ent": b["mean_ode_ent"],
                "mean_improvement": float(improvement[mask].mean()),
            })

    print("\n  ── Naturalness Improvement (closer to H=5.2 bits) ──")
    for b in imp_by_ode_ent:
        print(f"  Bin {b['bin']:2d} (ODE_ent={b['mean_ode_ent']:.4f}): "
              f"SDE improvement = {b['mean_improvement']:+.4f}")

    # Save results
    results = {
        "n_samples": n_ok, "seq_len": L,
        "entropy_stratified": bins,
        "stratified_spearman": {"r": float(r_stratified), "p": float(p_stratified)},
        "position_quartiles": pos_quartiles,
        "position_spearman": {"r": float(r_pos), "p": float(p_pos)},
        "naturalness_improvement": imp_by_ode_ent,
        "interpretation": {
            "regularization": bool(r_stratified < -0.3),
            "position_dependent": bool(abs(r_pos) > 0.1),
        }
    }
    with open(OUT / "p3_mechanism.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    with open(OUT / "p3_mechanism.md", "w") as f:
        f.write("# Phase P3 — Token-Level Mechanistic Analysis\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Entropy-Stratified Analysis\n\n")
        f.write("Does SDE entropy increase more for low-ODE-entropy tokens (= regularization)?\n\n")
        f.write("| Bin | ODE Entropy | SDE Entropy | Diff (SDE−ODE) | N tokens |\n|---|---|---|---|---|\n")
        for b in bins:
            f.write(f"| {b['bin']} | {b['mean_ode_ent']:.4f} | {b['mean_sde_ent']:.4f} | "
                    f"{b['mean_diff']:+.4f} | {b['n_tokens']:,} |\n")
        f.write(f"\nSpearman r = {r_stratified:.3f} (p = {p_stratified:.3e})\n\n")
        reg = "✅ YES" if r_stratified < -0.3 else "❌ NO" if r_stratified > 0.3 else "⚠️ WEAK"
        f.write(f"**Regularization hypothesis:** {reg}\n\n")
        f.write("## Per-Position Analysis\n\n")
        f.write("| Position | SDE−ODE Entropy |\n|---|---|\n")
        for q in pos_quartiles:
            f.write(f"| {q['quartile']} | {q['mean_diff']:+.5f} |\n")
        f.write(f"\nSpearman r = {r_pos:.3f}\n")

    print(f"\n  Saved → {OUT}")

if __name__ == "__main__":
    main()
