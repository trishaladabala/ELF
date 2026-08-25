#!/usr/bin/env python3
"""Test 4 — Token-Level vs. Sequence-Level Coupling Structure.

Repeats Test 2's OT-vs-random comparison at the *token position* level (per-position
groups) to check whether the coupling gain holds at the granularity ELF actually
operates at (per-token), or is purely a sequence-aggregation artifact.

Go/No-Go:
  Go:    Token-level OT improvement ≥ 50% of sequence-level improvement.
  No-Go: Token-level improvement is negligible — coupling gain evaporates per-token.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_SUBSAMPLE_PER_POS = 800   # Points per position group
SINKHORN_REG = 0.05


def load_data():
    x_128 = np.load(RESULTS_DIR / "x_128.npy")
    seq_ids = np.load(RESULTS_DIR / "seq_ids.npy")
    pos_ids = np.load(RESULTS_DIR / "pos_ids.npy")
    print(f"Loaded: x_128={x_128.shape}, seq_ids={seq_ids.shape}, pos_ids={pos_ids.shape}")
    return x_128, seq_ids, pos_ids


def load_test02_results():
    with open(RESULTS_DIR / "test_02.json") as f:
        return json.load(f)


def ot_vs_random_for_group(x: np.ndarray, reg: float) -> dict:
    """Compute OT vs random coupling cost for a group of embeddings."""
    import ot as pot

    rng = np.random.default_rng(42)
    n = x.shape[0]
    eps = rng.standard_normal(x.shape)

    # Random cost (MC)
    costs = []
    for _ in range(30):
        perm = rng.permutation(n)
        c = np.sum((x - eps[perm]) ** 2, axis=1).mean()
        costs.append(c)
    random_cost = float(np.mean(costs))

    # OT cost
    C = pot.dist(x, eps, metric="sqeuclidean")
    C_norm = C / (C.max() + 1e-10)
    a = np.ones(n) / n
    b = np.ones(n) / n
    T = pot.sinkhorn(a, b, C_norm, reg=reg, numItermax=2000, stopThr=1e-9)
    ot_cost = float(np.sum(T * C))

    ratio = ot_cost / random_cost if random_cost > 0 else 1.0
    reduction_pct = (1 - ratio) * 100

    return {
        "n": n,
        "random_cost": random_cost,
        "ot_cost": ot_cost,
        "ratio": ratio,
        "reduction_pct": reduction_pct,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 4 — Token-Level vs. Sequence-Level Coupling")
    print("=" * 60)

    x_128, seq_ids, pos_ids = load_data()
    test02 = load_test02_results()
    seq_level_reduction = test02["reduction_pct"]
    print(f"\nSequence-level reduction from Test 2: {seq_level_reduction:.2f}%")

    # Group tokens by relative position
    unique_positions = np.unique(pos_ids)
    print(f"Unique token positions: {len(unique_positions)} "
          f"(range {unique_positions.min()}-{unique_positions.max()})")

    # Select a representative set of positions (early, middle, late)
    pos_bins = [
        ("early (0-4)", np.isin(pos_ids, range(0, 5))),
        ("mid (10-14)", np.isin(pos_ids, range(10, 15))),
        ("late (25-29)", np.isin(pos_ids, range(25, 30))),
        ("all_positions", np.ones(len(pos_ids), dtype=bool)),
    ]

    rng = np.random.default_rng(42)
    results_per_group = {}

    for name, mask in pos_bins:
        indices = np.where(mask)[0]
        if len(indices) < 50:
            print(f"\n  Skipping '{name}': only {len(indices)} tokens")
            continue

        n = min(N_SUBSAMPLE_PER_POS, len(indices))
        sub_idx = rng.choice(indices, n, replace=False)
        x_group = x_128[sub_idx].astype(np.float64)

        print(f"\n── Position group: {name} (n={n}) ──")
        group_result = ot_vs_random_for_group(x_group, SINKHORN_REG)
        results_per_group[name] = group_result
        print(f"  Random cost: {group_result['random_cost']:.4f}")
        print(f"  OT cost:     {group_result['ot_cost']:.4f}")
        print(f"  Reduction:   {group_result['reduction_pct']:.2f}%")

    # Compute token-level average reduction (excluding 'all_positions')
    token_reductions = [
        r["reduction_pct"] for name, r in results_per_group.items()
        if name != "all_positions"
    ]
    if token_reductions:
        avg_token_reduction = np.mean(token_reductions)
    else:
        avg_token_reduction = results_per_group.get("all_positions", {}).get("reduction_pct", 0)

    # Compare to sequence level
    if seq_level_reduction > 0:
        token_to_seq_ratio = avg_token_reduction / seq_level_reduction
    else:
        token_to_seq_ratio = float("inf") if avg_token_reduction > 0 else 1.0

    # Decision
    if token_to_seq_ratio >= 0.5:
        decision = "GO"
        msg = (f"Token-level reduction ({avg_token_reduction:.1f}%) is "
               f"{token_to_seq_ratio:.1%} of sequence-level ({seq_level_reduction:.1f}%). "
               f"Coupling gain holds at per-token granularity.")
    else:
        decision = "NO-GO"
        msg = (f"Token-level reduction ({avg_token_reduction:.1f}%) is only "
               f"{token_to_seq_ratio:.1%} of sequence-level ({seq_level_reduction:.1f}%). "
               f"Coupling gain evaporates per-token — needs coarser granularity.")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results_per_group.keys())
    reductions = [results_per_group[n]["reduction_pct"] for n in names]
    colors = ["#4C72B0" if n != "all_positions" else "#55A868" for n in names]
    bars = ax.bar(names, reductions, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(seq_level_reduction, color="red", ls="--", lw=2,
               label=f"Seq-level (Test 2): {seq_level_reduction:.1f}%")
    ax.axhline(seq_level_reduction * 0.5, color="orange", ls=":", lw=1.5,
               label=f"50% threshold: {seq_level_reduction * 0.5:.1f}%")
    ax.set_ylabel("OT Reduction (%)")
    ax.set_title("Token-Level vs. Sequence-Level OT Improvement", fontsize=12)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_04_token_vs_seq.png", dpi=150)
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'test_04_token_vs_seq.png'}")

    # Save results
    metrics = {
        "seq_level_reduction_pct": seq_level_reduction,
        "avg_token_reduction_pct": float(avg_token_reduction),
        "token_to_seq_ratio": float(token_to_seq_ratio),
        "per_group_results": {k: v for k, v in results_per_group.items()},
        "decision": decision,
    }
    with open(RESULTS_DIR / "test_04.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if decision == "GO":
        print(f"✅ GO: {msg}")
    else:
        print(f"❌ NO-GO: {msg}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
