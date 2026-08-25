#!/usr/bin/env python3
"""Test 2 — Straight-Line vs. Approximate Optimal-Transport Coupling.

**This is the single most decision-relevant test in the whole plan.**

Compares random (independent/memoryless) pairing vs. entropic-OT-optimal pairing
(Sinkhorn) between clean embeddings x and Gaussian noise ε. Reports the ratio of
average squared path lengths and visualizes matched pairs.

Go/No-Go:
  Go (≥15% reduction): OT coupling is meaningfully shorter — proceed to Phase 0B.
  No-Go (<5-8%): Random pairing is near-optimal — shelve interpolation project.
  Ambiguous (5-15%): Proceed with caution, check Tests 3 & 4.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_SUBSAMPLE = 1500   # Number of points for OT computation
SINKHORN_REG = 0.05  # Entropic regularisation (Sinkhorn ε)


def load_embeddings():
    """Load 128-d embeddings from Test 1."""
    x_128 = np.load(RESULTS_DIR / "x_128.npy")
    print(f"Loaded x_128: {x_128.shape}")
    return x_128


def random_pairing_cost(x: np.ndarray, eps: np.ndarray) -> float:
    """Average squared path length under independent/random pairing."""
    # Under random pairing, E[||x_i - ε_j||²] = E[||x||²] + E[||ε||²] - 2·E[x]·E[ε]
    # But since we have matched sets of same size, we compute per-pair and average.
    costs = np.sum((x - eps) ** 2, axis=1)
    return float(costs.mean())


def ot_optimal_pairing_cost(x: np.ndarray, eps: np.ndarray, reg: float) -> tuple[float, np.ndarray]:
    """Average squared path length under entropic-OT-optimal (Sinkhorn) pairing.

    Returns (avg_cost, transport_plan).
    """
    import ot

    n = x.shape[0]
    # Cost matrix: squared Euclidean distances
    print(f"  Computing {n}×{n} cost matrix …")
    t0 = time.time()
    C = ot.dist(x, eps, metric="sqeuclidean")
    C = C / C.max()  # Normalise for numerical stability

    # Uniform marginals
    a = np.ones(n) / n
    b = np.ones(n) / n

    print(f"  Running Sinkhorn (reg={reg}) …")
    T = ot.sinkhorn(a, b, C, reg=reg, numItermax=2000, stopThr=1e-9)
    elapsed = time.time() - t0
    print(f"  Sinkhorn converged in {elapsed:.1f}s")

    # Compute OT cost in original scale (un-normalised)
    C_raw = np.sum((x[:, None, :] - eps[None, :, :]) ** 2, axis=2)
    ot_cost = float(np.sum(T * C_raw))

    return ot_cost, T


def shuffle_pairing_cost(x: np.ndarray, eps: np.ndarray, n_shuffles: int = 50) -> tuple[float, float]:
    """Monte Carlo estimate of random-pairing cost (mean ± std over shuffles)."""
    rng = np.random.default_rng(42)
    costs = []
    for _ in range(n_shuffles):
        perm = rng.permutation(x.shape[0])
        c = np.sum((x - eps[perm]) ** 2, axis=1).mean()
        costs.append(c)
    return float(np.mean(costs)), float(np.std(costs))


def visualise_pairings(
    x: np.ndarray, eps: np.ndarray, T: np.ndarray, n_show: int = 30
):
    """Visualise random vs OT pairings in 2D PCA projection."""
    from sklearn.decomposition import PCA

    all_pts = np.vstack([x, eps])
    pca = PCA(n_components=2, random_state=42).fit(all_pts)
    x_2d = pca.transform(x)
    eps_2d = pca.transform(eps)

    rng = np.random.default_rng(0)
    show_idx = rng.choice(x.shape[0], min(n_show, x.shape[0]), replace=False)

    # OT assignment: for each x_i, find the j with highest T[i, j]
    ot_assignments = T.argmax(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Random pairing
    ax = axes[0]
    ax.scatter(x_2d[:, 0], x_2d[:, 1], c="#4C72B0", s=6, alpha=0.3, label="data x")
    ax.scatter(eps_2d[:, 0], eps_2d[:, 1], c="#DD8452", s=6, alpha=0.3, label="noise ε")
    for i in show_idx:
        j = rng.integers(0, eps.shape[0])  # random pairing
        ax.plot(
            [x_2d[i, 0], eps_2d[j, 0]],
            [x_2d[i, 1], eps_2d[j, 1]],
            "k-", alpha=0.4, lw=0.7,
        )
    ax.set_title("Random Pairing (ELF default)", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # OT pairing
    ax = axes[1]
    ax.scatter(x_2d[:, 0], x_2d[:, 1], c="#4C72B0", s=6, alpha=0.3, label="data x")
    ax.scatter(eps_2d[:, 0], eps_2d[:, 1], c="#DD8452", s=6, alpha=0.3, label="noise ε")
    for i in show_idx:
        j = ot_assignments[i]
        ax.plot(
            [x_2d[i, 0], eps_2d[j, 0]],
            [x_2d[i, 1], eps_2d[j, 1]],
            "g-", alpha=0.5, lw=0.7,
        )
    ax.set_title("OT-Optimal Pairing (Sinkhorn)", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.suptitle("Random vs. OT Coupling — 2D PCA Projection", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_02_coupling_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {RESULTS_DIR / 'test_02_coupling_comparison.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 2 — OT vs. Random Coupling")
    print("=" * 60)

    # Load data
    x_128 = load_embeddings()

    # Subsample
    rng = np.random.default_rng(42)
    n = min(N_SUBSAMPLE, x_128.shape[0])
    idx = rng.choice(x_128.shape[0], n, replace=False)
    x = x_128[idx].astype(np.float64)

    # Generate matched Gaussian noise
    eps = rng.standard_normal(x.shape)

    print(f"\nSubsample size: {n}")
    print(f"Embedding dim:  {x.shape[1]}")

    # Random pairing cost (Monte Carlo)
    print("\n── Random pairing cost (Monte Carlo, 50 shuffles) ──")
    random_cost_mean, random_cost_std = shuffle_pairing_cost(x, eps)
    print(f"  Mean: {random_cost_mean:.4f}  Std: {random_cost_std:.4f}")

    # Identity pairing cost (x_i ↔ ε_i)
    identity_cost = random_pairing_cost(x, eps)
    print(f"  Identity pairing cost: {identity_cost:.4f}")

    # OT-optimal pairing cost
    print("\n── OT-optimal pairing cost (Sinkhorn) ──")
    ot_cost, T = ot_optimal_pairing_cost(x, eps, reg=SINKHORN_REG)
    print(f"  OT cost: {ot_cost:.4f}")

    # Ratio and reduction
    ratio = ot_cost / random_cost_mean
    reduction_pct = (1 - ratio) * 100

    print(f"\n── Key metric ──")
    print(f"  OT / Random ratio:    {ratio:.4f}")
    print(f"  Reduction:            {reduction_pct:.2f}%")

    # Decision
    if reduction_pct >= 15:
        decision = "GO"
        msg = (f"OT coupling achieves {reduction_pct:.1f}% reduction "
               f"(≥15% threshold). Non-memoryless coupling is justified.")
    elif reduction_pct >= 5:
        decision = "AMBIGUOUS"
        msg = (f"OT coupling achieves {reduction_pct:.1f}% reduction "
               f"(5-15% zone). Check Tests 3 & 4 before deciding.")
    else:
        decision = "NO-GO"
        msg = (f"OT coupling achieves only {reduction_pct:.1f}% reduction "
               f"(<5%). Random pairing is near-optimal; shelve project.")

    # Visualise
    print("\nGenerating visualisation …")
    visualise_pairings(x, eps, T)

    # Save results
    metrics = {
        "n_subsample": n,
        "embedding_dim": int(x.shape[1]),
        "sinkhorn_reg": SINKHORN_REG,
        "random_cost_mean": random_cost_mean,
        "random_cost_std": random_cost_std,
        "identity_cost": identity_cost,
        "ot_cost": ot_cost,
        "ot_random_ratio": ratio,
        "reduction_pct": reduction_pct,
        "decision": decision,
    }
    with open(RESULTS_DIR / "test_02.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if decision == "GO":
        print(f"✅ GO: {msg}")
    elif decision == "AMBIGUOUS":
        print(f"⚠️  AMBIGUOUS: {msg}")
    else:
        print(f"❌ NO-GO: {msg}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
