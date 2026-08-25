#!/usr/bin/env python3
"""Test 5 — ASBM Coupling-Estimation Convergence and Compute Scaling.

Implements ASBM-style coupling estimation in isolation on synthetic Gaussian-mixture
data at increasing dimensionality and sample size. Measures wall-clock time and
memory to extrapolate feasibility at ELF's actual per-batch scale.

Go/No-Go:
  Go:    Extrapolated cost at ELF scale < 20% of a flow-network forward pass.
  No-Go: Cost scaling is super-linear and would dominate training time.
"""

import json
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Test grid
DIMS = [16, 32, 64, 128]
SAMPLE_SIZES = [256, 512, 1024, 2048, 4096]
SINKHORN_REG = 0.05
N_REPEATS = 3

# ELF's target scale (Appendix D.2)
ELF_BATCH_SIZE = 512
ELF_SEQ_LEN = 128  # typical, up to 1024
ELF_DIM = 128


def generate_gmm_data(n: int, d: int, n_components: int = 5, rng=None):
    """Generate Gaussian mixture model data."""
    if rng is None:
        rng = np.random.default_rng(42)
    means = rng.standard_normal((n_components, d)) * 3
    assignments = rng.integers(0, n_components, size=n)
    x = rng.standard_normal((n, d)) * 0.5  # Within-cluster variance
    for i in range(n):
        x[i] += means[assignments[i]]
    return x.astype(np.float64)


def run_coupling_estimation(x: np.ndarray, eps: np.ndarray, reg: float):
    """Run Sinkhorn coupling estimation and measure cost."""
    import ot

    t0 = time.time()
    C = ot.dist(x, eps, metric="sqeuclidean")
    C_norm = C / (C.max() + 1e-10)
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(eps.shape[0]) / eps.shape[0]
    T = ot.sinkhorn(a, b, C_norm, reg=reg, numItermax=2000, stopThr=1e-9)
    elapsed = time.time() - t0

    # Peak memory estimate (cost matrix dominates)
    mem_bytes = C.nbytes + T.nbytes + x.nbytes + eps.nbytes
    return elapsed, mem_bytes, T


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 5 — ASBM Coupling-Estimation Compute Scaling")
    print("=" * 60)

    rng = np.random.default_rng(42)
    results = []

    for d, n in product(DIMS, SAMPLE_SIZES):
        times = []
        mems = []
        for rep in range(N_REPEATS):
            x = generate_gmm_data(n, d, rng=np.random.default_rng(42 + rep))
            eps = rng.standard_normal((n, d))
            elapsed, mem, _ = run_coupling_estimation(x, eps, SINKHORN_REG)
            times.append(elapsed)
            mems.append(mem)

        avg_time = np.mean(times)
        avg_mem = np.mean(mems)
        result = {
            "dim": d, "n": n,
            "avg_time_s": float(avg_time),
            "std_time_s": float(np.std(times)),
            "avg_mem_mb": float(avg_mem / 1e6),
        }
        results.append(result)
        print(f"  d={d:4d}, n={n:5d}: {avg_time:.3f}s ± {np.std(times):.3f}s, "
              f"mem={avg_mem / 1e6:.1f}MB")

    # Fit scaling curve: time ≈ α · n^β · d^γ
    from scipy.optimize import curve_fit

    ns = np.array([r["n"] for r in results])
    ds = np.array([r["dim"] for r in results])
    ts = np.array([r["avg_time_s"] for r in results])

    def scaling_model(X, alpha, beta, gamma):
        n, d = X
        return alpha * n**beta * d**gamma

    try:
        popt, pcov = curve_fit(
            scaling_model, (ns, ds), ts,
            p0=[1e-6, 2.0, 1.0],
            maxfev=10000,
        )
        alpha, beta, gamma = popt
        print(f"\n── Scaling fit: time ≈ {alpha:.2e} · n^{beta:.2f} · d^{gamma:.2f} ──")

        # Extrapolate to ELF's scale
        # Per-batch coupling: n = ELF_BATCH_SIZE * ELF_SEQ_LEN tokens, d = 128
        # But that's 512*128 = 65536 — impractical for full Sinkhorn.
        # More realistic: coupling per-sequence (n=SEQ_LEN, d=128) × BATCH_SIZE
        per_seq_time = scaling_model(
            (ELF_SEQ_LEN, ELF_DIM), alpha, beta, gamma
        )
        total_per_batch = per_seq_time * ELF_BATCH_SIZE
        print(f"\n── Extrapolation to ELF scale ──")
        print(f"  Per-sequence coupling (n={ELF_SEQ_LEN}, d={ELF_DIM}): {per_seq_time:.4f}s")
        print(f"  Total per batch ({ELF_BATCH_SIZE} sequences): {total_per_batch:.2f}s")

        # Also extrapolate global (all tokens in batch) cost
        global_n = min(ELF_BATCH_SIZE * ELF_SEQ_LEN, 8192)  # cap for sanity
        global_time = scaling_model((global_n, ELF_DIM), alpha, beta, gamma)
        print(f"  Global coupling (n={global_n}, d={ELF_DIM}): {global_time:.2f}s")
    except Exception as e:
        print(f"  Scaling fit failed: {e}")
        alpha = beta = gamma = None
        total_per_batch = None
        global_time = None

    # Rough flow-network forward pass estimate (a T5-small-scale model)
    # Typical: ~0.1-0.5s per batch on CPU for a 12-layer transformer
    ESTIMATED_FWD_TIME = 0.3  # seconds, conservative CPU estimate

    if total_per_batch is not None:
        overhead_ratio = total_per_batch / ESTIMATED_FWD_TIME
        if overhead_ratio < 0.2:
            decision = "GO"
            msg = (f"Coupling overhead ({total_per_batch:.2f}s) is {overhead_ratio:.1%} of "
                   f"fwd pass (~{ESTIMATED_FWD_TIME}s). Under 20% threshold.")
        else:
            decision = "NO-GO"
            msg = (f"Coupling overhead ({total_per_batch:.2f}s) is {overhead_ratio:.1%} of "
                   f"fwd pass (~{ESTIMATED_FWD_TIME}s). Exceeds 20% threshold — needs "
                   f"approximation or coarser granularity.")
    else:
        decision = "INCONCLUSIVE"
        msg = "Could not fit scaling curve."

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time vs sample size (coloured by dim)
    ax = axes[0]
    for d in DIMS:
        subset = [r for r in results if r["dim"] == d]
        ns_plot = [r["n"] for r in subset]
        ts_plot = [r["avg_time_s"] for r in subset]
        ax.loglog(ns_plot, ts_plot, "o-", label=f"d={d}")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Coupling Time vs. Sample Size", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time vs dimension (coloured by n)
    ax = axes[1]
    for n_val in SAMPLE_SIZES:
        subset = [r for r in results if r["n"] == n_val]
        ds_plot = [r["dim"] for r in subset]
        ts_plot = [r["avg_time_s"] for r in subset]
        ax.loglog(ds_plot, ts_plot, "s-", label=f"n={n_val}")
    ax.set_xlabel("Dimension (d)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Coupling Time vs. Dimension", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_05_scaling.png", dpi=150)
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'test_05_scaling.png'}")

    # Save results
    metrics = {
        "scaling_results": results,
        "scaling_fit": {
            "alpha": float(alpha) if alpha else None,
            "beta": float(beta) if beta else None,
            "gamma": float(gamma) if gamma else None,
        },
        "elf_extrapolation": {
            "per_seq_time_s": float(per_seq_time) if total_per_batch else None,
            "total_per_batch_s": float(total_per_batch) if total_per_batch else None,
            "global_time_s": float(global_time) if global_time else None,
            "estimated_fwd_time_s": ESTIMATED_FWD_TIME,
            "overhead_ratio": float(overhead_ratio) if total_per_batch else None,
        },
        "decision": decision,
    }
    with open(RESULTS_DIR / "test_05.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if decision == "GO":
        print(f"✅ GO: {msg}")
    elif decision == "NO-GO":
        print(f"❌ NO-GO: {msg}")
    else:
        print(f"⚠️  {decision}: {msg}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
