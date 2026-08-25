#!/usr/bin/env python3
"""Test 3 — Local Curvature / Manifold Linearity Estimate.

Measures local curvature of the embedding manifold (residual variance from
local PCA) and estimates global intrinsic dimensionality.

Go/No-Go:
  Supports nonlinear-path: high local residual variance AND intrinsic dim < 60% of 128.
  Does not support: low local residual variance AND intrinsic dim ≈ 128.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_SAMPLE = 500          # Points for local curvature analysis
K_NEIGHBORS = 10        # k-NN for local PCA
N_LOCAL_PCS = 5          # Top local PCs to keep (local manifold dimension estimate)
BOTTLENECK_DIM = 128     # Ambient dimension


def load_embeddings():
    x_128 = np.load(RESULTS_DIR / "x_128.npy")
    print(f"Loaded x_128: {x_128.shape}")
    return x_128


def local_curvature_analysis(x: np.ndarray, k: int, n_local_pcs: int) -> dict:
    """Compute local PCA residual variance for a sample of points.

    For each point, fit PCA in its k-NN neighbourhood and measure the fraction
    of variance NOT captured by the top n_local_pcs components.
    """
    print(f"Building KD-tree for {x.shape[0]} points …")
    tree = KDTree(x)

    residual_fractions = []
    total_variances = []

    for i in range(x.shape[0]):
        dists, idx = tree.query(x[i], k=k + 1)  # +1 because the point itself is included
        neighbors = x[idx]  # (k+1, d)

        # Center
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)

        # SVD
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        total_var = (S ** 2).sum()
        explained_var = (S[:n_local_pcs] ** 2).sum()

        if total_var > 0:
            residual_frac = 1.0 - explained_var / total_var
        else:
            residual_frac = 0.0

        residual_fractions.append(residual_frac)
        total_variances.append(total_var)

    return {
        "residual_fractions": np.array(residual_fractions),
        "total_variances": np.array(total_variances),
    }


def estimate_intrinsic_dim(x: np.ndarray) -> dict:
    """Estimate intrinsic dimensionality using multiple methods."""
    results = {}

    # Method 1: MLE (Levina & Bickel)
    try:
        from skdim.id import MLE
        mle = MLE()
        mle.fit(x)
        results["mle_id"] = float(mle.dimension_)
        print(f"  MLE intrinsic dim: {results['mle_id']:.2f}")
    except Exception as e:
        print(f"  MLE failed: {e}")
        results["mle_id"] = None

    # Method 2: TwoNN (Facco et al.)
    try:
        from skdim.id import TwoNN
        twonn = TwoNN()
        twonn.fit(x)
        results["twonn_id"] = float(twonn.dimension_)
        print(f"  TwoNN intrinsic dim: {results['twonn_id']:.2f}")
    except Exception as e:
        print(f"  TwoNN failed: {e}")
        results["twonn_id"] = None

    # Method 3: PCA-based (fraction of variance at knee)
    _, S, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    results["pca_dim_90pct"] = dim_90
    results["pca_dim_95pct"] = dim_95
    results["pca_dim_99pct"] = dim_99
    results["singular_values"] = S.tolist()[:50]  # Save top 50 for plotting
    results["cumulative_variance"] = cumvar.tolist()[:128]
    print(f"  PCA dims for 90/95/99% variance: {dim_90}/{dim_95}/{dim_99}")

    return results


def make_plots(curvature: dict, id_results: dict):
    """Save diagnostic plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Local residual variance distribution
    ax = axes[0]
    res = curvature["residual_fractions"]
    ax.hist(res, bins=50, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(res.mean(), color="red", linestyle="--", label=f"mean={res.mean():.3f}")
    ax.set_title(f"Local Residual Variance\n(k={K_NEIGHBORS}, top-{N_LOCAL_PCS} PCs)", fontsize=11)
    ax.set_xlabel("Residual fraction (unexplained)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # 2. Singular value spectrum
    ax = axes[1]
    svs = id_results.get("singular_values", [])
    if svs:
        ax.semilogy(range(1, len(svs) + 1), svs, "o-", markersize=3, color="#55A868")
        ax.set_title("Singular Value Spectrum (global)", fontsize=11)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Singular value (log)")

    # 3. Cumulative variance
    ax = axes[2]
    cumvar = id_results.get("cumulative_variance", [])
    if cumvar:
        ax.plot(range(1, len(cumvar) + 1), cumvar, "-", color="#C44E52", lw=2)
        ax.axhline(0.90, color="gray", ls="--", alpha=0.5, label="90%")
        ax.axhline(0.95, color="gray", ls=":", alpha=0.5, label="95%")
        dim90 = id_results.get("pca_dim_90pct", "?")
        dim95 = id_results.get("pca_dim_95pct", "?")
        ax.axvline(dim90, color="blue", ls="--", alpha=0.3)
        ax.axvline(dim95, color="blue", ls=":", alpha=0.3)
        ax.set_title(f"Cumulative Variance Explained\n(90%@{dim90}d, 95%@{dim95}d)", fontsize=11)
        ax.set_xlabel("Number of PCs")
        ax.set_ylabel("Cumulative variance fraction")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_03_curvature.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {RESULTS_DIR / 'test_03_curvature.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 3 — Local Curvature / Manifold Linearity Estimate")
    print("=" * 60)

    x_128 = load_embeddings()

    # Subsample for local curvature
    rng = np.random.default_rng(42)
    n = min(N_SAMPLE, x_128.shape[0])
    idx = rng.choice(x_128.shape[0], n, replace=False)
    x_sub = x_128[idx].astype(np.float64)

    # Local curvature analysis
    print(f"\n── Local Curvature (k={K_NEIGHBORS}, top-{N_LOCAL_PCS} PCs) ──")
    curvature = local_curvature_analysis(x_sub, K_NEIGHBORS, N_LOCAL_PCS)
    residuals = curvature["residual_fractions"]
    print(f"  Mean residual fraction:   {residuals.mean():.4f}")
    print(f"  Median residual fraction: {np.median(residuals):.4f}")
    print(f"  Std residual fraction:    {residuals.std():.4f}")

    # Intrinsic dimensionality
    print(f"\n── Intrinsic Dimensionality Estimates ──")
    # Use a larger sample for global ID estimation
    n_id = min(5000, x_128.shape[0])
    idx_id = rng.choice(x_128.shape[0], n_id, replace=False)
    x_id = x_128[idx_id].astype(np.float64)
    id_results = estimate_intrinsic_dim(x_id)

    # Determine best ID estimate
    id_estimates = []
    for key in ["mle_id", "twonn_id"]:
        v = id_results.get(key)
        if v is not None:
            id_estimates.append(v)
    best_id = np.mean(id_estimates) if id_estimates else id_results.get("pca_dim_95pct", None)

    # Decision logic
    high_curvature = residuals.mean() > 0.15  # Significant unexplained variance
    low_dim = best_id is not None and best_id < 0.6 * BOTTLENECK_DIM  # < 60% of 128 = 76.8

    if high_curvature and low_dim:
        decision = "SUPPORTS_NONLINEAR"
        msg = (f"High local curvature (mean residual={residuals.mean():.3f}) "
               f"AND low intrinsic dim ({best_id:.1f} << {BOTTLENECK_DIM}). "
               f"Nonlinear path is justified.")
    elif not high_curvature and not low_dim:
        decision = "DOES_NOT_SUPPORT"
        msg = (f"Low curvature (mean residual={residuals.mean():.3f}) "
               f"AND intrinsic dim ≈ ambient ({best_id:.1f} / {BOTTLENECK_DIM}). "
               f"Corroborates No-Go from Test 2.")
    else:
        decision = "MIXED"
        msg = (f"Mixed signal: curvature={'high' if high_curvature else 'low'} "
               f"(mean residual={residuals.mean():.3f}), "
               f"intrinsic dim={best_id:.1f}/{BOTTLENECK_DIM}.")

    # Plots
    print("\nGenerating plots …")
    make_plots(curvature, id_results)

    # Save results
    metrics = {
        "n_sample": n,
        "k_neighbors": K_NEIGHBORS,
        "n_local_pcs": N_LOCAL_PCS,
        "local_residual_mean": float(residuals.mean()),
        "local_residual_median": float(np.median(residuals)),
        "local_residual_std": float(residuals.std()),
        "high_curvature": bool(high_curvature),
        "mle_id": id_results.get("mle_id"),
        "twonn_id": id_results.get("twonn_id"),
        "pca_dim_90pct": id_results.get("pca_dim_90pct"),
        "pca_dim_95pct": id_results.get("pca_dim_95pct"),
        "pca_dim_99pct": id_results.get("pca_dim_99pct"),
        "best_id_estimate": float(best_id) if best_id is not None else None,
        "low_intrinsic_dim": bool(low_dim),
        "bottleneck_dim": BOTTLENECK_DIM,
        "decision": decision,
    }
    with open(RESULTS_DIR / "test_03.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if decision == "SUPPORTS_NONLINEAR":
        print(f"✅ SUPPORTS: {msg}")
    elif decision == "DOES_NOT_SUPPORT":
        print(f"❌ DOES NOT SUPPORT: {msg}")
    else:
        print(f"⚠️  MIXED: {msg}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
