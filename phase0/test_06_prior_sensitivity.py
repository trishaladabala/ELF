#!/usr/bin/env python3
"""Test 6 — Prior Energy Function Sensitivity.

Tests whether the choice of prior energy function materially changes the resulting
coupling, using 2-3 plausible prior energy choices on synthetic/real embeddings.

Go/No-Go:
  Go:    Results qualitatively similar across prior choices (low sensitivity).
  No-Go: Results vary substantially — prior selection is a first-class research question.
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
N_SAMPLE = 1000
DIM = 128
SINKHORN_REG = 0.05


def generate_gmm_data(n, d, n_components=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    means = rng.standard_normal((n_components, d)) * 3
    assignments = rng.integers(0, n_components, size=n)
    x = rng.standard_normal((n, d)) * 0.5
    for i in range(n):
        x[i] += means[assignments[i]]
    return x.astype(np.float64)


def compute_coupling_with_prior(
    x: np.ndarray, prior_name: str, reg: float, rng
) -> dict:
    """Compute OT coupling under a specific prior energy / noise distribution.

    Prior energies tested:
    1. 'gaussian':  ε ~ N(0, I) — standard isotropic Gaussian (ELF default)
    2. 'empirical': ε drawn from a fitted Gaussian matching the data's empirical
                    distribution (same mean, same covariance)
    3. 'uniform':   ε ~ Uniform on a hypersphere of radius matching data's mean norm
    """
    import ot

    n, d = x.shape

    if prior_name == "gaussian":
        eps = rng.standard_normal((n, d))
    elif prior_name == "empirical":
        # Match data's first two moments
        mean = x.mean(axis=0)
        cov = np.cov(x.T)
        # Use Cholesky for sampling (diagonal approx for speed in high-d)
        std = np.sqrt(np.diag(cov))
        eps = rng.standard_normal((n, d)) * std[None, :] + mean[None, :]
    elif prior_name == "uniform":
        # Uniform on hypersphere
        eps = rng.standard_normal((n, d))
        norms = np.linalg.norm(eps, axis=1, keepdims=True)
        target_radius = np.linalg.norm(x, axis=1).mean()
        eps = eps / norms * target_radius
    else:
        raise ValueError(f"Unknown prior: {prior_name}")

    # Compute OT coupling
    C = ot.dist(x, eps, metric="sqeuclidean")
    C_norm = C / (C.max() + 1e-10)
    a = np.ones(n) / n
    b = np.ones(n) / n

    t0 = time.time()
    T = ot.sinkhorn(a, b, C_norm, reg=reg, numItermax=2000, stopThr=1e-9)
    elapsed = time.time() - t0

    # OT cost
    ot_cost = float(np.sum(T * C))

    # Average path length (using OT assignment)
    assignments = T.argmax(axis=1)
    avg_path_sq = float(np.mean(np.sum((x - eps[assignments]) ** 2, axis=1)))

    # Random pairing cost for reference
    random_costs = []
    for _ in range(30):
        perm = rng.permutation(n)
        random_costs.append(np.mean(np.sum((x - eps[perm]) ** 2, axis=1)))
    random_cost = float(np.mean(random_costs))

    reduction = (1 - avg_path_sq / random_cost) * 100 if random_cost > 0 else 0

    return {
        "prior": prior_name,
        "ot_cost": ot_cost,
        "avg_path_sq": avg_path_sq,
        "random_cost": random_cost,
        "reduction_pct": reduction,
        "compute_time_s": elapsed,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 6 — Prior Energy Function Sensitivity")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Try both synthetic and real data if available
    datasets = {}

    # Synthetic GMM
    x_synth = generate_gmm_data(N_SAMPLE, DIM, rng=rng)
    datasets["synthetic_gmm"] = x_synth

    # Real embeddings (if Test 1 ran)
    x128_path = RESULTS_DIR / "x_128.npy"
    if x128_path.exists():
        x_real = np.load(x128_path)
        idx = rng.choice(x_real.shape[0], min(N_SAMPLE, x_real.shape[0]), replace=False)
        datasets["real_t5_embeddings"] = x_real[idx].astype(np.float64)

    priors = ["gaussian", "empirical", "uniform"]
    all_results = {}

    for ds_name, x in datasets.items():
        print(f"\n── Dataset: {ds_name} (n={x.shape[0]}, d={x.shape[1]}) ──")
        ds_results = {}

        for prior in priors:
            print(f"  Prior: {prior} …", end=" ", flush=True)
            result = compute_coupling_with_prior(x, prior, SINKHORN_REG, rng)
            ds_results[prior] = result
            print(f"reduction={result['reduction_pct']:.2f}%, "
                  f"path²={result['avg_path_sq']:.2f}, "
                  f"time={result['compute_time_s']:.2f}s")

        all_results[ds_name] = ds_results

    # Analyse sensitivity: max variation in reduction across priors
    sensitivity_results = {}
    for ds_name, ds_results in all_results.items():
        reductions = [r["reduction_pct"] for r in ds_results.values()]
        path_sqs = [r["avg_path_sq"] for r in ds_results.values()]
        max_variation = max(reductions) - min(reductions)
        path_variation = (max(path_sqs) - min(path_sqs)) / np.mean(path_sqs) * 100

        sensitivity_results[ds_name] = {
            "reductions": {k: v["reduction_pct"] for k, v in ds_results.items()},
            "path_sqs": {k: v["avg_path_sq"] for k, v in ds_results.items()},
            "max_reduction_variation_ppt": max_variation,
            "path_sq_relative_variation_pct": path_variation,
        }
        print(f"\n  {ds_name}: max reduction variation = {max_variation:.2f} pp, "
              f"path² relative variation = {path_variation:.1f}%")

    # Overall decision
    all_variations = [s["max_reduction_variation_ppt"] for s in sensitivity_results.values()]
    max_overall_var = max(all_variations)

    if max_overall_var < 5:
        decision = "GO"
        msg = (f"Low sensitivity (max variation {max_overall_var:.1f} pp). "
               f"Pick a defensible default and move on.")
    elif max_overall_var < 15:
        decision = "CAUTION"
        msg = (f"Moderate sensitivity ({max_overall_var:.1f} pp variation). "
               f"Document choice rationale carefully.")
    else:
        decision = "NO-GO"
        msg = (f"High sensitivity ({max_overall_var:.1f} pp variation). "
               f"Prior selection is a first-class research question.")

    # Plot
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (ds_name, ds_results) in zip(axes, all_results.items()):
        names = list(ds_results.keys())
        reductions = [ds_results[n]["reduction_pct"] for n in names]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars = ax.bar(names, reductions, color=colors[:len(names)],
                      edgecolor="black", alpha=0.8)
        ax.set_ylabel("OT Reduction (%)")
        ax.set_title(f"Prior Sensitivity — {ds_name}", fontsize=11)
        for bar, val in zip(bars, reductions):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_06_prior_sensitivity.png", dpi=150)
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'test_06_prior_sensitivity.png'}")

    # Save
    metrics = {
        "n_sample": N_SAMPLE,
        "dim": DIM,
        "priors_tested": priors,
        "results": {ds: {p: r for p, r in dr.items()} for ds, dr in all_results.items()},
        "sensitivity": sensitivity_results,
        "max_variation_pp": float(max_overall_var),
        "decision": decision,
    }
    with open(RESULTS_DIR / "test_06.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if decision == "GO":
        print(f"✅ GO: {msg}")
    elif decision == "CAUTION":
        print(f"⚠️  CAUTION: {msg}")
    else:
        print(f"❌ NO-GO: {msg}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
