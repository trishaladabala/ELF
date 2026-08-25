#!/usr/bin/env python3
"""Test A3 — Sinkhorn Convergence Diagnostics.

For each formulation from A2, logs convergence metrics. Then sweeps ε for
the 'empirical' formulation to test stability of the 30.9% figure.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import ot as pot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N = 1500
SEED = 42
SINKHORN_MAX_ITER = 5000  # Increase to ensure convergence
SINKHORN_STOP_THR = 1e-9


def generate_noise(x, prior, rng):
    n, d = x.shape
    if prior == "gaussian":
        return rng.standard_normal((n, d))
    elif prior == "empirical":
        mean = x.mean(axis=0)
        std = np.sqrt(np.var(x, axis=0))
        return rng.standard_normal((n, d)) * std[None, :] + mean[None, :]
    elif prior == "uniform":
        eps = rng.standard_normal((n, d))
        norms = np.linalg.norm(eps, axis=1, keepdims=True)
        target_radius = np.linalg.norm(x, axis=1).mean()
        return eps / norms * target_radius


def run_sinkhorn_with_diagnostics(x, eps, reg):
    """Run Sinkhorn with full convergence diagnostics."""
    n = x.shape[0]
    C_raw = pot.dist(x, eps, metric="sqeuclidean")
    C_norm = C_raw / (C_raw.max() + 1e-10)
    a = np.ones(n) / n
    b = np.ones(n) / n

    t0 = time.time()
    T, log = pot.sinkhorn(
        a, b, C_norm, reg=reg,
        numItermax=SINKHORN_MAX_ITER, stopThr=SINKHORN_STOP_THR,
        log=True,
    )
    elapsed = time.time() - t0

    # Marginal violations
    row_marg = T.sum(axis=1)
    col_marg = T.sum(axis=0)
    row_violation = float(np.abs(row_marg - a).sum())
    col_violation = float(np.abs(col_marg - b).sum())
    max_row_dev = float(np.abs(row_marg - a).max())
    max_col_dev = float(np.abs(col_marg - b).max())

    # Did it converge?
    n_iters = log.get("niter", -1) if isinstance(log, dict) else -1
    hit_cap = n_iters >= SINKHORN_MAX_ITER

    # Metrics
    assignments = T.argmax(axis=1)
    hard_path_sq = float(np.mean(np.sum((x - eps[assignments]) ** 2, axis=1)))
    soft_ot_cost = float(np.sum(T * C_raw))

    rng_s = np.random.default_rng(SEED + 2000)
    random_costs = []
    for _ in range(50):
        perm = rng_s.permutation(n)
        random_costs.append(np.mean(np.sum((x - eps[perm]) ** 2, axis=1)))
    random_cost = float(np.mean(random_costs))

    soft_reduction = (1.0 - soft_ot_cost / random_cost) * 100
    hard_reduction = (1.0 - hard_path_sq / random_cost) * 100

    return {
        "reg": float(reg),
        "n_iters": n_iters,
        "hit_iter_cap": bool(hit_cap),
        "row_violation_L1": row_violation,
        "col_violation_L1": col_violation,
        "max_row_deviation": max_row_dev,
        "max_col_deviation": max_col_dev,
        "compute_time_s": elapsed,
        "soft_reduction_pct": soft_reduction,
        "hard_reduction_pct": hard_reduction,
        "random_cost": random_cost,
        "soft_ot_cost": soft_ot_cost,
        "hard_path_sq": hard_path_sq,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST A3 — Sinkhorn Convergence Diagnostics")
    print("=" * 60)

    x_128 = np.load(RESULTS_DIR / "x_128.npy")
    rng_master = np.random.default_rng(SEED)
    idx = rng_master.choice(x_128.shape[0], N, replace=False)
    x = x_128[idx].astype(np.float64)

    # ── Part 1: Convergence diagnostics for all 3 priors at ε=0.05 ──
    print("\n── Part 1: Convergence at ε=0.05 for all priors ──")
    priors = ["gaussian", "empirical", "uniform"]
    prior_results = {}

    for prior in priors:
        rng_noise = np.random.default_rng(SEED + hash(prior) % 10000)
        eps = generate_noise(x, prior, rng_noise)
        result = run_sinkhorn_with_diagnostics(x, eps, 0.05)
        prior_results[prior] = result

        converged = result["row_violation_L1"] < 1e-3 and not result["hit_iter_cap"]
        print(f"\n  {prior}:")
        print(f"    Iterations:      {result['n_iters']}")
        print(f"    Hit iter cap:    {result['hit_iter_cap']}")
        print(f"    Row violation:   {result['row_violation_L1']:.2e}")
        print(f"    Col violation:   {result['col_violation_L1']:.2e}")
        print(f"    Max row dev:     {result['max_row_deviation']:.2e}")
        print(f"    Soft reduction:  {result['soft_reduction_pct']:.2f}%")
        print(f"    Hard reduction:  {result['hard_reduction_pct']:.2f}%")
        print(f"    Converged:       {'✅' if converged else '❌'}")

    # ── Part 2: ε sweep for 'empirical' prior ──
    print("\n── Part 2: ε sweep for 'empirical' prior ──")
    eps_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    eps_sweep_results = {}

    rng_emp = np.random.default_rng(SEED + hash("empirical") % 10000)
    eps_empirical = generate_noise(x, "empirical", rng_emp)

    for reg in eps_values:
        result = run_sinkhorn_with_diagnostics(x, eps_empirical, reg)
        eps_sweep_results[str(reg)] = result

        stable = result["row_violation_L1"] < 1e-3
        print(f"  ε={reg:<5}: soft={result['soft_reduction_pct']:>7.2f}%, "
              f"hard={result['hard_reduction_pct']:>7.2f}%, "
              f"iters={result['n_iters']}, "
              f"violation={result['row_violation_L1']:.2e} "
              f"{'✅' if stable else '⚠️'}")

    # Stability check: does the hard reduction swing by >10pp?
    hard_reductions = [r["hard_reduction_pct"] for r in eps_sweep_results.values()]
    soft_reductions = [r["soft_reduction_pct"] for r in eps_sweep_results.values()]
    hard_swing = max(hard_reductions) - min(hard_reductions)
    soft_swing = max(soft_reductions) - min(soft_reductions)

    print(f"\n  Hard reduction range: {min(hard_reductions):.1f}% – {max(hard_reductions):.1f}% "
          f"(swing={hard_swing:.1f}pp)")
    print(f"  Soft reduction range: {min(soft_reductions):.1f}% – {max(soft_reductions):.1f}% "
          f"(swing={soft_swing:.1f}pp)")

    # ── Part 3: ε sweep for 'gaussian' prior (the relevant one) ──
    print("\n── Part 3: ε sweep for 'gaussian' prior ──")
    gauss_sweep_results = {}
    rng_gauss = np.random.default_rng(SEED + hash("gaussian") % 10000)
    eps_gaussian = generate_noise(x, "gaussian", rng_gauss)

    for reg in eps_values:
        result = run_sinkhorn_with_diagnostics(x, eps_gaussian, reg)
        gauss_sweep_results[str(reg)] = result
        print(f"  ε={reg:<5}: soft={result['soft_reduction_pct']:>7.2f}%, "
              f"hard={result['hard_reduction_pct']:>7.2f}%, "
              f"iters={result['n_iters']}, "
              f"violation={result['row_violation_L1']:.2e}")

    gauss_soft = [r["soft_reduction_pct"] for r in gauss_sweep_results.values()]
    gauss_hard = [r["hard_reduction_pct"] for r in gauss_sweep_results.values()]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogx(eps_values, [eps_sweep_results[str(e)]["hard_reduction_pct"] for e in eps_values],
                "o-", color="#DD8452", lw=2, label="Empirical (hard)")
    ax.semilogx(eps_values, [eps_sweep_results[str(e)]["soft_reduction_pct"] for e in eps_values],
                "s--", color="#DD8452", lw=1.5, alpha=0.6, label="Empirical (soft)")
    ax.semilogx(eps_values, [gauss_sweep_results[str(e)]["hard_reduction_pct"] for e in eps_values],
                "o-", color="#4C72B0", lw=2, label="Gaussian (hard)")
    ax.semilogx(eps_values, [gauss_sweep_results[str(e)]["soft_reduction_pct"] for e in eps_values],
                "s--", color="#4C72B0", lw=1.5, alpha=0.6, label="Gaussian (soft)")
    ax.axvline(0.05, color="gray", ls=":", alpha=0.5, label="ε=0.05 (original)")
    ax.set_xlabel("Sinkhorn ε (regularization)")
    ax.set_ylabel("OT Reduction (%)")
    ax.set_title("Reduction vs. Sinkhorn ε", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    violations_emp = [eps_sweep_results[str(e)]["row_violation_L1"] for e in eps_values]
    violations_gauss = [gauss_sweep_results[str(e)]["row_violation_L1"] for e in eps_values]
    ax.loglog(eps_values, violations_emp, "o-", color="#DD8452", lw=2, label="Empirical")
    ax.loglog(eps_values, violations_gauss, "o-", color="#4C72B0", lw=2, label="Gaussian")
    ax.axhline(1e-3, color="red", ls="--", alpha=0.5, label="Convergence threshold")
    ax.set_xlabel("Sinkhorn ε")
    ax.set_ylabel("Row marginal violation (L1)")
    ax.set_title("Convergence Quality vs. ε", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testA3_convergence.png", dpi=150)
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'testA3_convergence.png'}")

    # Verdict
    all_converged = all(r["row_violation_L1"] < 1e-3 for r in prior_results.values())
    empirical_stable = hard_swing < 10
    gaussian_stable = (max(gauss_soft) - min(gauss_soft)) < 5

    print("\n" + "=" * 60)
    print("CONVERGENCE VERDICT")
    print("=" * 60)
    print(f"  All priors converged at ε=0.05:    {'✅' if all_converged else '❌'}")
    print(f"  Empirical stable across ε (<10pp): {'✅' if empirical_stable else '❌ (swing={hard_swing:.1f}pp)'}")
    print(f"  Gaussian stable across ε (<5pp):   {'✅' if gaussian_stable else '❌'}")

    # Save
    output = {
        "convergence_at_eps_005": prior_results,
        "eps_sweep_empirical": eps_sweep_results,
        "eps_sweep_gaussian": gauss_sweep_results,
        "empirical_hard_swing_pp": hard_swing,
        "empirical_soft_swing_pp": soft_swing,
        "gaussian_soft_swing_pp": max(gauss_soft) - min(gauss_soft),
        "all_converged": all_converged,
        "empirical_stable": empirical_stable,
        "gaussian_stable": gaussian_stable,
    }
    with open(RESULTS_DIR / "testA3_convergence.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {RESULTS_DIR / 'testA3_convergence.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
