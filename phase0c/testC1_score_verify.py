#!/usr/bin/env python3
"""C1 — Analytic Ground-Truth Check for the Score Identity.

No network involved. Uses a Gaussian mixture where:
  - The marginal p_t(z) is known in closed form
  - ∇log p_t(z) is computable analytically
  - E[x|z_t] (posterior mean) is computable analytically

Tests both the original B3 formula and the corrected candidate formula
against the true analytic score.

ELF interpolant: z_t = t·x + (1-t)·ε, where ε ~ N(0, σ²I)

For component k with prior N(μ_k, Σ_k), weight w_k:
  z_t | k ~ N(t·μ_k, t²·Σ_k + (1-t)²·σ²·I)

So p_t(z) = Σ_k w_k · N(z; t·μ_k, t²·Σ_k + (1-t)²·σ²·I)

The posterior mean E[x|z_t] = Σ_k r_k(z,t) · [μ_k + Σ_k · t · Σ_zt_k⁻¹ · (z - t·μ_k)]
where r_k is the posterior responsibility and Σ_zt_k = t²·Σ_k + (1-t)²·σ²·I.

Score formulas under test:
  Original (B3):   score = v / ((1-t)·σ²)
  Corrected:       score = (t·x_pred - z) / ((1-t)²·σ²)

where v = (x_pred - z)/(1-t), x_pred = E[x|z_t].
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SIGMA = 1.0  # Noise scale (ELF's denoiser_noise_scale)


class GaussianMixture:
    """A Gaussian mixture with known closed-form score under the linear interpolant."""

    def __init__(self, means, covs, weights):
        """
        Args:
            means: (K, D) array of component means
            covs: (K, D, D) array of component covariances
            weights: (K,) array of mixture weights (sum to 1)
        """
        self.means = np.array(means, dtype=np.float64)
        self.covs = np.array(covs, dtype=np.float64)
        self.weights = np.array(weights, dtype=np.float64)
        self.K, self.D = self.means.shape

    def sample_x(self, n, rng):
        """Sample data points x from the mixture."""
        components = rng.choice(self.K, size=n, p=self.weights)
        x = np.zeros((n, self.D))
        for k in range(self.K):
            mask = components == k
            nk = mask.sum()
            if nk > 0:
                x[mask] = rng.multivariate_normal(self.means[k], self.covs[k], size=nk)
        return x

    def interpolate(self, x, t, rng):
        """Compute z_t = t·x + (1-t)·ε, ε ~ N(0, σ²I)."""
        eps = rng.standard_normal(x.shape) * SIGMA
        return t * x + (1 - t) * eps, eps

    def marginal_cov_k(self, k, t):
        """Covariance of z_t | component k: t²·Σ_k + (1-t)²·σ²·I"""
        # Force symmetry for numerical stability
        cov = t**2 * self.covs[k] + (1 - t)**2 * SIGMA**2 * np.eye(self.D)
        return (cov + cov.T) / 2

    def marginal_mean_k(self, k, t):
        """Mean of z_t | component k: t·μ_k"""
        return t * self.means[k]

    def log_marginal_density(self, z, t):
        """log p_t(z) = log Σ_k w_k · N(z; t·μ_k, Σ_zt_k)"""
        log_components = np.zeros(self.K)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            log_components[k] = (np.log(self.weights[k]) +
                                 multivariate_normal.logpdf(z, mean=mean_k, cov=cov_k,
                                                           allow_singular=True))
        return logsumexp(log_components)

    def analytic_score(self, z, t):
        """True ∇log p_t(z) computed analytically.

        ∇log p_t(z) = Σ_k r_k · Σ_zt_k⁻¹ · (t·μ_k - z)

        where r_k = w_k · N(z; t·μ_k, Σ_zt_k) / Σ_j w_j · N(z; t·μ_j, Σ_zt_j)
        """
        # Compute responsibilities
        log_components = np.zeros(self.K)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            log_components[k] = (np.log(self.weights[k]) +
                                 multivariate_normal.logpdf(z, mean=mean_k, cov=cov_k,
                                                           allow_singular=True))

        log_total = logsumexp(log_components)
        responsibilities = np.exp(log_components - log_total)

        # Score = Σ_k r_k · Σ_zt_k⁻¹ · (t·μ_k - z)
        score = np.zeros(self.D)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            cov_inv = np.linalg.inv(cov_k)
            score += responsibilities[k] * cov_inv @ (mean_k - z)

        return score

    def analytic_posterior_mean(self, z, t):
        """E[x | z_t] — the exact posterior mean (what a perfect network predicts).

        For component k:
          E[x | z_t, k] = μ_k + t·Σ_k · Σ_zt_k⁻¹ · (z - t·μ_k)

        E[x | z_t] = Σ_k r_k · E[x | z_t, k]
        """
        # Responsibilities
        log_components = np.zeros(self.K)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            log_components[k] = (np.log(self.weights[k]) +
                                 multivariate_normal.logpdf(z, mean=mean_k, cov=cov_k,
                                                           allow_singular=True))
        log_total = logsumexp(log_components)
        responsibilities = np.exp(log_components - log_total)

        # Posterior mean per component
        x_pred = np.zeros(self.D)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            cov_zt_inv = np.linalg.inv(cov_k)
            # E[x | z_t, k] = μ_k + t · Σ_k · Σ_zt_k⁻¹ · (z - t·μ_k)
            x_pred_k = self.means[k] + t * self.covs[k] @ cov_zt_inv @ (z - mean_k)
            x_pred += responsibilities[k] * x_pred_k

        return x_pred


def numerical_score(gmm, z, t, eps=1e-5):
    """Finite-difference verification of the analytic score."""
    score_fd = np.zeros(gmm.D)
    for d in range(gmm.D):
        z_plus = z.copy(); z_plus[d] += eps
        z_minus = z.copy(); z_minus[d] -= eps
        score_fd[d] = (gmm.log_marginal_density(z_plus, t) -
                       gmm.log_marginal_density(z_minus, t)) / (2 * eps)
    return score_fd


def original_b3_score(x_pred, z, t, sigma=SIGMA):
    """Original B3 formula: score = v / ((1-t)·σ²)
    where v = (x_pred - z) / (1-t).
    """
    v = (x_pred - z) / max(1 - t, 1e-10)
    return v / (max(1 - t, 1e-10) * sigma**2)


def corrected_score(x_pred, z, t, sigma=SIGMA):
    """Corrected formula: score = (t·x_pred - z) / ((1-t)²·σ²)

    Derivation:
      z_t = t·x + (1-t)·ε  →  ε = (z_t - t·x) / (1-t)
      For p(ε) = N(0, σ²I): ∇_z log p_t(z) = ∇_z [-||ε||²/(2σ²)]
        = -∂ε/∂z · ε / σ²
        = -(1/(1-t)) · (z - t·x) / ((1-t)·σ²)
        = (t·x - z) / ((1-t)²·σ²)

    Replacing x with E[x|z_t] = x_pred:
      score ≈ (t·x_pred - z) / ((1-t)²·σ²)
    """
    denom = max(1 - t, 1e-10)**2 * sigma**2
    return (t * x_pred - z) / denom


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("C1 — Analytic Ground-Truth Score Verification")
    print("=" * 60)

    # Setup: 3-component GMM in 8 dimensions
    D = 8
    rng = np.random.default_rng(42)

    means = np.array([
        [2.0, 1.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0],
        [-1.0, 2.0, 1.0, 0.5, 0.0, -0.5, 0.0, 0.0],
        [0.0, -1.0, -1.0, 2.0, -0.5, 1.0, 0.5, -0.5],
    ])
    # Build well-conditioned PD covariances: A A^T + diagonal
    A1 = 0.3 * rng.standard_normal((D, D))
    A2 = 0.2 * rng.standard_normal((D, D))
    A3 = 0.4 * rng.standard_normal((D, D))
    covs = np.array([
        A1 @ A1.T + 0.5 * np.eye(D),
        A2 @ A2.T + 0.3 * np.eye(D),
        A3 @ A3.T + 0.7 * np.eye(D),
    ])
    # Ensure positive definite
    for k in range(3):
        covs[k] = (covs[k] + covs[k].T) / 2 + 0.3 * np.eye(D)
    weights = np.array([0.4, 0.35, 0.25])

    gmm = GaussianMixture(means, covs, weights)
    print(f"  GMM: K={gmm.K}, D={gmm.D}")

    # ── Step 1: Verify analytic score against finite differences ──
    print("\n── Step 1: Verify analytic score vs finite differences ──")

    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_test = 20
    fd_errors = []

    for t in t_values:
        x_samples = gmm.sample_x(n_test, rng)
        z_samples, _ = gmm.interpolate(x_samples, t, rng)

        errors = []
        for z in z_samples:
            score_analytic = gmm.analytic_score(z, t)
            score_fd = numerical_score(gmm, z, t, eps=1e-5)
            err = np.max(np.abs(score_analytic - score_fd))
            errors.append(err)

        mean_err = np.mean(errors)
        max_err = np.max(errors)
        fd_errors.append({"t": t, "mean_err": mean_err, "max_err": max_err})
        status = "✅" if max_err < 1e-3 else "❌"
        print(f"  t={t:.1f}: mean_err={mean_err:.2e}, max_err={max_err:.2e} {status}")

    fd_pass = all(e["max_err"] < 1e-3 for e in fd_errors)
    print(f"  Finite-difference validation: {'✅ PASS' if fd_pass else '❌ FAIL'}")

    # ── Step 2: Compare both score formulas against analytic truth ──
    print("\n── Step 2: Compare score formulas against analytic truth ──")

    n_test = 50
    results_by_t = {}

    for t in t_values:
        x_samples = gmm.sample_x(n_test, rng)
        z_samples, _ = gmm.interpolate(x_samples, t, rng)

        orig_errors, corr_errors = [], []
        for z in z_samples:
            # True score
            true_score = gmm.analytic_score(z, t)

            # Perfect x prediction (analytic posterior mean)
            x_pred = gmm.analytic_posterior_mean(z, t)

            # Original B3 formula
            score_orig = original_b3_score(x_pred, z, t)
            orig_err = np.linalg.norm(score_orig - true_score)
            orig_errors.append(orig_err)

            # Corrected formula
            score_corr = corrected_score(x_pred, z, t)
            corr_err = np.linalg.norm(score_corr - true_score)
            corr_errors.append(corr_err)

        results_by_t[str(t)] = {
            "original_mean_err": float(np.mean(orig_errors)),
            "original_max_err": float(np.max(orig_errors)),
            "corrected_mean_err": float(np.mean(corr_errors)),
            "corrected_max_err": float(np.max(corr_errors)),
            "ratio": float(np.mean(orig_errors) / max(np.mean(corr_errors), 1e-30)),
        }

        print(f"  t={t:.1f}: original err={np.mean(orig_errors):.6f} "
              f"(max={np.max(orig_errors):.6f})  |  "
              f"corrected err={np.mean(corr_errors):.6f} "
              f"(max={np.max(corr_errors):.6f})  |  "
              f"ratio={np.mean(orig_errors)/max(np.mean(corr_errors),1e-30):.1f}×")

    # ── Step 3: Verify the algebraic equivalence in detail ──
    print("\n── Step 3: Algebraic verification ──")
    print("  Original B3:  score = (x_pred - z) / ((1-t)² · σ²)")
    print("  Corrected:    score = (t·x_pred - z) / ((1-t)² · σ²)")
    print("  Difference:   ((1-t)·x_pred) / ((1-t)² · σ²) = x_pred / ((1-t) · σ²)")
    print("  → The original formula has an extra x_pred/(1-t) term")
    print("  → This grows as t→1 and is NOT negligible")

    # ── Overall Go/No-Go ──
    corrected_pass = all(
        results_by_t[str(t)]["corrected_max_err"] < 1e-6
        for t in t_values
    )
    original_wrong = all(
        results_by_t[str(t)]["original_mean_err"] > results_by_t[str(t)]["corrected_mean_err"] * 5
        for t in t_values
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Error comparison
    ax = axes[0]
    orig_means = [results_by_t[str(t)]["original_mean_err"] for t in t_values]
    corr_means = [results_by_t[str(t)]["corrected_mean_err"] for t in t_values]
    ax.semilogy(t_values, orig_means, "o-", color="#C44E52", lw=2, markersize=8,
                label="Original B3 formula")
    ax.semilogy(t_values, corr_means, "s-", color="#55A868", lw=2, markersize=8,
                label="Corrected formula")
    ax.set_xlabel("t")
    ax.set_ylabel("Mean L2 error vs true score")
    ax.set_title("Score Formula Error vs. Analytic Ground Truth")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Error ratio
    ax = axes[1]
    ratios = [results_by_t[str(t)]["ratio"] for t in t_values]
    ax.bar([str(t) for t in t_values], ratios, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("Error ratio (original / corrected)")
    ax.set_title("How Much Worse is the Original Formula?")
    ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="parity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("C1: Analytic Score Verification (no network)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testC1_score_verification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'testC1_score_verification.png'}")

    # Save
    output = {
        "gmm": {"K": gmm.K, "D": gmm.D, "sigma": SIGMA},
        "fd_validation_pass": fd_pass,
        "fd_errors": fd_errors,
        "score_comparison": results_by_t,
        "corrected_pass": corrected_pass,
        "original_wrong": original_wrong,
    }
    with open(RESULTS_DIR / "testC1_score_verification.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    if corrected_pass and original_wrong:
        print("✅ PASS: Corrected formula matches analytic truth (near machine precision).")
        print("         Original formula is demonstrably wrong (large structured error).")
        print("         Safe to proceed to C2.")
    elif corrected_pass:
        print("✅ Corrected formula passes, but original formula error is smaller than expected.")
        print("   Investigate further before concluding.")
    else:
        print("❌ FAIL: Corrected formula does NOT match analytic truth.")
        print("         Derivation needs another pass. Do NOT proceed to C2.")
    print(f"{'=' * 60}")
    return 0 if corrected_pass else 1


if __name__ == "__main__":
    sys.exit(main())
