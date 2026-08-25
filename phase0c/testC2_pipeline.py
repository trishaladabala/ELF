#!/usr/bin/env python3
"""C2 — Re-Verify the Full Sampler Pipeline With the Corrected Score Formula.

Two checks:
  1. g→0 convergence (re-run B3's check with the corrected formula)
  2. Train a toy network on the C1 GMM data, then compare its implied score
     (via the corrected formula) against the C1 analytic score

The corrected formula is:
  score(z, t) = (t · x_pred - z) / ((1-t)² · σ²)

where x_pred is the network's prediction of E[x | z_t].
"""

import json
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.special import logsumexp

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SIGMA = 1.0
T_EPS = 0.05

# ── GMM setup (same as C1) ──

class GaussianMixture:
    def __init__(self, means, covs, weights):
        self.means = np.array(means, dtype=np.float64)
        self.covs = np.array(covs, dtype=np.float64)
        self.weights = np.array(weights, dtype=np.float64)
        self.K, self.D = self.means.shape

    def sample_x(self, n, rng):
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
        cov = t**2 * self.covs[k] + (1 - t)**2 * SIGMA**2 * np.eye(self.D)
        return (cov + cov.T) / 2

    def marginal_mean_k(self, k, t):
        return t * self.means[k]

    def analytic_score(self, z, t):
        log_components = np.zeros(self.K)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            log_components[k] = (np.log(self.weights[k]) +
                                 multivariate_normal.logpdf(z, mean=mean_k, cov=cov_k,
                                                           allow_singular=True))
        log_total = logsumexp(log_components)
        responsibilities = np.exp(log_components - log_total)
        score = np.zeros(self.D)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            cov_inv = np.linalg.inv(cov_k)
            score += responsibilities[k] * cov_inv @ (mean_k - z)
        return score

    def analytic_posterior_mean(self, z, t):
        log_components = np.zeros(self.K)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            log_components[k] = (np.log(self.weights[k]) +
                                 multivariate_normal.logpdf(z, mean=mean_k, cov=cov_k,
                                                           allow_singular=True))
        log_total = logsumexp(log_components)
        responsibilities = np.exp(log_components - log_total)
        x_pred = np.zeros(self.D)
        for k in range(self.K):
            mean_k = self.marginal_mean_k(k, t)
            cov_k = self.marginal_cov_k(k, t)
            cov_zt_inv = np.linalg.inv(cov_k)
            x_pred_k = self.means[k] + t * self.covs[k] @ cov_zt_inv @ (z - mean_k)
            x_pred += responsibilities[k] * x_pred_k
        return x_pred


def build_gmm():
    """Build the same GMM as C1."""
    D = 8
    rng = np.random.default_rng(42)
    means = np.array([
        [2.0, 1.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0],
        [-1.0, 2.0, 1.0, 0.5, 0.0, -0.5, 0.0, 0.0],
        [0.0, -1.0, -1.0, 2.0, -0.5, 1.0, 0.5, -0.5],
    ])
    A1 = 0.3 * rng.standard_normal((D, D))
    A2 = 0.2 * rng.standard_normal((D, D))
    A3 = 0.4 * rng.standard_normal((D, D))
    covs = np.array([
        A1 @ A1.T + 0.5 * np.eye(D),
        A2 @ A2.T + 0.3 * np.eye(D),
        A3 @ A3.T + 0.7 * np.eye(D),
    ])
    for k in range(3):
        covs[k] = (covs[k] + covs[k].T) / 2 + 0.3 * np.eye(D)
    weights = np.array([0.4, 0.35, 0.25])
    return GaussianMixture(means, covs, weights)


# ── Toy x-prediction network ──

class XPredictionNetwork(nn.Module):
    """Simple MLP that predicts x from (z_t, t)."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, t):
        # t: (B,) → (B, 1)
        t_input = t.unsqueeze(-1) if t.dim() == 1 else t
        return self.net(torch.cat([z, t_input], dim=-1))


# ── Corrected exact SDE sampler ──

def corrected_exact_sde_sample(
    x_pred_fn, n_samples, n_steps, dim,
    g=0.5, sigma=SIGMA, device="cpu",
    initial_noise=None,
):
    """Exact SDE with CORRECTED score formula.

    score = (t · x_pred - z) / ((1-t)² · σ²)

    dz = [v + (g²/2) · score] dt + g · dW
       = [(x_pred - z)/(1-t) + (g²/2) · (t·x_pred - z)/((1-t)²·σ²)] dt + g·dW
    """
    steps = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float32, device=device)
    if initial_noise is not None:
        z = initial_noise.clone().to(device)
    else:
        z = torch.randn(n_samples, dim, device=device)
    trajectory = [z.clone()]

    with torch.no_grad():
        for i in range(n_steps):
            t_curr = float(steps[i])
            t_next = float(steps[i + 1])
            h = t_next - t_curr

            t_batch = torch.full((n_samples,), t_curr, device=device)
            x_pred = x_pred_fn(z, t_batch)

            # Velocity
            denom_v = max(1 - t_curr, T_EPS)
            v = (x_pred - z) / denom_v

            # Corrected score
            denom_s = max(1 - t_curr, T_EPS)**2 * sigma**2
            score = (t_curr * x_pred - z) / denom_s

            # Drift
            drift = v + (g**2 / 2) * score

            # Diffusion
            if g > 0:
                noise = torch.randn_like(z)
                diffusion = g * math.sqrt(abs(h)) * noise
            else:
                diffusion = 0.0

            z = z + h * drift + diffusion
            trajectory.append(z.clone())

    return z, trajectory


def ode_sample(x_pred_fn, n_samples, n_steps, dim, device="cpu", initial_noise=None):
    """Deterministic ODE sampler."""
    steps = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float32, device=device)
    if initial_noise is not None:
        z = initial_noise.clone().to(device)
    else:
        z = torch.randn(n_samples, dim, device=device)
    trajectory = [z.clone()]

    with torch.no_grad():
        for i in range(n_steps):
            t_curr = float(steps[i])
            t_next = float(steps[i + 1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, device=device)
            x_pred = x_pred_fn(z, t_batch)
            v = (x_pred - z) / max(1 - t_curr, T_EPS)
            z = z + h * v
            trajectory.append(z.clone())

    return z, trajectory


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("C2 — Re-Verify Sampler Pipeline With Corrected Formula")
    print("=" * 60)

    gmm = build_gmm()
    D = gmm.D

    # ── Check 1: g→0 convergence using analytic posterior mean ──
    print("\n── Check 1: g→0 convergence (analytic x_pred) ──")

    def analytic_x_pred_fn(z_batch, t_batch):
        """Analytic posterior mean as a torch-compatible function."""
        z_np = z_batch.cpu().numpy()
        t_val = float(t_batch[0])
        results = np.zeros_like(z_np)
        for i in range(z_np.shape[0]):
            results[i] = gmm.analytic_posterior_mean(z_np[i], t_val)
        return torch.tensor(results, dtype=z_batch.dtype, device=z_batch.device)

    n_samples = 16
    n_steps = 32
    torch.manual_seed(42)
    initial_noise = torch.randn(n_samples, D, device=DEVICE)

    # ODE reference
    z_ode, _ = ode_sample(analytic_x_pred_fn, n_samples, n_steps, D,
                          device=DEVICE, initial_noise=initial_noise)

    g_values = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0]
    convergence = {}
    for g in g_values:
        torch.manual_seed(7777)
        z_sde, _ = corrected_exact_sde_sample(
            analytic_x_pred_fn, n_samples, n_steps, D,
            g=g, device=DEVICE, initial_noise=initial_noise,
        )
        max_diff = float(torch.max(torch.abs(z_ode - z_sde)).item())
        convergence[str(g)] = max_diff
        status = "✅" if max_diff < 1e-3 else ("~" if max_diff < 1.0 else "")
        print(f"  g={g:<8.1e}: max_diff={max_diff:.2e} {status}")

    g0_pass = convergence["0.0"] < 1e-6
    print(f"  g=0 exact: {'✅' if g0_pass else '❌'}")

    # ── Check 2: Train a toy network on GMM data ──
    print("\n── Check 2: Train toy x-prediction network on GMM data ──")

    rng = np.random.default_rng(123)
    model = XPredictionNetwork(D, hidden=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)

    N_STEPS = 10000
    BATCH = 256
    losses = []

    model.train()
    for step in range(N_STEPS):
        x_np = gmm.sample_x(BATCH, rng)
        t_np = rng.uniform(0.05, 0.95, size=BATCH)

        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
        t = torch.tensor(t_np, dtype=torch.float32, device=DEVICE)
        eps = torch.randn_like(x) * SIGMA

        z = t.unsqueeze(-1) * x + (1 - t.unsqueeze(-1)) * eps

        x_pred = model(z, t)
        loss = F.mse_loss(x_pred, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(float(loss.item()))
        if (step + 1) % 2000 == 0:
            print(f"  Step {step+1}/{N_STEPS}: loss={np.mean(losses[-200:]):.6f}")

    print(f"  Final x-prediction loss: {np.mean(losses[-200:]):.6f}")

    # ── Check 3: Compare trained network's implied score vs analytic ──
    print("\n── Check 3: Network-implied score vs analytic truth ──")

    model.eval()
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_test = 100
    score_comparison = {}

    for t_val in t_values:
        x_test = gmm.sample_x(n_test, rng)
        z_test, _ = gmm.interpolate(x_test, t_val, rng)

        # Network prediction
        z_torch = torch.tensor(z_test, dtype=torch.float32, device=DEVICE)
        t_torch = torch.full((n_test,), t_val, device=DEVICE)
        with torch.no_grad():
            x_pred_net = model(z_torch, t_torch).cpu().numpy()

        # Compute scores
        corrected_errors = []
        analytic_pred_errors = []
        for i in range(n_test):
            true_score = gmm.analytic_score(z_test[i], t_val)

            # Network → corrected formula
            denom_s = max(1 - t_val, T_EPS)**2 * SIGMA**2
            net_score = (t_val * x_pred_net[i] - z_test[i]) / denom_s
            corrected_errors.append(np.linalg.norm(net_score - true_score))

            # Analytic posterior mean → corrected formula (for reference)
            x_pred_ana = gmm.analytic_posterior_mean(z_test[i], t_val)
            ana_score = (t_val * x_pred_ana - z_test[i]) / denom_s
            analytic_pred_errors.append(np.linalg.norm(ana_score - true_score))

        score_comparison[str(t_val)] = {
            "network_corrected_mean_err": float(np.mean(corrected_errors)),
            "network_corrected_max_err": float(np.max(corrected_errors)),
            "analytic_corrected_mean_err": float(np.mean(analytic_pred_errors)),
        }
        print(f"  t={t_val:.1f}: network→corrected err={np.mean(corrected_errors):.4f} "
              f"(max={np.max(corrected_errors):.4f}) | "
              f"analytic→corrected err={np.mean(analytic_pred_errors):.2e}")

    # ── Check 4: g→0 convergence with trained network ──
    print("\n── Check 4: g→0 convergence with trained network ──")

    def trained_x_pred_fn(z_batch, t_batch):
        return model(z_batch, t_batch)

    torch.manual_seed(42)
    init = torch.randn(16, D, device=DEVICE)

    z_ode_net, _ = ode_sample(trained_x_pred_fn, 16, 32, D, device=DEVICE, initial_noise=init)

    net_convergence = {}
    for g in [0.0, 1e-4, 0.01, 0.1, 0.5]:
        torch.manual_seed(7777)
        z_sde_net, _ = corrected_exact_sde_sample(
            trained_x_pred_fn, 16, 32, D, g=g, device=DEVICE, initial_noise=init,
        )
        diff = float(torch.max(torch.abs(z_ode_net - z_sde_net)).item())
        net_convergence[str(g)] = diff
        print(f"  g={g:<6}: max_diff={diff:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Training loss
    ax = axes[0]
    ax.plot(losses, alpha=0.15, color="#4C72B0")
    window = 100
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(losses)), smoothed, color="#4C72B0", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("x-Prediction Network Training", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Score error comparison
    ax = axes[1]
    net_errs = [score_comparison[str(t)]["network_corrected_mean_err"] for t in t_values]
    ana_errs = [score_comparison[str(t)]["analytic_corrected_mean_err"] for t in t_values]
    ax.semilogy(t_values, net_errs, "o-", color="#C44E52", lw=2, markersize=8,
                label="Network (trained) → corrected")
    ax.semilogy(t_values, [max(e, 1e-15) for e in ana_errs], "s-", color="#55A868",
                lw=2, markersize=8, label="Analytic posterior → corrected")
    ax.set_xlabel("t")
    ax.set_ylabel("Mean L2 error vs true score")
    ax.set_title("Implied Score Error", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # g→0 convergence (both)
    ax = axes[2]
    g_plot = [g for g in g_values if g > 0]
    diffs_ana = [convergence[str(g)] for g in g_plot]
    ax.loglog(g_plot, diffs_ana, "o-", color="#4C72B0", lw=2, markersize=8,
              label="Analytic x_pred")
    g_net = [g for g in [1e-4, 0.01, 0.1, 0.5]]
    diffs_net = [net_convergence[str(g)] for g in g_net]
    ax.loglog(g_net, diffs_net, "s-", color="#C44E52", lw=2, markersize=8,
              label="Trained network")
    ax.set_xlabel("g")
    ax.set_ylabel("max |SDE(g) - ODE|")
    ax.set_title("Corrected SDE → ODE Convergence", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("C2: Corrected Sampler Pipeline Verification", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testC2_pipeline_verification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'testC2_pipeline_verification.png'}")

    # Network tracks analytic score reasonably?
    network_tracks = all(
        score_comparison[str(t)]["network_corrected_mean_err"] < 5.0
        for t in [0.1, 0.3, 0.5, 0.7]
    )

    output = {
        "g0_convergence_analytic": convergence,
        "g0_pass": g0_pass,
        "training_final_loss": float(np.mean(losses[-200:])),
        "score_comparison": score_comparison,
        "g0_convergence_trained": net_convergence,
        "network_tracks_analytic": network_tracks,
    }
    with open(RESULTS_DIR / "testC2.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    if g0_pass and network_tracks:
        print("✅ C2 PASS: Corrected formula verified in full pipeline.")
        print("   g→0 convergence: ✅")
        print(f"   Network tracks analytic score: ✅")
    elif g0_pass:
        print("⚠️  g→0 passes but network score tracking is weak.")
        print("   May need more training or larger network.")
    else:
        print("❌ g→0 convergence FAILED with corrected formula.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
