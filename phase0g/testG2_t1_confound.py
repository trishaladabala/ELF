#!/usr/bin/env python3
"""G2 — Disambiguating the t→1 Confound.

Tests whether the near-boundary score error is fundamental (amplification)
or fixable (undertraining) by retraining with oversampled t near 1.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0c.testC2_pipeline import build_gmm, XPredictionNetwork

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SIGMA = 1.0


def sample_t_oversampled(batch_size, rng):
    """Sample t with extra density near t=1.

    Mix: 50% standard uniform [0,1], 50% concentrated in [0.9, 1.0].
    """
    n_standard = batch_size // 2
    n_boundary = batch_size - n_standard
    t_standard = rng.uniform(0, 1, size=(n_standard, 1))
    t_boundary = rng.uniform(0.9, 1.0, size=(n_boundary, 1))
    t = np.concatenate([t_standard, t_boundary], axis=0)
    # Shuffle
    perm = rng.permutation(batch_size)
    return t[perm]


def train_network(gmm, time_sampler, n_steps=15000, label=""):
    """Train x-prediction network with a given time sampling strategy."""
    D = gmm.D
    rng = np.random.default_rng(42)
    net = XPredictionNetwork(D).to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    batch_size = 512
    for step in range(n_steps):
        x = gmm.sample_x(batch_size, rng)
        t = time_sampler(batch_size, rng)
        eps = rng.standard_normal(x.shape) * SIGMA
        z = t * x + (1 - t) * eps

        x_pt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        z_pt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        t_pt = torch.tensor(t, dtype=torch.float32, device=DEVICE).squeeze(-1)

        optimizer.zero_grad()
        x_pred = net(z_pt, t_pt)
        loss = ((x_pred - x_pt)**2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 5000 == 0:
            print(f"    [{label}] Step {step+1}/{n_steps}: loss={loss.item():.4f}")

    net.eval()
    return net


def evaluate_score_error(net, gmm, test_ts, n_test=2000, label=""):
    """Evaluate network score error at multiple t values."""
    D = gmm.D
    rng = np.random.default_rng(123)  # Different seed from training
    errors = []

    for t_val in test_ts:
        x = gmm.sample_x(n_test, rng)
        z, _ = gmm.interpolate(x, t_val, rng)

        analytic_scores = np.zeros_like(z)
        for i in range(n_test):
            analytic_scores[i] = gmm.analytic_score(z[i], t_val)

        z_pt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        t_pt = torch.full((n_test,), t_val, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            net_x_preds = net(z_pt, t_pt).cpu().numpy()

        denom_s = max(1 - t_val, 0.05)**2 * SIGMA**2
        net_scores = (t_val * net_x_preds - z) / denom_s

        err = np.linalg.norm(net_scores - analytic_scores, axis=-1)
        mean_err = float(np.mean(err))
        errors.append(mean_err)
        print(f"    [{label}] t={t_val:.2f}: score error = {mean_err:.4f}")

    return np.array(errors)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("G2 — Disambiguating the t→1 Confound")
    print("=" * 60)

    gmm = build_gmm()

    # Dense t-grid near boundary
    test_ts = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

    # ── Original: uniform time sampling ──
    print("\n── Training with STANDARD uniform t ──")
    def uniform_sampler(bs, rng):
        return rng.uniform(0, 1, size=(bs, 1))

    net_standard = train_network(gmm, uniform_sampler, n_steps=15000, label="standard")
    errors_standard = evaluate_score_error(net_standard, gmm, test_ts, label="standard")

    # ── Oversampled: extra density near t=1 ──
    print("\n── Training with OVERSAMPLED t near 1 ──")
    net_oversampled = train_network(gmm, sample_t_oversampled, n_steps=15000, label="oversampled")
    errors_oversampled = evaluate_score_error(net_oversampled, gmm, test_ts, label="oversampled")

    # Fit slopes in log-log
    one_minus_t = np.array([1.0 - t for t in test_ts])
    log_x = np.log(one_minus_t)

    log_y_std = np.log(errors_standard)
    slope_std = np.polyfit(log_x, log_y_std, 1)[0]

    log_y_over = np.log(errors_oversampled)
    slope_over = np.polyfit(log_x, log_y_over, 1)[0]

    print(f"\n  Standard slope:    {slope_std:.3f}")
    print(f"  Oversampled slope: {slope_over:.3f}")
    print(f"  Theoretical:       -2.0")

    # Check near-boundary improvement
    err_ratio_99 = errors_oversampled[-1] / errors_standard[-1]
    err_ratio_95 = errors_oversampled[-3] / errors_standard[-3]
    substantial_improvement = err_ratio_99 < 0.5  # At least 2x improvement at t=0.99

    print(f"\n  Error ratio at t=0.95: {err_ratio_95:.3f} (oversampled/standard)")
    print(f"  Error ratio at t=0.99: {err_ratio_99:.3f} (oversampled/standard)")

    # Interpretation
    print(f"\n{'=' * 60}")
    if substantial_improvement:
        print("🟢 G2: Oversampling t near 1 substantially reduced boundary error.")
        print("   → The instability is PARTLY FIXABLE via training-schedule design.")
        print("   → This is an engineering lever for Idea 6, not a fundamental dead end.")
        interpretation = "undertraining_contributes"
    else:
        if abs(slope_over - slope_std) < 0.15:
            print("🔴 G2: Oversampling had negligible effect on boundary error.")
            print("   → The instability is FUNDAMENTAL to the exact-SDE formulation.")
            print("   → Idea 6's framing should be 'characterizing a real tradeoff'.")
            interpretation = "fundamental_instability"
        else:
            print("🟡 G2: Some improvement from oversampling, but error still dominant.")
            print("   → Both factors contribute: partial undertraining + fundamental amplification.")
            interpretation = "mixed"
    print(f"{'=' * 60}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(one_minus_t, errors_standard, 'o-', color='#C44E52',
              label=f'Standard (slope={slope_std:.2f})', markersize=8, lw=2)
    ax.loglog(one_minus_t, errors_oversampled, 's-', color='#4C72B0',
              label=f'Oversampled t∈[0.9,1] (slope={slope_over:.2f})', markersize=8, lw=2)

    # Reference line
    ref_y = errors_standard[0] * (one_minus_t / one_minus_t[0])**(-2.0)
    ax.loglog(one_minus_t, ref_y, 'k--', label=r'Theoretical $O((1-t)^{-2})$', alpha=0.5)

    ax.set_xlabel('1 - t', fontsize=12)
    ax.set_ylabel('Mean Score Error (L2)', fontsize=12)
    ax.set_title('G2: Is the Near-Boundary Error Fixable?', fontsize=13)
    ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testG2_t1_confound_disambiguation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testG2_t1_confound_disambiguation.png'}")

    # Save
    with open(RESULTS_DIR / "testG2_t1_confound_disambiguation.json", "w") as f:
        json.dump({
            "test_ts": test_ts,
            "errors_standard": [float(e) for e in errors_standard],
            "errors_oversampled": [float(e) for e in errors_oversampled],
            "slope_standard": float(slope_std),
            "slope_oversampled": float(slope_over),
            "err_ratio_t099": float(err_ratio_99),
            "err_ratio_t095": float(err_ratio_95),
            "substantial_improvement": bool(substantial_improvement),
            "interpretation": interpretation,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
