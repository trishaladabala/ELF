#!/usr/bin/env python3
"""E2 — Near-t=1 Stability Characterization.

Characterize the growth of the score error as t -> 1 using the GMM setup from C2.
"""

import json
import sys
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("E2 — Near-t=1 Stability Characterization")
    print("=" * 60)

    gmm = build_gmm()
    D = gmm.D
    rng = np.random.default_rng(42)

    # Re-train the same small network on GMM data
    print("Training toy x-prediction network on GMM data...")
    net = XPredictionNetwork(D).to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)

    batch_size = 512
    n_steps = 10000
    for step in range(n_steps):
        x = gmm.sample_x(batch_size, rng)
        t = rng.uniform(0, 1, size=(batch_size, 1))
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

        if (step + 1) % 2500 == 0:
            print(f"  Step {step+1}/{n_steps}: loss={loss.item():.4f}")

    net.eval()

    # Densely sample near t=1
    test_ts = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    n_test = 2000

    print("\nEvaluating network score vs analytic score near t=1...")
    errors_network = []
    
    # We also evaluate ELF's heuristic equivalent near t=1
    # ELF heuristic effectively uses: score_heuristic = (x_pred - z) / denom
    # Wait, ELF's SDE doesn't use the explicit score. It uses the velocity v.
    # The drift is v. The score formula itself is only used in the EXACT SDE.
    # We'll just track the error in the exact score formula when substituting x_pred instead of x_analytic.
    
    for t_val in test_ts:
        x = gmm.sample_x(n_test, rng)
        z, _ = gmm.interpolate(x, t_val, rng)

        # Analytic scores
        analytic_scores = np.zeros_like(z)
        analytic_x_preds = np.zeros_like(z)
        for i in range(n_test):
            analytic_scores[i] = gmm.analytic_score(z[i], t_val)
            analytic_x_preds[i] = gmm.analytic_posterior_mean(z[i], t_val)

        # Network scores
        z_pt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        t_pt = torch.full((n_test,), t_val, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            net_x_preds = net(z_pt, t_pt).cpu().numpy()

        # Compute corrected score using network x_pred
        denom_s = max(1 - t_val, 0.05)**2 * SIGMA**2
        net_scores = (t_val * net_x_preds - z) / denom_s
        
        # We also compute the exact score using analytic x_pred (which we know from C1 matches the true score)
        analytic_formula_scores = (t_val * analytic_x_preds - z) / denom_s

        err_net = np.linalg.norm(net_scores - analytic_scores, axis=-1)
        mean_err_net = np.mean(err_net)
        errors_network.append(mean_err_net)

        print(f"  t={t_val:.2f} (1-t={1-t_val:.2f}): Mean Score Error = {mean_err_net:.4f}")

    # Plot
    one_minus_t = np.array([1.0 - t for t in test_ts])
    err = np.array(errors_network)
    
    # Fit line in log-log space
    log_x = np.log(one_minus_t)
    log_y = np.log(err)
    coeffs = np.polyfit(log_x, log_y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    
    print(f"\n  Fitted slope on log-log scale: {slope:.3f} (Predicted: ~-2.0)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(one_minus_t, err, 'o-', color='#C44E52', label='Network Score Error', markersize=8)
    
    # Reference line for 1/(1-t)^2
    ref_y = np.exp(intercept) * (one_minus_t / one_minus_t[0])**(-2.0)
    ax.loglog(one_minus_t, ref_y, 'k--', label=r'Theoretical $O((1-t)^{-2})$ slope', alpha=0.6)
    
    ax.set_xlabel('1 - t')
    ax.set_ylabel('Mean Score Error (L2)')
    ax.set_title('E2: Network Score Error Growth near $t=1$')
    # reverse x axis since 1-t approaches 0 on the right
    ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0]) 
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testE2_boundary_stability.png", dpi=150)
    plt.close()
    print(f"  Saved plot → {RESULTS_DIR / 'testE2_boundary_stability.png'}")

    with open(RESULTS_DIR / "testE2.json", "w") as f:
        json.dump({
            "t_vals": test_ts,
            "errors": [float(e) for e in errors_network],
            "slope": float(slope),
            "matches_theory": bool(abs(slope - (-2.0)) < 0.3)
        }, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
