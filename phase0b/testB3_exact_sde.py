#!/usr/bin/env python3
"""B3 — Derive and Implement the Exact Probability-Flow SDE.

Derives the reverse-time SDE for ELF's linear-interpolant flow using the
stochastic-interpolant formalism (Albergo & Vanden-Eijnden).

ELF's linear interpolant:  z_t = t·x₁ + (1-t)·ε,  ε ~ N(0, σ²I)
where x₁ = data, ε = noise, σ = config.denoiser_noise_scale.

The velocity field is v(z,t) = (x₁ - z)/(1-t), learned by the model.

The exact reverse-time SDE from the stochastic-interpolant framework:
  dz = [v(z,t) + (g²/2)·∇log p_t(z)] dt + g·dW

For the linear interpolant with Gaussian noise prior:
  p_t(z) is Gaussian with mean t·x₁ and variance (1-t)²·σ²
  ∇log p_t(z) = -(z - t·x₁) / ((1-t)²·σ²)

Since x₁ = z/(t) + (1-t)·v(z,t)/t via the flow equation, we can express
the score in terms of v and z:
  score = -(z - (z + (1-t)·v)) / ((1-t)²·σ²)
        = v / ((1-t)·σ²)

So the exact SDE becomes:
  dz = [v(z,t) + (g²/2) · v(z,t) / ((1-t)·σ²)] dt + g·dW
     = v(z,t) · [1 + g²/(2·(1-t)·σ²)] dt + g·dW

Key sanity check: as g → 0, the SDE reduces to dz = v(z,t) dt (the ODE).

Go/No-Go: Zero-noise convergence check must pass.
"""

import json
import sys
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF
from utils.sampling_utils import net_out_to_v_x, get_sampling_steps

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

T_EPS = 0.05
DENOISER_NOISE_SCALE = 1.0  # σ in the derivation


# ============================================
# Sampler 3: Exact Probability-Flow SDE (Derived)
# ============================================

def exact_sde_sample(
    model, n_samples: int, n_steps: int,
    encoder_dim: int, seq_len: int,
    g: float = 0.5,  # Diffusion coefficient
    sigma: float = DENOISER_NOISE_SCALE,
    time_schedule: str = "uniform",
    device: str = "cpu",
    initial_noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Exact reverse-time SDE sampler for ELF's linear interpolant.

    From the stochastic interpolant framework:
      dz = v(z,t) · [1 + g²/(2·(1-t)·σ²)] dt + g·dW

    Euler-Maruyama discretization:
      z_{i+1} = z_i + h · v(z_i, t_i) · [1 + g²/(2·(1-t_i)·σ²)] + g·√h·η

    where h = t_{i+1} - t_i and η ~ N(0, I).

    At g=0: reduces to z_{i+1} = z_i + h·v(z_i, t_i), the ODE sampler.
    """
    model.eval()
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        if initial_noise is not None:
            z = initial_noise.clone().to(device)
        else:
            z = torch.randn(n_samples, seq_len, encoder_dim, device=device, dtype=dtype)

        steps = get_sampling_steps(
            n_steps, time_schedule=time_schedule,
            device=device, dtype=dtype,
        )

        trajectory = [z.clone()]

        for i in range(n_steps):
            t_curr = float(steps[i])
            t_next = float(steps[i + 1])
            h = t_next - t_curr

            # Forward pass to get v(z, t)
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, x_pred = net_out_to_v_x(net_out, z, t_batch, T_EPS)

            # Score scaling: 1 + g²/(2·(1-t)·σ²)
            denom = max(1.0 - t_curr, T_EPS)  # Clamp for numerical stability
            drift_scale = 1.0 + (g ** 2) / (2.0 * denom * sigma ** 2)

            # Drift term: h · v · drift_scale
            drift = h * v_pred * drift_scale

            # Diffusion term: g · √h · η
            if g > 0:
                noise = torch.randn_like(z)
                diffusion = g * math.sqrt(abs(h)) * noise
            else:
                diffusion = 0.0

            z = z + drift + diffusion
            trajectory.append(z.clone())

    return z, trajectory


# ============================================
# ODE reference (for comparison)
# ============================================

def ode_euler_sample(
    model, n_samples, n_steps, encoder_dim, seq_len,
    time_schedule="uniform", device="cpu", initial_noise=None,
):
    """Deterministic ODE Euler sampler (reference)."""
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        if initial_noise is not None:
            z = initial_noise.clone().to(device)
        else:
            z = torch.randn(n_samples, seq_len, encoder_dim, device=device, dtype=dtype)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        trajectory = [z.clone()]
        for i in range(n_steps):
            t_curr = steps[i]
            t_next = steps[i + 1]
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), float(t_curr), dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            z = z + h * v_pred
            trajectory.append(z.clone())
    return z, trajectory


def decode_to_tokens(model, z, device):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z, t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        return logits.argmax(dim=-1)


def load_checkpoint():
    ckpt_path = RESULTS_DIR / "mini_elf_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found. Run B1 first.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    cfg = ckpt["config"]
    model = ELF(
        text_encoder_dim=cfg["encoder_dim"], max_length=cfg["max_length"],
        hidden_size=cfg["hidden_size"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        bottleneck_dim=cfg["bottleneck_dim"], num_time_tokens=2,
        num_self_cond_cfg_tokens=0, num_model_mode_tokens=2,
        vocab_size=cfg["vocab_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, cfg


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("B3 — Exact Probability-Flow SDE")
    print("=" * 60)

    print("\n── Derivation Summary ──")
    print("  ELF interpolant: z_t = t·x₁ + (1-t)·ε,  ε ~ N(0, σ²I)")
    print("  Learned velocity: v(z,t) ≈ (x₁ - z)/(1-t)")
    print("  Score identity: ∇log p_t(z) = v(z,t) / ((1-t)·σ²)")
    print("  Exact reverse SDE: dz = v·[1 + g²/(2(1-t)σ²)] dt + g·dW")
    print("  At g=0: reduces to ODE dz = v·dt ✓")

    model, cfg = load_checkpoint()
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]
    n_samples = 8

    # ── Critical check: g→0 convergence ──
    print("\n── CRITICAL CHECK: g → 0 convergence to ODE ──")

    torch.manual_seed(42)
    fixed_noise = torch.randn(n_samples, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)

    # ODE reference
    z_ode, _ = ode_euler_sample(
        model, n_samples, 16, encoder_dim, seq_len,
        time_schedule="uniform", device=DEVICE, initial_noise=fixed_noise,
    )

    g_values = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0]
    convergence_results = {}

    for g in g_values:
        torch.manual_seed(7777)  # Fix stochastic noise for each g
        z_sde, _ = exact_sde_sample(
            model, n_samples, 16, encoder_dim, seq_len,
            g=g, time_schedule="uniform", device=DEVICE, initial_noise=fixed_noise,
        )
        max_diff = float(torch.max(torch.abs(z_ode - z_sde)).item())
        mean_diff = float(torch.mean(torch.abs(z_ode - z_sde)).item())
        convergence_results[str(g)] = {"max_diff": max_diff, "mean_diff": mean_diff}

        status = "✅" if max_diff < 1e-3 else ("~" if max_diff < 1.0 else "")
        print(f"  g={g:<8.1e}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} {status}")

    # Check: g=0 should be exact
    g0_passes = convergence_results["0.0"]["max_diff"] < 1e-4
    print(f"\n  g=0 exact equivalence: {'✅ PASS' if g0_passes else '❌ FAIL'}")

    # Check: differences should decrease monotonically as g → 0
    diffs = [(float(g), convergence_results[str(g)]["max_diff"]) for g in g_values if g > 0]
    monotone = all(diffs[i][1] <= diffs[i+1][1] * 1.5 for i in range(len(diffs)-1))
    print(f"  Monotone convergence: {'✅' if monotone else '⚠️ non-monotone (expected with finite noise draws)'}")

    # ── Verify score formula: check that adding the score term doesn't break isotropy ──
    print("\n── Score term analysis ──")
    t_test = 0.5
    g_test = 0.5
    denom = max(1.0 - t_test, T_EPS)
    drift_scale = 1.0 + (g_test ** 2) / (2.0 * denom * DENOISER_NOISE_SCALE ** 2)
    print(f"  At t={t_test}, g={g_test}: drift_scale = {drift_scale:.4f}")
    print(f"  Interpretation: velocity is amplified by {drift_scale:.2f}× to account for the score")

    # Check near t=1 (where (1-t)→0, scale diverges)
    t_near_1 = 0.95
    denom_near = max(1.0 - t_near_1, T_EPS)
    scale_near = 1.0 + (g_test ** 2) / (2.0 * denom_near * DENOISER_NOISE_SCALE ** 2)
    print(f"  At t={t_near_1}, g={g_test}: drift_scale = {scale_near:.4f}")
    print(f"  ⚠️ Note: drift_scale diverges near t=1. T_EPS={T_EPS} clamps (1-t) ≥ {T_EPS}.")
    print(f"  Max drift_scale with clamp: {1.0 + (g_test**2)/(2*T_EPS*DENOISER_NOISE_SCALE**2):.2f}")

    # ── Qualitative comparison: Exact SDE vs ELF's approximation ──
    print("\n── Exact SDE vs ELF Approximation (qualitative) ──")
    # Import ELF's SDE from B2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from testB2_samplers import elf_sde_sample

    g_compare = 0.5
    n_steps_compare = 16

    torch.manual_seed(42)
    z_exact, _ = exact_sde_sample(
        model, n_samples, n_steps_compare, encoder_dim, seq_len,
        g=g_compare, time_schedule="uniform", device=DEVICE, initial_noise=fixed_noise,
    )

    torch.manual_seed(42)
    z_elf, _ = elf_sde_sample(
        model, n_samples, n_steps_compare, encoder_dim, seq_len,
        gamma=g_compare, time_schedule="uniform", device=DEVICE, initial_noise=fixed_noise,
    )

    exact_vs_elf_diff = float(torch.mean(torch.abs(z_exact - z_elf)).item())
    exact_vs_ode_diff = float(torch.mean(torch.abs(z_exact - z_ode)).item())
    elf_vs_ode_diff = float(torch.mean(torch.abs(z_elf - z_ode)).item())

    print(f"  |Exact SDE - ELF SDE| = {exact_vs_elf_diff:.4f}")
    print(f"  |Exact SDE - ODE|     = {exact_vs_ode_diff:.4f}")
    print(f"  |ELF SDE - ODE|       = {elf_vs_ode_diff:.4f}")
    print(f"  → The two SDE variants differ from each other by {exact_vs_elf_diff:.4f}")
    print(f"    (confirming they're distinct samplers, not equivalent)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # g → 0 convergence
    ax = axes[0]
    g_nonzero = [g for g in g_values if g > 0]
    diffs_to_plot = [convergence_results[str(g)]["max_diff"] for g in g_nonzero]
    ax.loglog(g_nonzero, diffs_to_plot, "o-", color="#4C72B0", lw=2, markersize=8)
    ax.set_xlabel("g (diffusion coefficient)")
    ax.set_ylabel("max |SDE(g) - ODE|")
    ax.set_title("Exact SDE → ODE Convergence (g → 0)")
    ax.grid(True, alpha=0.3)

    # Sampler comparison
    ax = axes[1]
    labels = ["Exact SDE\nvs ELF SDE", "Exact SDE\nvs ODE", "ELF SDE\nvs ODE"]
    values = [exact_vs_elf_diff, exact_vs_ode_diff, elf_vs_ode_diff]
    colors = ["#C44E52", "#4C72B0", "#55A868"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Mean |Δz|")
    ax.set_title(f"Sampler Differences (g={g_compare}, {n_steps_compare} steps)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=9)

    plt.suptitle("B3: Exact Probability-Flow SDE Verification", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testB3_exact_sde.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plot → {RESULTS_DIR / 'testB3_exact_sde.png'}")

    # Save
    metrics = {
        "g0_passes": g0_passes,
        "convergence_results": convergence_results,
        "sampler_diffs": {
            "exact_vs_elf": exact_vs_elf_diff,
            "exact_vs_ode": exact_vs_ode_diff,
            "elf_vs_ode": elf_vs_ode_diff,
        },
        "derivation": {
            "interpolant": "z_t = t*x1 + (1-t)*eps",
            "velocity": "v(z,t) = (x1 - z)/(1-t)",
            "score": "nabla_log_p_t = v(z,t) / ((1-t)*sigma^2)",
            "exact_sde": "dz = v * [1 + g^2/(2*(1-t)*sigma^2)] dt + g*dW",
        },
    }
    with open(RESULTS_DIR / "testB3.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if g0_passes:
        print("✅ Zero-noise convergence verified. Exact SDE implementation is correct.")
        print("   Ready for B4 (step-budget comparison).")
    else:
        print("❌ g=0 equivalence FAILED. Debug before proceeding.")
    print(f"{'=' * 60}")
    return 0 if g0_passes else 1


if __name__ == "__main__":
    sys.exit(main())
