#!/usr/bin/env python3
"""C5 — Rerun B4 Step-Budget Comparison With All Three Fixes.

Uses:
  1. Corrected score formula (from C1/C2)
  2. Understood diversity metric (from C3 — now meaningful on real data)
  3. Real language model checkpoint (from C4)

Same protocol as B4: 5 samplers × 5 step budgets × 2 time schedules.
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
SIGMA = 1.0
N_SAMPLES = 32
STEP_BUDGETS = [4, 8, 16, 32, 64]
TIME_SCHEDULES = ["uniform", "logit_normal"]


def load_real_checkpoint():
    ckpt_path = RESULTS_DIR / "mini_elf_real_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"❌ Real checkpoint not found. Run C4 first.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    return model, cfg, ckpt.get("training", {})


# ── ODE Sampler ──

def ode_sample(model, n_samples, n_steps, encoder_dim, seq_len,
               time_schedule="uniform", device="cpu", initial_noise=None):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = initial_noise.clone().to(device) if initial_noise is not None else \
            torch.randn(n_samples, seq_len, encoder_dim, device=device, dtype=dtype)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            z = z + h * v_pred
    return z


# ── ELF's SDE (Algorithm 6) ──

def elf_sde_sample(model, n_samples, n_steps, encoder_dim, seq_len,
                   gamma=0.5, time_schedule="uniform", device="cpu",
                   initial_noise=None):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = initial_noise.clone().to(device) if initial_noise is not None else \
            torch.randn(n_samples, seq_len, encoder_dim, device=device, dtype=dtype)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            alpha = max(0.0, min(1.0, 1.0 - gamma * h))
            t_back = alpha * t_curr
            if gamma > 0:
                eps = torch.randn(z.shape, dtype=dtype, device=device)
                z_back = alpha * z + (1.0 - alpha) * eps
            else:
                z_back = z
            t_batch = torch.full((n_samples,), t_back, dtype=dtype, device=device)
            net_out, _ = model(z_back, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z_back, t_batch, T_EPS)
            z = z_back + (t_next - t_back) * v_pred
    return z


# ── Corrected Exact SDE ──

def corrected_sde_sample(model, n_samples, n_steps, encoder_dim, seq_len,
                         g=0.5, time_schedule="uniform", device="cpu",
                         initial_noise=None):
    """Exact SDE with CORRECTED score: (t·x_pred - z) / ((1-t)²·σ²)."""
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = initial_noise.clone().to(device) if initial_noise is not None else \
            torch.randn(n_samples, seq_len, encoder_dim, device=device, dtype=dtype)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            _, x_pred = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            # Velocity
            denom_v = max(1 - t_curr, T_EPS)
            v = (x_pred - z) / denom_v
            # Corrected score
            denom_s = max(1 - t_curr, T_EPS)**2 * SIGMA**2
            score = (t_curr * x_pred - z) / denom_s
            # Drift + diffusion
            drift = v + (g**2 / 2) * score
            noise = torch.randn_like(z) if g > 0 else 0.0
            diffusion = g * math.sqrt(abs(h)) * noise if g > 0 else 0.0
            z = z + h * drift + diffusion
    return z


def compute_quality(model, z, device):
    """Decoder entropy and confidence."""
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        max_prob = probs.max(dim=-1).values.mean()
        tokens = logits.argmax(dim=-1)
        unique_per_seq = [len(torch.unique(tokens[i])) for i in range(tokens.shape[0])]
        diversity = float(np.mean(unique_per_seq))
    return {
        "entropy": float(entropy.item()),
        "max_prob": float(max_prob.item()),
        "diversity": diversity,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("C5 — Step-Budget Comparison With All Fixes")
    print("=" * 60)

    model, cfg, training_info = load_real_checkpoint()
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]
    vocab_size = cfg["vocab_size"]

    print(f"  Model: {cfg}")
    print(f"  Training: {training_info}")
    print(f"  Random entropy: ln({vocab_size}) = {np.log(vocab_size):.4f}")

    sampler_configs = [
        {"name": "ODE", "fn": lambda ns, sched, noise:
            ode_sample(model, N_SAMPLES, ns, encoder_dim, seq_len,
                       time_schedule=sched, device=DEVICE, initial_noise=noise)},
        {"name": "ELF-SDE γ=0.5", "fn": lambda ns, sched, noise:
            elf_sde_sample(model, N_SAMPLES, ns, encoder_dim, seq_len,
                           gamma=0.5, time_schedule=sched, device=DEVICE, initial_noise=noise)},
        {"name": "ELF-SDE γ=1.0", "fn": lambda ns, sched, noise:
            elf_sde_sample(model, N_SAMPLES, ns, encoder_dim, seq_len,
                           gamma=1.0, time_schedule=sched, device=DEVICE, initial_noise=noise)},
        {"name": "Corrected-SDE g=0.5", "fn": lambda ns, sched, noise:
            corrected_sde_sample(model, N_SAMPLES, ns, encoder_dim, seq_len,
                                 g=0.5, time_schedule=sched, device=DEVICE, initial_noise=noise)},
        {"name": "Corrected-SDE g=1.0", "fn": lambda ns, sched, noise:
            corrected_sde_sample(model, N_SAMPLES, ns, encoder_dim, seq_len,
                                 g=1.0, time_schedule=sched, device=DEVICE, initial_noise=noise)},
    ]

    all_results = {}
    n_runs = 3

    for sched in TIME_SCHEDULES:
        print(f"\n  ═══ {sched} ═══")
        sched_results = {}
        for scfg in sampler_configs:
            name = scfg["name"]
            print(f"    {name}:")
            step_results = {}
            for n_steps in STEP_BUDGETS:
                entropies, max_probs, divs, times = [], [], [], []
                for run in range(n_runs):
                    torch.manual_seed(42 + run)
                    noise = torch.randn(N_SAMPLES, seq_len, encoder_dim,
                                        dtype=torch.float32, device=DEVICE)
                    t0 = time.time()
                    z_final = scfg["fn"](n_steps, sched, noise)
                    elapsed = time.time() - t0
                    q = compute_quality(model, z_final, DEVICE)
                    entropies.append(q["entropy"])
                    max_probs.append(q["max_prob"])
                    divs.append(q["diversity"])
                    times.append(elapsed)

                step_results[str(n_steps)] = {
                    "entropy_mean": float(np.mean(entropies)),
                    "entropy_std": float(np.std(entropies)),
                    "max_prob_mean": float(np.mean(max_probs)),
                    "diversity_mean": float(np.mean(divs)),
                    "time_ms": float(np.mean(times)*1000),
                }
                print(f"      steps={n_steps:3d}: "
                      f"entropy={np.mean(entropies):.4f}±{np.std(entropies):.4f}, "
                      f"max_p={np.mean(max_probs):.4f}, "
                      f"div={np.mean(divs):.1f}, "
                      f"time={np.mean(times)*1000:.0f}ms")
            sched_results[name] = step_results
        all_results[sched] = sched_results

    # Interpret
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")

    pattern_holds = False
    corrected_beats_elf = False

    for sched in TIME_SCHEDULES:
        sd = all_results[sched]
        print(f"\n  ── {sched} ──")
        for sde_name in ["Corrected-SDE g=0.5", "ELF-SDE γ=0.5"]:
            ode_4 = sd["ODE"]["4"]["entropy_mean"]
            sde_4 = sd[sde_name]["4"]["entropy_mean"]
            ode_64 = sd["ODE"]["64"]["entropy_mean"]
            sde_64 = sd[sde_name]["64"]["entropy_mean"]
            adv_4 = ode_4 - sde_4  # positive = SDE better (lower entropy)
            adv_64 = ode_64 - sde_64

            print(f"    {sde_name}:")
            print(f"      4  steps: ODE={ode_4:.4f}, SDE={sde_4:.4f}, "
                  f"adv={adv_4:+.4f} {'(SDE ↓)' if adv_4 > 0 else '(ODE ↓)'}")
            print(f"      64 steps: ODE={ode_64:.4f}, SDE={sde_64:.4f}, "
                  f"adv={adv_64:+.4f}")

            if adv_4 > 0.001:
                pattern_holds = True

        # Check corrected vs ELF
        for steps_key in ["4", "8"]:
            corr = sd["Corrected-SDE g=0.5"][steps_key]["entropy_mean"]
            elf = sd["ELF-SDE γ=0.5"][steps_key]["entropy_mean"]
            if corr < elf - 0.001:
                corrected_beats_elf = True

    # Diversity check (C3 fix: should now be meaningful)
    for sched in TIME_SCHEDULES:
        sd = all_results[sched]
        print(f"\n  ── Diversity ({sched}) ──")
        for name in ["ODE", "ELF-SDE γ=0.5", "Corrected-SDE g=0.5"]:
            divs = [sd[name][str(s)]["diversity_mean"] for s in STEP_BUDGETS]
            print(f"    {name:25s}: {' → '.join(f'{d:.1f}' for d in divs)}")

    # Outcome
    print(f"\n{'=' * 60}")
    if pattern_holds and corrected_beats_elf:
        outcome = "strong_signal"
        print("🟢 OUTCOME 1 (STRONG): SDE beats ODE at low steps AND corrected > ELF approx.")
        print("   → Strong case for requesting Magus time.")
    elif pattern_holds:
        outcome = "modest_signal"
        print("🟡 OUTCOME 2 (MODEST): SDE advantage at low steps, but corrected ≈ ELF.")
        print("   → Reasonable to proceed (Findings-tier).")
    else:
        outcome = "no_signal"
        print("🟠 OUTCOME 3 (NO SIGNAL): No clear SDE advantage even on real data.")
        print("   → Budget a smaller first cluster run rather than full protocol.")
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {
        "ODE": "#4C72B0", "ELF-SDE γ=0.5": "#55A868", "ELF-SDE γ=1.0": "#8172B2",
        "Corrected-SDE g=0.5": "#C44E52", "Corrected-SDE g=1.0": "#DD8452",
    }
    markers = {"ODE": "o", "ELF-SDE γ=0.5": "s", "ELF-SDE γ=1.0": "D",
               "Corrected-SDE g=0.5": "^", "Corrected-SDE g=1.0": "v"}

    for col, sched in enumerate(TIME_SCHEDULES):
        sd = all_results[sched]
        # Entropy
        ax = axes[0][col]
        for name, data in sd.items():
            steps = sorted([int(s) for s in data.keys()])
            ents = [data[str(s)]["entropy_mean"] for s in steps]
            stds = [data[str(s)]["entropy_std"] for s in steps]
            ax.errorbar(steps, ents, yerr=stds, fmt=f"{markers[name]}-",
                        color=colors[name], lw=2, markersize=7, label=name,
                        capsize=3, alpha=0.85)
        ax.set_xlabel("Steps"); ax.set_ylabel("Entropy (↓ better)")
        ax.set_title(f"Quality — {sched}", fontsize=11)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

        # Diversity
        ax = axes[1][col]
        for name, data in sd.items():
            steps = sorted([int(s) for s in data.keys()])
            divs = [data[str(s)]["diversity_mean"] for s in steps]
            ax.plot(steps, divs, f"{markers[name]}-", color=colors[name],
                    lw=2, markersize=7, label=name, alpha=0.85)
        ax.set_xlabel("Steps"); ax.set_ylabel("Unique Tokens (↑ better)")
        ax.set_title(f"Diversity — {sched}", fontsize=11)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.suptitle("C5: Step-Budget Comparison (Real Language, Corrected SDE)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testC5_step_budget_real.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testC5_step_budget_real.png'}")

    with open(RESULTS_DIR / "testC5.json", "w") as f:
        json.dump({
            "results": all_results,
            "pattern_holds": pattern_holds,
            "corrected_beats_elf": corrected_beats_elf,
            "outcome": outcome,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
