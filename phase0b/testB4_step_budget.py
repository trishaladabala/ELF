#!/usr/bin/env python3
"""B4 — Toy-Scale Step-Budget Comparison.

The actual pilot result: evaluates all four samplers across a step-budget sweep.

Samplers:
  1. ODE Euler (deterministic)
  2. ELF's SDE approximation (Algorithm 6) at multiple γ values
  3. Exact probability-flow SDE at multiple g values

Time schedules: uniform and logit-normal

Step budgets: 4, 8, 16, 32, 64

Quality proxy: reconstruction loss — average L2 distance between the sampled
final latent z and the "ideal" x₁ (decoded → re-encoded through the model's
decode head). Since the toy model trained on random data, we measure how well
the sampler can produce outputs the decoder head assigns high confidence to.
We use decoder entropy as a proxy: lower entropy = more confident = better quality.
"""

import json
import sys
import time
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

# Import samplers from B2 and B3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from testB2_samplers import ode_euler_sample, elf_sde_sample, decode_to_tokens
from testB3_exact_sde import exact_sde_sample

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

T_EPS = 0.05
N_SAMPLES = 32
STEP_BUDGETS = [4, 8, 16, 32, 64]
TIME_SCHEDULES = ["uniform", "logit_normal"]


def load_checkpoint():
    ckpt_path = RESULTS_DIR / "mini_elf_checkpoint.pt"
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


def compute_quality_proxy(model, z, device):
    """Compute quality proxy: decoder confidence (negative entropy).

    Lower entropy = more confident predictions = better quality.
    Also returns mean max probability (another proxy).
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))

        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        max_prob = probs.max(dim=-1).values.mean()

        # Token diversity within sequences
        tokens = logits.argmax(dim=-1)
        unique_per_seq = [len(torch.unique(tokens[i])) for i in range(tokens.shape[0])]
        diversity = float(np.mean(unique_per_seq))

    return {
        "entropy": float(entropy.item()),
        "max_prob": float(max_prob.item()),
        "diversity": diversity,
    }


def run_sweep(model, cfg, n_runs=3):
    """Run the full step-budget sweep across all samplers."""
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]

    # Sampler configurations
    sampler_configs = [
        {"name": "ODE", "type": "ode"},
        {"name": "ELF-SDE γ=0.5", "type": "elf_sde", "gamma": 0.5},
        {"name": "ELF-SDE γ=1.0", "type": "elf_sde", "gamma": 1.0},
        {"name": "Exact-SDE g=0.5", "type": "exact_sde", "g": 0.5},
        {"name": "Exact-SDE g=1.0", "type": "exact_sde", "g": 1.0},
    ]

    all_results = {}

    for sched in TIME_SCHEDULES:
        print(f"\n  ═══ Time schedule: {sched} ═══")
        sched_results = {}

        for sampler_cfg in sampler_configs:
            name = sampler_cfg["name"]
            print(f"\n    Sampler: {name}")
            step_results = {}

            for n_steps in STEP_BUDGETS:
                entropies, max_probs, diversities, times = [], [], [], []

                for run in range(n_runs):
                    torch.manual_seed(42 + run)
                    noise = torch.randn(N_SAMPLES, seq_len, encoder_dim,
                                        dtype=torch.float32, device=DEVICE)

                    t0 = time.time()

                    if sampler_cfg["type"] == "ode":
                        z_final, _ = ode_euler_sample(
                            model, N_SAMPLES, n_steps, encoder_dim, seq_len,
                            time_schedule=sched, device=DEVICE, initial_noise=noise,
                        )
                    elif sampler_cfg["type"] == "elf_sde":
                        z_final, _ = elf_sde_sample(
                            model, N_SAMPLES, n_steps, encoder_dim, seq_len,
                            gamma=sampler_cfg["gamma"],
                            time_schedule=sched, device=DEVICE, initial_noise=noise,
                        )
                    elif sampler_cfg["type"] == "exact_sde":
                        z_final, _ = exact_sde_sample(
                            model, N_SAMPLES, n_steps, encoder_dim, seq_len,
                            g=sampler_cfg["g"],
                            time_schedule=sched, device=DEVICE, initial_noise=noise,
                        )

                    elapsed = time.time() - t0
                    quality = compute_quality_proxy(model, z_final, DEVICE)

                    entropies.append(quality["entropy"])
                    max_probs.append(quality["max_prob"])
                    diversities.append(quality["diversity"])
                    times.append(elapsed)

                step_results[str(n_steps)] = {
                    "entropy_mean": float(np.mean(entropies)),
                    "entropy_std": float(np.std(entropies)),
                    "max_prob_mean": float(np.mean(max_probs)),
                    "max_prob_std": float(np.std(max_probs)),
                    "diversity_mean": float(np.mean(diversities)),
                    "time_mean_s": float(np.mean(times)),
                }

                print(f"      steps={n_steps:3d}: entropy={np.mean(entropies):.4f}±{np.std(entropies):.4f}, "
                      f"max_p={np.mean(max_probs):.4f}, div={np.mean(diversities):.1f}, "
                      f"time={np.mean(times)*1000:.0f}ms")

            sched_results[name] = step_results

        all_results[sched] = sched_results

    return all_results


def plot_results(results):
    """Generate step-budget comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors = {
        "ODE": "#4C72B0",
        "ELF-SDE γ=0.5": "#55A868",
        "ELF-SDE γ=1.0": "#8172B2",
        "Exact-SDE g=0.5": "#C44E52",
        "Exact-SDE g=1.0": "#DD8452",
    }
    markers = {
        "ODE": "o",
        "ELF-SDE γ=0.5": "s",
        "ELF-SDE γ=1.0": "D",
        "Exact-SDE g=0.5": "^",
        "Exact-SDE g=1.0": "v",
    }

    for col_idx, sched in enumerate(TIME_SCHEDULES):
        sched_data = results[sched]

        # Entropy (lower = better quality)
        ax = axes[0][col_idx]
        for name, step_data in sched_data.items():
            steps = sorted([int(s) for s in step_data.keys()])
            entropies = [step_data[str(s)]["entropy_mean"] for s in steps]
            stds = [step_data[str(s)]["entropy_std"] for s in steps]
            ax.errorbar(steps, entropies, yerr=stds, fmt=f"{markers[name]}-",
                        color=colors[name], lw=2, markersize=7, label=name,
                        capsize=3, alpha=0.85)
        ax.set_xlabel("Sampling Steps")
        ax.set_ylabel("Decoder Entropy (↓ better)")
        ax.set_title(f"Quality vs Steps — {sched}", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

        # Max probability (higher = better)
        ax = axes[1][col_idx]
        for name, step_data in sched_data.items():
            steps = sorted([int(s) for s in step_data.keys()])
            max_probs = [step_data[str(s)]["max_prob_mean"] for s in steps]
            ax.plot(steps, max_probs, f"{markers[name]}-",
                    color=colors[name], lw=2, markersize=7, label=name, alpha=0.85)
        ax.set_xlabel("Sampling Steps")
        ax.set_ylabel("Mean Max Probability (↑ better)")
        ax.set_title(f"Decoder Confidence vs Steps — {sched}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.suptitle("B4: Toy-Scale Step-Budget Comparison\n(all samplers, both schedules)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testB4_step_budget.png", dpi=150, bbox_inches="tight")
    plt.close()


def interpret_results(results):
    """Analyze whether the expected pattern holds."""
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    for sched in TIME_SCHEDULES:
        print(f"\n  ── {sched} schedule ──")
        sched_data = results[sched]

        # At low step count (4 steps): compare SDE vs ODE
        ode_4 = sched_data["ODE"]["4"]["entropy_mean"]
        ode_64 = sched_data["ODE"]["64"]["entropy_mean"]

        for sde_name in ["ELF-SDE γ=0.5", "Exact-SDE g=0.5"]:
            sde_4 = sched_data[sde_name]["4"]["entropy_mean"]
            sde_64 = sched_data[sde_name]["64"]["entropy_mean"]

            # Lower entropy = better
            advantage_low = ode_4 - sde_4  # positive = SDE better
            advantage_high = ode_64 - sde_64

            print(f"    {sde_name}:")
            print(f"      4 steps: ODE={ode_4:.4f}, SDE={sde_4:.4f}, "
                  f"advantage={advantage_low:+.4f} {'(SDE better)' if advantage_low > 0 else '(ODE better)'}")
            print(f"      64 steps: ODE={ode_64:.4f}, SDE={sde_64:.4f}, "
                  f"advantage={advantage_high:+.4f} {'(SDE better)' if advantage_high > 0 else '(ODE better)'}")
            print(f"      Gap narrows: {'✅ Yes' if abs(advantage_low) > abs(advantage_high) else '❌ No'}")

    # Overall pattern assessment
    print("\n  ── Overall Assessment ──")
    pattern_holds = False
    exact_beats_approx = False

    # Check for the expected pattern: SDE advantage at low steps
    for sched in TIME_SCHEDULES:
        sched_data = results[sched]
        for sde_name in ["Exact-SDE g=0.5", "ELF-SDE γ=0.5"]:
            adv_4 = sched_data["ODE"]["4"]["entropy_mean"] - sched_data[sde_name]["4"]["entropy_mean"]
            adv_64 = sched_data["ODE"]["64"]["entropy_mean"] - sched_data[sde_name]["64"]["entropy_mean"]
            if adv_4 > 0 and adv_4 > adv_64:
                pattern_holds = True

    for sched in TIME_SCHEDULES:
        exact_4 = results[sched]["Exact-SDE g=0.5"]["4"]["entropy_mean"]
        elf_4 = results[sched]["ELF-SDE γ=0.5"]["4"]["entropy_mean"]
        if exact_4 < elf_4:  # Lower entropy = better
            exact_beats_approx = True

    return pattern_holds, exact_beats_approx


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("B4 — Toy-Scale Step-Budget Comparison")
    print("=" * 60)

    model, cfg = load_checkpoint()
    print(f"  Model loaded: {cfg}")
    print(f"  Step budgets: {STEP_BUDGETS}")
    print(f"  Schedules: {TIME_SCHEDULES}")
    print(f"  Samples per config: {N_SAMPLES}")

    results = run_sweep(model, cfg, n_runs=3)

    # Plot
    plot_results(results)
    print(f"\n  Saved plot → {RESULTS_DIR / 'testB4_step_budget.png'}")

    # Interpret
    pattern_holds, exact_beats_approx = interpret_results(results)

    # Outcome
    print(f"\n{'=' * 60}")
    if pattern_holds and exact_beats_approx:
        print("🟢 OUTCOME 1: Pattern holds AND exact SDE beats ELF approximation.")
        print("   → Strong signal to proceed to Magus with this project.")
        outcome = "strong_proceed"
    elif pattern_holds:
        print("🟡 OUTCOME 2: Pattern holds, but exact SDE ≈ ELF approximation.")
        print("   → Proceeding is reasonable (Findings-tier contribution).")
        outcome = "reasonable_proceed"
    else:
        print("🟠 OUTCOME 3: Pattern doesn't hold at toy scale.")
        print("   → Not a definitive kill, but budget a smaller first cluster run.")
        outcome = "cautious"

    print(f"{'=' * 60}")

    # Save
    with open(RESULTS_DIR / "testB4.json", "w") as f:
        json.dump({
            "results": results,
            "pattern_holds": pattern_holds,
            "exact_beats_approx": exact_beats_approx,
            "outcome": outcome,
        }, f, indent=2)

    print(f"  Saved → {RESULTS_DIR / 'testB4.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
