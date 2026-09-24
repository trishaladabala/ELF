#!/usr/bin/env python3
"""Gate 1 verification: reproduce qualitative phase structure from the
analysis paper on ELF-B-owt SDE-32, 512 samples.

Produces three plots:
  (a) Mean margin rises monotonically over denoising phase
  (b) Self-conditioning disagreement peaks in a middle region
  (c) Per-token basin-entry times are visibly staggered

If the staggering cannot be reproduced, the project premise is wrong.
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.sampling.instrumented_sampler import instrumented_generate
from utils.sampling_utils import get_sampling_steps
from configs.config import SamplingConfig

# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-owt"
NUM_SAMPLES = 512
BATCH_SIZE = 8
NUM_STEPS = 32
SDE_GAMMA = 1.5
SC_CFG = 3.0
CFG = 1.0
TIME_SCHEDULE = "logit_normal"
SEED = 42
RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "gate1"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Load model ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  Gate 1: Instrumented OWT Generation")
    print("="*70)

    wrapper = ELFWrapper(CHECKPOINT, device=device)
    model = wrapper.model
    config = wrapper.config
    d_model = wrapper.d_model
    max_length = wrapper.max_length
    param_dtype = next(model.parameters()).dtype

    sampling_config = SamplingConfig(
        sampling_method="sde",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        sde_gamma=SDE_GAMMA,
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )

    # ── Generate with instrumentation ──────────────────────────────────
    print(f"\nGenerating {NUM_SAMPLES} samples with instrumentation...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Accumulators across batches
    all_margins = []        # list of [num_steps] arrays, each (batch, L)
    all_sc_deltas = []      # same
    all_top1_ids = []       # same
    all_final_ids = []      # (batch, L)
    all_entropies = []      # list of [num_steps] arrays
    all_t_values = []       # (num_steps,)

    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    gen_start = time.time()

    for batch_idx in range(num_batches):
        current_batch = min(BATCH_SIZE, NUM_SAMPLES - batch_idx * BATCH_SIZE)
        if current_batch <= 0:
            break

        # Fresh generator per batch for SDE noise reproducibility
        generator = torch.Generator(device=device).manual_seed(SEED + batch_idx)

        t_steps = get_sampling_steps(
            n_steps=NUM_STEPS, time_schedule=TIME_SCHEDULE,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=param_dtype,
        )

        z = torch.randn(
            (current_batch, max_length, d_model),
            dtype=param_dtype, device=device, generator=generator,
        ) * config.denoiser_noise_scale

        _, trajectory = instrumented_generate(
            model=model, wrapper=wrapper,
            generator=generator, z=z,
            t_steps=t_steps,
            cond_seq=None, cond_seq_mask=None,
            config=config, sampling_config=sampling_config,
            cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
            compute_diagnostics=True,
            compute_L_i=False,  # Skip L_i for Gate 1 (speed)
        )

        # Collect per-step data
        batch_margins = []
        batch_sc_deltas = []
        batch_top1_ids = []
        batch_entropies = []
        t_vals = []

        for step_diag in trajectory.steps:
            batch_margins.append(step_diag.margin.numpy())  # (B, L)
            if step_diag.sc_delta is not None:
                batch_sc_deltas.append(step_diag.sc_delta.numpy())
            batch_top1_ids.append(step_diag.top1_ids.numpy())
            batch_entropies.append(step_diag.entropy.numpy())
            t_vals.append(step_diag.t)

        all_margins.append(batch_margins)
        all_sc_deltas.append(batch_sc_deltas)
        all_top1_ids.append(batch_top1_ids)
        all_final_ids.append(trajectory.final_token_ids.numpy())
        all_entropies.append(batch_entropies)
        if not all_t_values:
            all_t_values = t_vals

        elapsed = time.time() - gen_start
        if (batch_idx + 1) % 8 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{num_batches} — "
                  f"{(batch_idx+1)*BATCH_SIZE}/{NUM_SAMPLES} — "
                  f"{elapsed:.1f}s")

    gen_time = time.time() - gen_start
    print(f"\nGeneration complete: {gen_time:.1f}s")

    # ── Aggregate across batches ───────────────────────────────────────
    num_steps_recorded = len(all_margins[0])
    print(f"Steps recorded: {num_steps_recorded}")

    # Per-step mean margin (across all samples and positions)
    mean_margins = []
    for s in range(num_steps_recorded):
        step_margins = np.concatenate([batch[s] for batch in all_margins], axis=0)
        mean_margins.append(float(np.mean(step_margins)))

    # Per-step mean SC delta
    mean_sc_deltas = []
    if all_sc_deltas and all_sc_deltas[0]:
        for s in range(len(all_sc_deltas[0])):
            step_deltas = np.concatenate(
                [batch[s] for batch in all_sc_deltas if s < len(batch)], axis=0)
            mean_sc_deltas.append(float(np.mean(step_deltas)))

    # Basin entry times: for each sample and position, the earliest step
    # where top1_id matches the final decoded token and stays matched
    print("\nComputing basin entry times...")
    all_entry_times = []
    for batch_idx in range(len(all_top1_ids)):
        batch_top1 = all_top1_ids[batch_idx]  # list of (B, L) per step
        batch_final = all_final_ids[batch_idx]  # (B, L)
        B, L = batch_final.shape

        for sample_idx in range(B):
            for pos in range(L):
                final_tok = batch_final[sample_idx, pos]
                entry_step = num_steps_recorded  # default: never entered
                # Scan from end backwards to find earliest sustained match
                for s in range(num_steps_recorded - 1, -1, -1):
                    if batch_top1[s][sample_idx, pos] == final_tok:
                        entry_step = s
                    else:
                        break
                all_entry_times.append(entry_step)

    entry_times = np.array(all_entry_times)

    # ── Peak VRAM ──────────────────────────────────────────────────────
    peak_vram_mb = 0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # ── Plot (a): Mean margin vs. step ─────────────────────────────────
    print("\nGenerating plots...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(num_steps_recorded), mean_margins, 'b-o', linewidth=2,
            markersize=4, label="Mean margin (top1 − top2)")
    ax.set_xlabel("Denoising Step", fontsize=14)
    ax.set_ylabel("Mean Margin", fontsize=14)
    ax.set_title("(a) Decoder Margin vs. Denoising Step — ELF-B-owt SDE-32",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plot_a = RESULTS_DIR / "plot_a_margin_vs_step.png"
    plt.savefig(plot_a, dpi=150)
    plt.close()
    print(f"  Saved {plot_a}")

    # Check monotonicity
    is_monotone = all(mean_margins[i] <= mean_margins[i+1]
                      for i in range(len(mean_margins) - 1))
    # Allow slight non-monotonicity (noise)
    mostly_monotone = sum(
        1 for i in range(len(mean_margins) - 1)
        if mean_margins[i] > mean_margins[i+1]
    ) <= 3  # at most 3 violations

    # ── Plot (b): SC disagreement vs. step ─────────────────────────────
    if mean_sc_deltas:
        fig, ax = plt.subplots(figsize=(10, 6))
        # SC deltas start from step 1 (need previous prediction)
        sc_steps = list(range(1, 1 + len(mean_sc_deltas)))
        ax.plot(sc_steps, mean_sc_deltas, 'r-o', linewidth=2, markersize=4,
                label="Mean ‖x̂ₛ − x̂ₛ₋₁‖")
        ax.set_xlabel("Denoising Step", fontsize=14)
        ax.set_ylabel("Self-Conditioning Delta", fontsize=14)
        ax.set_title("(b) Self-Conditioning Disagreement vs. Step — ELF-B-owt SDE-32",
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plot_b = RESULTS_DIR / "plot_b_sc_disagreement.png"
        plt.savefig(plot_b, dpi=150)
        plt.close()
        print(f"  Saved {plot_b}")

        # Check for middle peak
        peak_idx = np.argmax(mean_sc_deltas)
        n_sc = len(mean_sc_deltas)
        sc_peaks_in_middle = n_sc // 4 < peak_idx < 3 * n_sc // 4
    else:
        sc_peaks_in_middle = False
        print("  ⚠ No SC delta data (skipping plot b)")

    # ── Plot (c): Basin entry histogram ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(entry_times, bins=num_steps_recorded,
            range=(0, num_steps_recorded),
            color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel("Basin Entry Step", fontsize=14)
    ax.set_ylabel("Count (positions)", fontsize=14)
    ax.set_title("(c) Per-Token Basin Entry Times — ELF-B-owt SDE-32",
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_c = RESULTS_DIR / "plot_c_basin_entry.png"
    plt.savefig(plot_c, dpi=150)
    plt.close()
    print(f"  Saved {plot_c}")

    # Check staggering: entry times should have significant spread
    entry_std = float(np.std(entry_times))
    entry_iqr = float(np.percentile(entry_times, 75) -
                       np.percentile(entry_times, 25))
    is_staggered = entry_iqr >= 3  # at least 3 steps IQR

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  GATE 1 — VERIFICATION RESULTS")
    print("="*70)
    print(f"  Checkpoint:    {CHECKPOINT}")
    print(f"  Samples:       {NUM_SAMPLES}")
    print(f"  Steps:         {NUM_STEPS}, SDE γ={SDE_GAMMA}")
    print(f"  Gen. time:     {gen_time:.1f}s ({gen_time/60:.1f}m)")
    print(f"  Peak VRAM:     {peak_vram_mb:.0f} MB")
    print()
    print(f"  (a) Margin monotonicity:")
    print(f"      First margin:  {mean_margins[0]:.4f}")
    print(f"      Last margin:   {mean_margins[-1]:.4f}")
    print(f"      Monotone:      {'Yes' if is_monotone else 'Mostly' if mostly_monotone else 'NO'}")
    check_a = is_monotone or mostly_monotone
    print(f"      {'✓' if check_a else '✗'} PASS" if check_a else
          f"      ✗ FAIL — margin not monotonically rising")
    print()
    if mean_sc_deltas:
        print(f"  (b) SC disagreement peak:")
        print(f"      Peak at step: {peak_idx + 1} / {n_sc}")
        print(f"      Peak value:   {max(mean_sc_deltas):.4f}")
        print(f"      In middle:    {'Yes' if sc_peaks_in_middle else 'No'}")
        check_b = sc_peaks_in_middle
        print(f"      {'✓' if check_b else '⚠'} {'PASS' if check_b else 'MARGINAL'}")
    else:
        check_b = False
    print()
    print(f"  (c) Basin entry staggering:")
    print(f"      Entry time std:  {entry_std:.2f} steps")
    print(f"      Entry time IQR:  {entry_iqr:.2f} steps")
    print(f"      P25:             {np.percentile(entry_times, 25):.1f}")
    print(f"      P50 (median):    {np.percentile(entry_times, 50):.1f}")
    print(f"      P75:             {np.percentile(entry_times, 75):.1f}")
    print(f"      Staggered:       {'Yes' if is_staggered else 'NO'}")
    check_c = is_staggered
    print(f"      {'✓' if check_c else '✗'} {'PASS' if check_c else 'FAIL — project premise may be wrong'}")

    # Overall gate
    gate_pass = check_a and check_c  # b is informational
    print()
    if gate_pass:
        print("  ✓ GATE 1 PASSED — Phase structure reproduced")
    else:
        print("  ✗ GATE 1 FAILED — Cannot reproduce phase structure")
        if not check_c:
            print("    Basin entry is NOT staggered. "
                  "The project premise is wrong. STOP.")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "checkpoint": CHECKPOINT,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "seed": SEED,
        "gen_time_s": gen_time,
        "peak_vram_mb": peak_vram_mb,
        "mean_margins": mean_margins,
        "mean_sc_deltas": mean_sc_deltas,
        "entry_time_stats": {
            "std": entry_std,
            "iqr": entry_iqr,
            "p25": float(np.percentile(entry_times, 25)),
            "p50": float(np.percentile(entry_times, 50)),
            "p75": float(np.percentile(entry_times, 75)),
        },
        "checks": {
            "margin_monotone": bool(check_a),
            "sc_peak_middle": bool(check_b),
            "entry_staggered": bool(check_c),
        },
        "gate_1_pass": bool(gate_pass),
    }
    results_file = RESULTS_DIR / "gate1_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")
    print(f"  Plots saved to {RESULTS_DIR}/")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
