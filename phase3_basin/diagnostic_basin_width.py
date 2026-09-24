#!/usr/bin/env python3
"""Phase 0 — Decoder-Basin Width Diagnostic.

Measures the decoder-basin width of the ELF-B checkpoint at each ODE
sampling timestep. For each step, the model's clean-state prediction
(x_pred) is extracted, decoded to tokens, then perturbed with Gaussian
noise at multiple scales and re-decoded. The token flip rate, decoder
margin, and correct-token probability are recorded per timestep.

This is a pure-inference diagnostic — no training, no gradient computation.
"""

import json, math, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from modules.model import ELF_models
from utils.sampling_utils import (
    _ode_step, get_sampling_steps, restore_cond, net_out_to_v_x,
    _forward_sample,
)
from configs.config import Config

# ── Configuration ──────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent / "results" / "diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 128           # Number of sequences (keep moderate for diagnostic)
BS = 4            # Batch size
STEPS = 32        # ODE sampling steps
L = 1024          # Sequence length
ENC = 512         # Encoder embedding dim (T5-small)
SC_CFG = 3.0      # Self-cond CFG scale
NOISE_SCALE = 2.0

# Perturbation scales to test
SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5]

# ── Config ─────────────────────────────────────────────────────────────
cfg = Config()
cfg.denoiser_p_mean = -1.5
cfg.denoiser_p_std = 0.8
cfg.denoiser_noise_scale = NOISE_SCALE
cfg.t_eps = 0.05
cfg.num_self_cond_cfg_tokens = 4
cfg.self_cond_prob = 0.5

COMMON = dict(
    text_encoder_dim=ENC, max_length=L, bottleneck_dim=128,
    num_time_tokens=4, num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4, vocab_size=32100,
)


# ── Model loading ──────────────────────────────────────────────────────
def load_model():
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "embedded-language-flows/ELF-B-owt-torch"
    files = list_repo_files(repo)
    ckpts = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    path = hf_hub_download(repo, ckpts[0])
    model = ELF_models["ELF-B"](**COMMON)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_params1") or ckpt.get("params") or ckpt
    model.load_state_dict(sd, strict=False)
    return model.to(DEV, torch.float32).eval()


# ── Factored decoder (replicates model's decoder head) ─────────────────
def decode_z_to_logits(model, z):
    """Decode z (in encoder-dim space) to vocab logits by passing it through
    the model with decoder_step_active=True.
    """
    batch_size = z.shape[0]
    t_final = torch.ones((batch_size,), dtype=z.dtype, device=z.device)
    sc_batch = torch.full((batch_size,), SC_CFG, dtype=z.dtype, device=z.device)
    
    # Self-conditioning concatenation if needed
    if cfg.self_cond_prob > 0:
        z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
    else:
        z_input = z
        
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
            _, decoder_logits = model(
                z_input, t_final, deterministic=True,
                self_cond_cfg_scale=sc_batch,
                decoder_step_active=True,
            )
    return decoder_logits


def compute_basin_metrics(model, x_pred, sigmas):
    """Compute basin-width metrics for a batch of x_pred embeddings.

    For each perturbation scale σ in `sigmas`, perturbs x_pred with
    Gaussian noise δ ~ N(0, I) and measures how the decoded tokens change.

    Returns a dict with per-sigma metrics.
    """
    # Decode the clean x_pred
    logits_clean = decode_z_to_logits(model, x_pred)
    probs_clean = F.softmax(logits_clean.float(), dim=-1)
    tokens_clean = logits_clean.argmax(dim=-1)  # (B, S)

    # Correct-token probability (clean)
    correct_prob_clean = probs_clean.gather(
        -1, tokens_clean.unsqueeze(-1)
    ).squeeze(-1)  # (B, S)

    # Margin (clean): logit_top1 - logit_top2
    top2_logits = logits_clean.float().topk(2, dim=-1).values  # (B, S, 2)
    margin_clean = (top2_logits[:, :, 0] - top2_logits[:, :, 1])  # (B, S)

    results = {
        "clean": {
            "mean_margin": margin_clean.mean().item(),
            "median_margin": margin_clean.median().item(),
            "mean_correct_prob": correct_prob_clean.mean().item(),
        }
    }

    for sigma in sigmas:
        # Perturb x_pred in encoder-dim space
        delta = torch.randn_like(x_pred)
        x_perturbed = x_pred + sigma * delta

        # Decode perturbed
        logits_pert = decode_z_to_logits(model, x_perturbed)
        tokens_pert = logits_pert.argmax(dim=-1)  # (B, S)
        probs_pert = F.softmax(logits_pert.float(), dim=-1)

        # Token flip rate: fraction of positions where decoded token changed
        flips = (tokens_clean != tokens_pert).float()
        flip_rate = flips.mean().item()

        # Correct-token probability after perturbation (using clean tokens as "correct")
        correct_prob_pert = probs_pert.gather(
            -1, tokens_clean.unsqueeze(-1)
        ).squeeze(-1)

        # Margin after perturbation
        top2_pert = logits_pert.float().topk(2, dim=-1).values
        margin_pert = (top2_pert[:, :, 0] - top2_pert[:, :, 1])

        # KL divergence: KL(clean || perturbed)
        kl = F.kl_div(
            F.log_softmax(logits_pert.float(), dim=-1),
            probs_clean,
            reduction="batchmean",
        ).item()

        results[f"sigma_{sigma}"] = {
            "flip_rate": flip_rate,
            "mean_margin": margin_pert.mean().item(),
            "mean_correct_prob": correct_prob_pert.mean().item(),
            "mean_kl_divergence": kl,
            "prob_drop": (correct_prob_clean - correct_prob_pert).mean().item(),
        }

    return results


# ── Main diagnostic ────────────────────────────────────────────────────
def run_diagnostic():
    print("=" * 70)
    print("Phase 0 — Decoder-Basin Width Diagnostic")
    print("=" * 70)
    t0 = time.time()

    print("\n[1/3] Loading ELF-B checkpoint...")
    model = load_model()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Pre-generate noise (shared across all experiments for reproducibility)
    torch.manual_seed(42)
    all_noise = torch.randn(N, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE

    # Per-timestep basin metrics
    per_step_metrics = []
    n_batches = (N + BS - 1) // BS

    print(f"\n[2/3] Running ODE sampling with basin probing at each step...")
    print(f"  N={N}, L={L}, steps={STEPS}, batch_size={BS}")
    print(f"  Perturbation σ: {SIGMAS}")

    # Generate time steps (shared)
    steps = get_sampling_steps(
        STEPS, time_schedule="logit_normal",
        P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
        device=DEV, dtype=torch.float32,
    )
    step_values = [steps[i].item() for i in range(len(steps))]

    # For each batch, run ODE sampling and probe x_pred at each step
    # We accumulate per-step metrics across batches
    step_metrics_accum = {i: [] for i in range(STEPS)}

    for bi in tqdm(range(n_batches), desc="Batches"):
        bs = min(BS, N - bi * BS)
        noise = all_noise[bi * BS: bi * BS + bs].clone()

        z = noise.clone()
        cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
        mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
        x_prev = None

        with torch.no_grad():
            for i in range(STEPS):
                t_val = step_values[i]
                t_next = step_values[i + 1]

                # Run one ODE step
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
                    z, x_pred = _ode_step(
                        model, z, t_val, t_next, x_prev,
                        cfg, 1.0, SC_CFG, cond, mask,
                    )
                x_prev = x_pred

                # Probe basin width at this timestep
                # x_pred is the model's clean-state prediction at this step
                # It lives in encoder-dim space (B, L, 512)
                basin_metrics = compute_basin_metrics(model, x_pred, SIGMAS)
                basin_metrics["step_index"] = i
                basin_metrics["t"] = t_val
                basin_metrics["t_next"] = t_next
                step_metrics_accum[i].append(basin_metrics)

    # Aggregate metrics across batches
    print("\n[3/3] Aggregating results...")
    aggregated = []
    for step_i in range(STEPS):
        batch_results = step_metrics_accum[step_i]
        n_batches_actual = len(batch_results)

        agg = {
            "step_index": step_i,
            "t": batch_results[0]["t"],
            "t_next": batch_results[0]["t_next"],
            "clean": {
                "mean_margin": np.mean([r["clean"]["mean_margin"] for r in batch_results]),
                "median_margin": np.mean([r["clean"]["median_margin"] for r in batch_results]),
                "mean_correct_prob": np.mean([r["clean"]["mean_correct_prob"] for r in batch_results]),
            },
        }

        for sigma in SIGMAS:
            key = f"sigma_{sigma}"
            agg[key] = {
                "flip_rate": np.mean([r[key]["flip_rate"] for r in batch_results]),
                "mean_margin": np.mean([r[key]["mean_margin"] for r in batch_results]),
                "mean_correct_prob": np.mean([r[key]["mean_correct_prob"] for r in batch_results]),
                "mean_kl_divergence": np.mean([r[key]["mean_kl_divergence"] for r in batch_results]),
                "prob_drop": np.mean([r[key]["prob_drop"] for r in batch_results]),
            }

        aggregated.append(agg)

    # Save results
    results = {
        "config": {
            "N": N, "L": L, "steps": STEPS, "batch_size": BS,
            "sigmas": SIGMAS, "noise_scale": NOISE_SCALE,
            "sc_cfg": SC_CFG, "model": "ELF-B (105M)",
        },
        "step_values": step_values,
        "per_step": aggregated,
        "total_time_s": time.time() - t0,
    }

    out_path = OUT / "basin_width_by_timestep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Step':>4} {'t':>6} | {'Margin':>8} {'P(correct)':>10} | ", end="")
    for s in SIGMAS:
        print(f"Flip@{s:<5}", end=" ")
    print()
    print("-" * 90)

    for a in aggregated:
        print(f"{a['step_index']:>4} {a['t']:>6.3f} | ", end="")
        print(f"{a['clean']['mean_margin']:>8.3f} {a['clean']['mean_correct_prob']:>10.4f} | ", end="")
        for s in SIGMAS:
            key = f"sigma_{s}"
            print(f"{a[key]['flip_rate']:>8.3f} ", end="")
        print()

    # Generate diagnostic report
    generate_report(results)
    generate_plots(results)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    return results


def generate_report(results):
    """Generate a markdown diagnostic report with go/no-go recommendation."""
    per_step = results["per_step"]

    # Find intermediate timestep range (t ∈ [0.3, 0.8])
    intermediate = [s for s in per_step if 0.3 <= s["t"] <= 0.8]
    late = [s for s in per_step if s["t"] > 0.8]
    early = [s for s in per_step if s["t"] < 0.3]

    # Key metric: flip rate at σ=0.1 in the intermediate range
    if intermediate:
        flip_01_inter = np.mean([s["sigma_0.1"]["flip_rate"] for s in intermediate])
        flip_005_inter = np.mean([s["sigma_0.05"]["flip_rate"] for s in intermediate])
        margin_inter = np.mean([s["clean"]["mean_margin"] for s in intermediate])
    else:
        flip_01_inter = flip_005_inter = margin_inter = float("nan")

    if late:
        flip_01_late = np.mean([s["sigma_0.1"]["flip_rate"] for s in late])
        margin_late = np.mean([s["clean"]["mean_margin"] for s in late])
    else:
        flip_01_late = margin_late = float("nan")

    # Go/No-Go decision
    go = flip_01_inter > 0.10  # >10% flip rate at σ=0.1 in intermediate range

    report = f"""# Phase 0 — Decoder-Basin Width Diagnostic Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Model:** ELF-B (105M params)
**Config:** N={results['config']['N']}, L={results['config']['L']}, {results['config']['steps']} ODE steps

## Summary

| Region | t range | Mean Margin | Flip Rate (σ=0.05) | Flip Rate (σ=0.1) |
|---|---|---|---|---|
| Early | t < 0.3 | {np.mean([s['clean']['mean_margin'] for s in early]) if early else float('nan'):.3f} | {np.mean([s['sigma_0.05']['flip_rate'] for s in early]) if early else float('nan'):.3f} | {np.mean([s['sigma_0.1']['flip_rate'] for s in early]) if early else float('nan'):.3f} |
| Intermediate | 0.3 ≤ t ≤ 0.8 | {margin_inter:.3f} | {flip_005_inter:.3f} | {flip_01_inter:.3f} |
| Late | t > 0.8 | {margin_late:.3f} | {np.mean([s['sigma_0.05']['flip_rate'] for s in late]) if late else float('nan'):.3f} | {flip_01_late:.3f} |

## Go/No-Go

**Criterion:** Token flip rate > 10% at σ=0.1 in the intermediate range (t ∈ [0.3, 0.8]).

**Intermediate flip rate (σ=0.1): {flip_01_inter:.1%}**

**Verdict: {'✅ GO — Narrow basins confirmed at intermediate timesteps. Phase 1 (fine-tuning) is justified.' if go else '❌ NO-GO — Basins are already wide. The problem this method targets does not appear at ELF-B scale.'}**

## Interpretation

{'The high token flip rate at intermediate timesteps indicates that the model s clean-state predictions lie near token-decision boundaries. Small perturbations in the embedding space cause different tokens to be decoded, confirming narrow decoder basins. Basin-widening training should target these intermediate timesteps.' if go else 'The low token flip rate suggests that ELF-B s decoder basins are already quite wide at intermediate timesteps. This could mean the flow-matching training has already learned to produce predictions well within decoder basins, or that the factored decoder head is inherently robust.'}

## Per-Step Detail

See `basin_width_by_timestep.json` for full per-step metrics.
"""

    report_path = OUT / "diagnostic_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")


def generate_plots(results):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
        return

    per_step = results["per_step"]
    t_vals = [s["t"] for s in per_step]

    # ── Plot 1: Token Flip Rate vs Timestep ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Flip rate
    ax = axes[0]
    for sigma in SIGMAS:
        key = f"sigma_{sigma}"
        flip_rates = [s[key]["flip_rate"] for s in per_step]
        ax.plot(t_vals, flip_rates, "o-", label=f"σ={sigma}", markersize=3)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Token Flip Rate")
    ax.set_title("Token Flip Rate vs Timestep")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Panel 2: Decoder margin
    ax = axes[1]
    margins = [s["clean"]["mean_margin"] for s in per_step]
    ax.plot(t_vals, margins, "s-", color="darkblue", markersize=4)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Mean Decoder Margin (logit gap)")
    ax.set_title("Decoder Margin vs Timestep")
    ax.grid(True, alpha=0.3)

    # Panel 3: Correct-token probability
    ax = axes[2]
    clean_probs = [s["clean"]["mean_correct_prob"] for s in per_step]
    ax.plot(t_vals, clean_probs, "^-", color="green", label="Clean", markersize=4)
    for sigma in [0.05, 0.1, 0.5]:
        key = f"sigma_{sigma}"
        pert_probs = [s[key]["mean_correct_prob"] for s in per_step]
        ax.plot(t_vals, pert_probs, "o-", label=f"σ={sigma}", markersize=3)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("P(correct token)")
    ax.set_title("Correct-Token Probability vs Timestep")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUT / "basin_width_profile.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {plot_path}")

    # ── Plot 2: Heatmap — Flip rate vs (timestep, σ) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    flip_matrix = np.array([
        [s[f"sigma_{sigma}"]["flip_rate"] for s in per_step]
        for sigma in SIGMAS
    ])
    im = ax.imshow(
        flip_matrix, aspect="auto", cmap="YlOrRd",
        extent=[t_vals[0], t_vals[-1], len(SIGMAS) - 0.5, -0.5],
        vmin=0, vmax=1,
    )
    ax.set_yticks(range(len(SIGMAS)))
    ax.set_yticklabels([str(s) for s in SIGMAS])
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Perturbation σ")
    ax.set_title("Token Flip Rate Heatmap")
    plt.colorbar(im, ax=ax, label="Flip Rate")
    plt.tight_layout()
    heatmap_path = OUT / "token_flip_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap: {heatmap_path}")


if __name__ == "__main__":
    run_diagnostic()
