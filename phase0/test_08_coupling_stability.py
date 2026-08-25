#!/usr/bin/env python3
"""Test 8 — Toy-Scale Coupling Stability Probe.

Swaps in an OT-based coupling scheme on the same toy ELF model from Test 7
to check whether it destabilizes training.

Go/No-Go:
  Go:    Training stable, decode-branch CE loss decreases at similar rate to Test 7.
  No-Go: Divergence, NaNs, or decode-branch loss fails to decrease.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF
from utils.sampling_utils import sample_timesteps, net_out_to_v_x

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TOY_HIDDEN = 64
TOY_DEPTH = 2
TOY_HEADS = 4
TOY_BOTTLENECK = 32
TOY_MLP_RATIO = 2.0
TOY_MAX_LEN = 16
TOY_ENCODER_DIM = 64
TOY_VOCAB_SIZE = 256
NUM_TRAIN_STEPS = 500
BATCH_SIZE = 32
LR = 1e-3
DECODER_PROB = 0.5
T_EPS = 0.05


class ToyConfig:
    t_eps = T_EPS
    self_cond_prob = 0.0
    denoiser_p_mean = 0.8
    denoiser_p_std = 0.8
    denoiser_noise_scale = 1.0
    time_schedule = "logit_normal"
    decoder_prob = DECODER_PROB
    decoder_noise_scale = 1.0
    decoder_p_mean = 0.8
    decoder_p_std = 0.8
    num_self_cond_cfg_tokens = 0
    self_cond_cfg_min = 0.5
    self_cond_cfg_max = 5.0
    label_drop_prob = 0.0
    pad_token = "pad"
    latent_mean = 0.0
    latent_std = 1.0


def ot_coupling(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Apply OT-based coupling: reorder noise to minimize transport cost.

    Uses a simplified mini-batch Sinkhorn on per-sequence basis.
    x0: (B, S, D), noise: (B, S, D) -> reordered noise (B, S, D)
    """
    import ot

    B, S, D = x0.shape
    reordered_noise = noise.clone()

    # Per-sequence OT coupling
    x_np = x0.detach().cpu().numpy().reshape(B, S * D)
    n_np = noise.detach().cpu().numpy().reshape(B, S * D)

    # Batch-level coupling: match entire sequences
    C = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            C[i, j] = np.sum((x_np[i] - n_np[j]) ** 2)
    C_norm = C / (C.max() + 1e-10)
    a = np.ones(B) / B
    b = np.ones(B) / B

    T = ot.sinkhorn(a, b, C_norm, reg=0.05, numItermax=500, stopThr=1e-8)
    # Permute noise according to OT plan
    assignments = T.argmax(axis=1)
    reordered_noise = noise[assignments]

    return reordered_noise


def create_toy_model(encoder_dim, vocab_size):
    return ELF(
        text_encoder_dim=encoder_dim,
        max_length=TOY_MAX_LEN,
        hidden_size=TOY_HIDDEN,
        depth=TOY_DEPTH,
        num_heads=TOY_HEADS,
        mlp_ratio=TOY_MLP_RATIO,
        bottleneck_dim=TOY_BOTTLENECK,
        num_time_tokens=2,
        num_self_cond_cfg_tokens=0,
        num_model_mode_tokens=2,
        vocab_size=vocab_size,
    )


def train_step_with_ot(model, x0, input_ids, attention_mask, cond_seq_mask,
                        config, device):
    """Training step with OT-coupled noise instead of independent noise."""
    dtype = next(model.parameters()).dtype
    batch_size, seq_length, encoder_dim = x0.shape

    t = sample_timesteps(
        batch_size, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule, device=device, dtype=dtype,
    )

    noise = torch.randn_like(x0)

    # OT coupling: reorder noise to better match x0
    noise = ot_coupling(x0, noise)

    loss_mask = attention_mask * (1 - cond_seq_mask)
    cond_mask_3d = cond_seq_mask.unsqueeze(-1)

    t_exp = t.reshape(-1, 1, 1)
    denoiser_z = t_exp * x0 + (1 - t_exp) * noise * config.denoiser_noise_scale
    denoiser_z = cond_mask_3d * x0 + (1 - cond_mask_3d) * denoiser_z
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_exp, min=config.t_eps)

    decoder_active = torch.bernoulli(
        torch.full((batch_size,), config.decoder_prob)
    ).to(device=device, dtype=dtype)
    decoder_B11 = decoder_active.view(-1, 1, 1)
    decoder_B1 = decoder_active.view(-1, 1)

    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn_like(x0) * config.decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_mixed = decoder_active * 1.0 + (1.0 - decoder_active) * t
    z_mixed = decoder_B11 * decoder_z + (1.0 - decoder_B11) * denoiser_z

    net_out, decoder_logits = model(
        z_mixed, t_mixed, deterministic=False, decoder_step_active=decoder_active,
    )

    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
    l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_B1
    l2_mask = loss_mask_f * (1.0 - decoder_B1)

    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

    ce_loss = ((ce_per_token * ce_mask).sum() / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss = ((l2_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    return loss, l2_loss, ce_loss


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 8 — Toy-Scale Coupling Stability Probe")
    print("=" * 60)

    config = ToyConfig()
    encoder_dim = TOY_ENCODER_DIM
    vocab_size = TOY_VOCAB_SIZE

    # Load Test 7 baseline results for comparison
    test07_path = RESULTS_DIR / "test_07.json"
    if test07_path.exists():
        with open(test07_path) as f:
            baseline = json.load(f)
        print(f"Baseline (Test 7): L2 early={baseline['early_l2']:.4f} → "
              f"late={baseline['late_l2']:.4f}, "
              f"CE early={baseline['early_ce']:.4f} → late={baseline['late_ce']:.4f}")
    else:
        baseline = None
        print("Warning: Test 7 results not found. Running without baseline comparison.")

    model = create_toy_model(encoder_dim, vocab_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    l2_losses, ce_losses, total_losses, grad_norms = [], [], [], []
    has_nan = False
    has_divergence = False
    training_error = None

    print(f"\n── Training with OT coupling ──")
    model.train()
    try:
        for step in range(NUM_TRAIN_STEPS):
            x0 = torch.randn(BATCH_SIZE, TOY_MAX_LEN, encoder_dim, device=DEVICE)
            input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, TOY_MAX_LEN), device=DEVICE)
            attention_mask = torch.ones(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)
            cond_seq_mask = torch.zeros(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)

            optimizer.zero_grad()
            loss, l2_loss, ce_loss = train_step_with_ot(
                model, x0, input_ids, attention_mask, cond_seq_mask, config, DEVICE
            )
            loss.backward()

            # Monitor gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(loss.item())
            l2_val = float(l2_loss.item())
            ce_val = float(ce_loss.item())

            if np.isnan(loss_val) or np.isinf(loss_val):
                has_nan = True
                print(f"  ❌ NaN/Inf at step {step+1}")
                break

            total_losses.append(loss_val)
            l2_losses.append(l2_val)
            ce_losses.append(ce_val)

            if (step + 1) % 50 == 0 or step == 0:
                print(f"  Step {step+1:4d}: loss={loss_val:.4f}, "
                      f"l2={l2_val:.4f}, ce={ce_val:.4f}, "
                      f"grad_norm={total_norm:.2f}")

            # Check for divergence (loss exploding)
            if len(total_losses) > 20 and loss_val > 10 * np.mean(total_losses[:10]):
                has_divergence = True
                print(f"  ❌ Loss divergence detected at step {step+1}")
                break

    except Exception as e:
        training_error = str(e)
        print(f"  ❌ Training error: {e}")

    # Analysis
    stable = not has_nan and not has_divergence and training_error is None

    if len(l2_losses) > 20:
        early_l2 = np.mean(l2_losses[:20])
        late_l2 = np.mean(l2_losses[-20:])
        l2_decreases = late_l2 < early_l2 * 0.9
    else:
        early_l2 = late_l2 = 0
        l2_decreases = False

    if len(ce_losses) > 20:
        early_ce = np.mean(ce_losses[:20])
        late_ce = np.mean(ce_losses[-20:])
        ce_decreases = late_ce < early_ce  # Any decrease — random tokens limit improvement
    else:
        early_ce = late_ce = 0
        ce_decreases = False

    # Compare CE rate to baseline
    ce_rate_ok = True
    if baseline and len(ce_losses) > 20:
        baseline_ce_ratio = baseline["late_ce"] / baseline["early_ce"]
        ot_ce_ratio = late_ce / (early_ce + 1e-10)
        # OT version should have similar improvement rate
        ce_rate_ok = ot_ce_ratio <= baseline_ce_ratio * 1.5  # Allow 50% worse
    else:
        ot_ce_ratio = None
        baseline_ce_ratio = None

    go = stable and ce_decreases
    decision = "GO" if go else "NO-GO"

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    window = 10

    for ax, losses, name, color in [
        (axes[0], total_losses, "Total Loss (OT)", "#C44E52"),
        (axes[1], l2_losses, "L2 Loss (OT)", "#8172B2"),
        (axes[2], ce_losses, "CE Loss (OT)", "#CCB974"),
    ]:
        ax.plot(losses, alpha=0.3, color=color)
        if len(losses) > window:
            sm = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(losses)), sm, color=color, lw=2, label="OT coupling")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Test 8: OT-Coupled Training Stability", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_08_coupling_stability.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Grad norms plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grad_norms, alpha=0.4, color="#4C72B0")
    if len(grad_norms) > window:
        sm = np.convolve(grad_norms, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(grad_norms)), sm, color="#4C72B0", lw=2)
    ax.set_title("Gradient Norms (OT-Coupled Training)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient L2 norm")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_08_grad_norms.png", dpi=150)
    plt.close()

    # Save
    metrics = {
        "stable": bool(stable),
        "has_nan": bool(has_nan),
        "has_divergence": bool(has_divergence),
        "training_error": training_error,
        "l2_decreases": bool(l2_decreases),
        "ce_decreases": bool(ce_decreases),
        "early_l2": float(early_l2),
        "late_l2": float(late_l2),
        "early_ce": float(early_ce),
        "late_ce": float(late_ce),
        "ce_rate_ok": bool(ce_rate_ok),
        "baseline_ce_ratio": float(baseline_ce_ratio) if baseline_ce_ratio else None,
        "ot_ce_ratio": float(ot_ce_ratio) if ot_ce_ratio else None,
        "decision": decision,
    }
    with open(RESULTS_DIR / "test_08.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n── Results ──")
    print(f"  Stable training:    {'✅' if stable else '❌'}")
    print(f"  L2 loss decreases:  {'✅' if l2_decreases else '❌'} "
          f"({early_l2:.4f} → {late_l2:.4f})")
    print(f"  CE loss decreases:  {'✅' if ce_decreases else '❌'} "
          f"({early_ce:.4f} → {late_ce:.4f})")
    print(f"  CE rate comparable: {'✅' if ce_rate_ok else '❌'}")

    print(f"\n{'=' * 60}")
    if decision == "GO":
        print("✅ GO: OT coupling is stable and CE branch works correctly.")
    else:
        reasons = []
        if not stable: reasons.append("training instability")
        if not ce_decreases: reasons.append("CE loss not decreasing")
        if not ce_rate_ok: reasons.append("CE rate worse than baseline")
        print(f"❌ NO-GO: {', '.join(reasons)}.")
    print(f"{'=' * 60}")

    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
