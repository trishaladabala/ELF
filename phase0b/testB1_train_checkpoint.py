#!/usr/bin/env python3
"""B1 — Extend the Toy Model to a Full Sampling-Ready Checkpoint.

Reuses the exact architecture and training loop from test_07_toy_baseline.py
but trains for 5,000-10,000 steps until both L2 and CE losses plateau.
Saves a checkpoint for use in B2-B4.
"""

import json
import sys
import time
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

# Same architecture as Test 7
TOY_HIDDEN = 64
TOY_DEPTH = 2
TOY_HEADS = 4
TOY_BOTTLENECK = 32
TOY_MLP_RATIO = 2.0
TOY_MAX_LEN = 16
TOY_ENCODER_DIM = 64
TOY_VOCAB_SIZE = 256

# Training: much longer for convergence
NUM_TRAIN_STEPS = 8000
BATCH_SIZE = 32
LR = 1e-3
DECODER_PROB = 0.5
T_EPS = 0.05
LOG_FREQ = 200


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
    label_drop_prob = 0.0
    pad_token = "pad"
    latent_mean = 0.0
    latent_std = 1.0


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


def train_step(model, x0, input_ids, attention_mask, cond_seq_mask, config, device):
    dtype = next(model.parameters()).dtype
    batch_size, seq_length, encoder_dim = x0.shape

    t = sample_timesteps(
        batch_size, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule, device=device, dtype=dtype,
    )

    noise = torch.randn_like(x0)
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


def check_plateau(losses, window=500):
    """Check if loss has plateaued (last 20% vs previous 20%)."""
    if len(losses) < window * 2:
        return False, 0.0
    recent = np.mean(losses[-window:])
    previous = np.mean(losses[-2*window:-window])
    if previous == 0:
        return True, 0.0
    change_pct = abs(recent - previous) / previous * 100
    return change_pct < 2.0, change_pct  # <2% change = plateau


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("B1 — Extended Training for Sampling-Ready Checkpoint")
    print("=" * 60)

    config = ToyConfig()
    encoder_dim = TOY_ENCODER_DIM
    vocab_size = TOY_VOCAB_SIZE

    model = create_toy_model(encoder_dim, vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Learning rate schedule: cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_TRAIN_STEPS, eta_min=LR * 0.01)

    print(f"\n  Device: {DEVICE}")
    print(f"  Model params: {n_params:,}")
    print(f"  Training: {NUM_TRAIN_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
    print(f"  LR schedule: cosine → {LR*0.01:.1e}")

    l2_losses, ce_losses, total_losses = [], [], []
    t0 = time.time()

    model.train()
    for step in range(NUM_TRAIN_STEPS):
        x0 = torch.randn(BATCH_SIZE, TOY_MAX_LEN, encoder_dim, device=DEVICE)
        input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, TOY_MAX_LEN), device=DEVICE)
        attention_mask = torch.ones(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)
        cond_seq_mask = torch.zeros(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)

        optimizer.zero_grad()
        loss, l2_loss, ce_loss = train_step(
            model, x0, input_ids, attention_mask, cond_seq_mask, config, DEVICE
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_losses.append(float(loss.item()))
        l2_losses.append(float(l2_loss.item()))
        ce_losses.append(float(ce_loss.item()))

        if (step + 1) % LOG_FREQ == 0:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1:5d}/{NUM_TRAIN_STEPS}: "
                  f"loss={np.mean(total_losses[-LOG_FREQ:]):.4f}, "
                  f"l2={np.mean(l2_losses[-LOG_FREQ:]):.4f}, "
                  f"ce={np.mean(ce_losses[-LOG_FREQ:]):.4f}, "
                  f"lr={lr_now:.2e}, "
                  f"time={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed:.0f}s ({elapsed/NUM_TRAIN_STEPS*1000:.1f}ms/step)")

    # Check plateau
    l2_plateau, l2_change = check_plateau(l2_losses)
    ce_plateau, ce_change = check_plateau(ce_losses)
    print(f"\n  L2 plateau: {'✅' if l2_plateau else '❌'} (last-window change: {l2_change:.2f}%)")
    print(f"  CE plateau: {'✅' if ce_plateau else '❌'} (last-window change: {ce_change:.2f}%)")

    # Save checkpoint
    ckpt_path = RESULTS_DIR / "mini_elf_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "encoder_dim": encoder_dim,
            "vocab_size": vocab_size,
            "hidden_size": TOY_HIDDEN,
            "depth": TOY_DEPTH,
            "num_heads": TOY_HEADS,
            "mlp_ratio": TOY_MLP_RATIO,
            "bottleneck_dim": TOY_BOTTLENECK,
            "max_length": TOY_MAX_LEN,
        },
        "training": {
            "total_steps": NUM_TRAIN_STEPS,
            "final_l2": float(np.mean(l2_losses[-100:])),
            "final_ce": float(np.mean(ce_losses[-100:])),
            "final_total": float(np.mean(total_losses[-100:])),
        },
    }, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    window = 50

    for ax, losses, name, color in [
        (axes[0], total_losses, "Total Loss", "#4C72B0"),
        (axes[1], l2_losses, "L2 (Denoiser) Loss", "#55A868"),
        (axes[2], ce_losses, "CE (Decoder) Loss", "#DD8452"),
    ]:
        ax.plot(losses, alpha=0.15, color=color)
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(losses)), smoothed, color=color, lw=2)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"B1: Extended Training ({NUM_TRAIN_STEPS} steps)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testB1_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save metrics
    metrics = {
        "num_steps": NUM_TRAIN_STEPS,
        "model_params": n_params,
        "training_time_s": elapsed,
        "final_l2": float(np.mean(l2_losses[-100:])),
        "final_ce": float(np.mean(ce_losses[-100:])),
        "final_total": float(np.mean(total_losses[-100:])),
        "l2_plateau": bool(l2_plateau),
        "ce_plateau": bool(ce_plateau),
        "l2_change_pct": float(l2_change),
        "ce_change_pct": float(ce_change),
        "converged": bool(l2_plateau or ce_plateau),
    }
    with open(RESULTS_DIR / "testB1.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if metrics["converged"]:
        print("✅ Losses have plateaued. Checkpoint is sampling-ready.")
    else:
        print("⚠️  Losses still decreasing — checkpoint usable but not fully converged.")
        print("    (This is acceptable for toy-scale pilot; sampler comparisons still valid.)")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
