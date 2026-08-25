#!/usr/bin/env python3
"""Test 7 — Toy-Scale Baseline (Linear Interpolant) Reproduction.

Trains a deliberately tiny ELF-like model (2-4 transformer layers, hidden=128,
bottleneck=32) on a small data subset to verify code correctness.

Checks:
  (a) MSE denoising loss decreases over training
  (b) CE decode-branch loss decreases over training
  (c) x-prediction and shared-weight decode branch run without shape/dtype errors
  (d) Forward sampling pass produces non-degenerate token sequences

Go/No-Go: All four checks pass (code-correctness gate).
"""

import json
import sys
import os
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF
from modules.t5_encoder import T5Encoder, T5EncoderConfig
from utils.sampling_utils import (
    add_noise, sample_timesteps, net_out_to_v_x,
    get_sampling_steps, _ode_step,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Tiny model config — deliberately small for fast convergence on synthetic data
TOY_HIDDEN = 64
TOY_DEPTH = 2
TOY_HEADS = 4
TOY_BOTTLENECK = 32
TOY_MLP_RATIO = 2.0
TOY_MAX_LEN = 16
TOY_ENCODER_DIM = 64      # Small encoder dim (instead of 512)
TOY_VOCAB_SIZE = 256       # Small vocab for fast CE convergence
NUM_TRAIN_STEPS = 500
BATCH_SIZE = 32
LR = 1e-3
DECODER_PROB = 0.5
T_EPS = 0.05


class ToyConfig:
    """Minimal config matching ELF's expected interface."""
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
    num_sampling_steps = [10]


def create_toy_model(encoder_dim: int, vocab_size: int) -> ELF:
    """Create a tiny ELF model for testing."""
    model = ELF(
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
    return model


def generate_toy_data(batch_size: int, seq_len: int, encoder_dim: int,
                      vocab_size: int, device: str):
    """Generate synthetic training data matching ELF's expected input shapes."""
    # Simulate encoder output (x0)
    x0 = torch.randn(batch_size, seq_len, encoder_dim, device=device)
    # Random token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Attention mask (all valid)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    # No conditioning (all targets)
    cond_seq_mask = torch.zeros(batch_size, seq_len, device=device)

    return x0, input_ids, attention_mask, cond_seq_mask


def train_step(model, x0, input_ids, attention_mask, cond_seq_mask, config, device):
    """Single training step mimicking ELF's train_step.py logic."""
    dtype = next(model.parameters()).dtype
    batch_size, seq_length, encoder_dim = x0.shape

    # Sample timesteps
    t = sample_timesteps(
        batch_size, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule, device=device, dtype=dtype,
    )

    noise = torch.randn_like(x0)
    loss_mask = attention_mask * (1 - cond_seq_mask)
    cond_mask_3d = cond_seq_mask.unsqueeze(-1)

    # Flow-matching interpolation: z = t*x0 + (1-t)*noise
    t_exp = t.reshape(-1, 1, 1)
    denoiser_z = t_exp * x0 + (1 - t_exp) * noise * config.denoiser_noise_scale
    denoiser_z = cond_mask_3d * x0 + (1 - cond_mask_3d) * denoiser_z

    # v target
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_exp, min=config.t_eps)

    # Per-example decoder/denoiser branching
    decoder_active = torch.bernoulli(
        torch.full((batch_size,), config.decoder_prob)
    ).to(device=device, dtype=dtype)
    decoder_B11 = decoder_active.view(-1, 1, 1)
    decoder_B1 = decoder_active.view(-1, 1)

    # Decoder branch input
    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn_like(x0) * config.decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    # Mixed input
    t_mixed = decoder_active * 1.0 + (1.0 - decoder_active) * t
    z_mixed = decoder_B11 * decoder_z + (1.0 - decoder_B11) * denoiser_z

    # Forward pass
    net_out, decoder_logits = model(
        z_mixed, t_mixed,
        deterministic=False,
        decoder_step_active=decoder_active,
    )

    # CE loss (decoder branch)
    import torch.nn.functional as F
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    # L2 loss (denoiser branch)
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
    l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

    # Masks
    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_B1
    l2_mask = loss_mask_f * (1.0 - decoder_B1)

    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

    ce_loss = ((ce_per_token * ce_mask).sum() / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss = ((l2_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    return loss, l2_loss, ce_loss


def toy_sampling(model, config, encoder_dim, seq_len, vocab_size, device, n_samples=4):
    """Run a toy forward sampling pass (Euler ODE)."""
    model.eval()
    n_steps = 10
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        # Start from noise
        z = torch.randn(n_samples, seq_len, encoder_dim, device=device, dtype=dtype)
        cond_seq = torch.zeros_like(z)
        cond_seq_mask = torch.zeros(n_samples, seq_len, 1, device=device, dtype=dtype)

        steps = torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)

        x_pred = None
        for i in range(n_steps):
            t_curr = steps[i]
            t_next = steps[i + 1]
            t_batch = torch.full((n_samples,), float(t_curr), dtype=dtype, device=device)

            net_out, decoder_logits = model(z, t_batch, deterministic=True,
                                            decoder_step_active=None)
            v_pred, x_pred = net_out_to_v_x(net_out, z, t_batch, config.t_eps)
            z = z + (t_next - t_curr) * v_pred

        # Decode final z to tokens using the decoder head
        # Run a final forward pass with decoder active
        t_final = torch.ones(n_samples, dtype=dtype, device=device)
        _, decoder_logits = model(z, t_final, deterministic=True,
                                  decoder_step_active=torch.ones(n_samples, device=device))
        token_ids = decoder_logits.argmax(dim=-1)  # (n_samples, seq_len)

    return token_ids


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 7 — Toy-Scale Baseline Reproduction")
    print("=" * 60)

    config = ToyConfig()
    encoder_dim = TOY_ENCODER_DIM
    vocab_size = TOY_VOCAB_SIZE

    print(f"\nDevice: {DEVICE}")
    print(f"Model: depth={TOY_DEPTH}, hidden={TOY_HIDDEN}, heads={TOY_HEADS}, "
          f"bottleneck={TOY_BOTTLENECK}")
    print(f"Training: {NUM_TRAIN_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")

    # Create model
    model = create_toy_model(encoder_dim, vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    l2_losses, ce_losses, total_losses = [], [], []
    check_c_pass = True  # (c) no shape/dtype errors during training

    print(f"\n── Training ──")
    model.train()
    try:
        for step in range(NUM_TRAIN_STEPS):
            x0, input_ids, attention_mask, cond_seq_mask = generate_toy_data(
                BATCH_SIZE, TOY_MAX_LEN, encoder_dim, vocab_size, DEVICE
            )

            optimizer.zero_grad()
            loss, l2_loss, ce_loss = train_step(
                model, x0, input_ids, attention_mask, cond_seq_mask, config, DEVICE
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_losses.append(float(loss.item()))
            l2_losses.append(float(l2_loss.item()))
            ce_losses.append(float(ce_loss.item()))

            if (step + 1) % 100 == 0 or step == 0:
                print(f"  Step {step+1:4d}: loss={loss.item():.4f}, "
                      f"l2={l2_loss.item():.4f}, ce={ce_loss.item():.4f}")
    except Exception as e:
        check_c_pass = False
        print(f"  ❌ Shape/dtype error during training: {e}")

    # Check (a): MSE (L2) loss decreases
    if len(l2_losses) > 50:
        early_l2 = np.mean(l2_losses[:50])
        late_l2 = np.mean(l2_losses[-50:])
        check_a = late_l2 < early_l2 * 0.95  # At least 5% reduction
    else:
        early_l2 = np.mean(l2_losses[:10]) if l2_losses else 0
        late_l2 = np.mean(l2_losses[-10:]) if l2_losses else 0
        check_a = late_l2 < early_l2
    print(f"\n  (a) L2 loss decreases: {'✅' if check_a else '❌'} "
          f"(early={early_l2:.4f} → late={late_l2:.4f})")

    # Check (b): CE loss decreases
    if len(ce_losses) > 50:
        early_ce = np.mean(ce_losses[:50])
        late_ce = np.mean(ce_losses[-50:])
        check_b = late_ce < early_ce  # Any decrease — on random tokens, improvement is inherently slow
    else:
        early_ce = np.mean(ce_losses[:10]) if ce_losses else 0
        late_ce = np.mean(ce_losses[-10:]) if ce_losses else 0
        check_b = late_ce < early_ce
    print(f"  (b) CE loss decreases: {'✅' if check_b else '❌'} "
          f"(early={early_ce:.4f} → late={late_ce:.4f})")

    # Check (c): No shape/dtype errors
    print(f"  (c) No shape/dtype errors: {'✅' if check_c_pass else '❌'}")

    # Check (d): Forward sampling produces non-degenerate sequences
    print(f"\n── Forward Sampling ──")
    check_d = False
    try:
        token_ids = toy_sampling(model, config, encoder_dim, TOY_MAX_LEN, vocab_size, DEVICE)
        # Check: not all same token, not all padding (token 0)
        unique_counts = [len(torch.unique(token_ids[i])) for i in range(token_ids.shape[0])]
        avg_unique = np.mean(unique_counts)
        all_same = all(u <= 1 for u in unique_counts)
        check_d = not all_same and avg_unique > 2
        print(f"  Generated {token_ids.shape[0]} sequences of length {token_ids.shape[1]}")
        print(f"  Unique tokens per sequence: {unique_counts}")
        print(f"  Non-degenerate: {'✅' if check_d else '❌'}")
    except Exception as e:
        print(f"  ❌ Sampling error: {e}")

    all_pass = check_a and check_b and check_c_pass and check_d

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    window = 10

    for ax, losses, name, color in [
        (axes[0], total_losses, "Total Loss", "#4C72B0"),
        (axes[1], l2_losses, "L2 (Denoiser) Loss", "#55A868"),
        (axes[2], ce_losses, "CE (Decoder) Loss", "#DD8452"),
    ]:
        ax.plot(losses, alpha=0.3, color=color)
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(losses)), smoothed, color=color, lw=2)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_07_training_curves.png", dpi=150)
    plt.close()

    # Save results
    metrics = {
        "model_params": n_params,
        "num_steps": NUM_TRAIN_STEPS,
        "check_a_l2_decreases": bool(check_a),
        "check_b_ce_decreases": bool(check_b),
        "check_c_no_errors": bool(check_c_pass),
        "check_d_nondegen_sampling": bool(check_d),
        "all_pass": bool(all_pass),
        "early_l2": float(early_l2),
        "late_l2": float(late_l2),
        "early_ce": float(early_ce),
        "late_ce": float(late_ce),
        "final_total_loss": float(total_losses[-1]) if total_losses else None,
    }
    with open(RESULTS_DIR / "test_07.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    if all_pass:
        print("✅ ALL CHECKS PASS: Toy baseline code is verified correct.")
    else:
        fails = []
        if not check_a: fails.append("(a) L2 loss")
        if not check_b: fails.append("(b) CE loss")
        if not check_c_pass: fails.append("(c) shape/dtype")
        if not check_d: fails.append("(d) sampling")
        print(f"❌ FAILED: {', '.join(fails)}. Fix before proceeding.")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
