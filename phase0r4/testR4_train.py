#!/usr/bin/env python3
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0r.testR2_curvature_reg import prepare_real_data, create_model, ToyConfig, ENCODER_DIM, TOY_MAX_LEN

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

NUM_TRAIN_STEPS = 5000
BATCH_SIZE = 32
LR = 1e-3

def train_step(model, x0, input_ids, attention_mask, cond_seq_mask, config, device, consistency_weight=0.0):
    dtype = next(model.parameters()).dtype
    batch_size, seq_length, encoder_dim = x0.shape

    from utils.sampling_utils import sample_timesteps
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

    from utils.sampling_utils import net_out_to_v_x
    v_pred, x_pred = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
    l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_B1
    l2_mask = loss_mask_f * (1.0 - decoder_B1)

    # Consistency Smoothing
    if consistency_weight > 0.0:
        dt = 0.05
        t_prev = torch.clamp(t - dt, min=0.0)
        t_prev_exp = t_prev.reshape(-1, 1, 1)
        
        # We sample z at t_prev using the exact forward process (the straight line ODE)
        z_prev = t_prev_exp * x0 + (1 - t_prev_exp) * noise * config.denoiser_noise_scale
        z_prev = cond_mask_3d * x0 + (1 - cond_mask_3d) * z_prev
        
        net_out_prev, _ = model(z_prev, t_prev, deterministic=False)
        _, x_pred_prev = net_out_to_v_x(net_out_prev, z_prev, t_prev, config.t_eps)
        
        # MSE between x_pred and x_pred_prev
        # Target is detached to prevent collapse, standard for consistency models
        consistency_loss_per_token = ((x_pred - x_pred_prev.detach()) ** 2).mean(dim=-1)
        consistency_loss = (consistency_loss_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)
    else:
        consistency_loss = torch.tensor(0.0, device=device)

    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0) + consistency_weight * consistency_loss

    ce_loss = ((ce_per_token * ce_mask).sum() / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss = ((l2_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    return loss, l2_loss, ce_loss, consistency_loss

def finetune_toy_model(embeddings, token_ids, vocab_size, consistency_weight, ckpt_name):
    print("=" * 60)
    print(f"R4 — Training {ckpt_name} (ConsisWeight={consistency_weight})")
    print("=" * 60)

    n_sequences = len(embeddings)
    random_baseline = np.log(vocab_size)
    config = ToyConfig()
    
    c4_ckpt = torch.load(Path(__file__).resolve().parent.parent / "phase0c" / "results" / "mini_elf_real_checkpoint.pt", map_location=DEVICE, weights_only=False)
    model = create_model(ENCODER_DIM, vocab_size).to(DEVICE)
    model.load_state_dict(c4_ckpt["model_state_dict"])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_TRAIN_STEPS, eta_min=LR*0.01)

    rng = np.random.default_rng(42)
    l2_losses, ce_losses, total_losses, cons_losses = [], [], [], []
    LOG_FREQ = 500
    t0 = time.time()

    model.train()
    for step in range(NUM_TRAIN_STEPS):
        idx = rng.choice(n_sequences, BATCH_SIZE, replace=False)
        x0 = torch.tensor(embeddings[idx], dtype=torch.float32, device=DEVICE)
        ids = torch.tensor(token_ids[idx], dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)
        cond_seq_mask = torch.zeros(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)

        optimizer.zero_grad()
        loss, l2_loss, ce_loss, cons_loss = train_step(
            model, x0, ids, attention_mask, cond_seq_mask, config, DEVICE, consistency_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_losses.append(float(loss.item()))
        l2_losses.append(float(l2_loss.item()))
        ce_losses.append(float(ce_loss.item()))
        cons_losses.append(float(cons_loss.item()))

        if (step + 1) % LOG_FREQ == 0:
            elapsed = time.time() - t0
            avg_ce = np.mean(ce_losses[-LOG_FREQ:])
            print(f"  Step {step+1:5d}/{NUM_TRAIN_STEPS}: loss={np.mean(total_losses[-LOG_FREQ:]):.4f}, ce={avg_ce:.4f}, cons={np.mean(cons_losses[-LOG_FREQ:]):.4f}")

    ckpt_path = RESULTS_DIR / ckpt_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "encoder_dim": ENCODER_DIM,
            "vocab_size": vocab_size,
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "bottleneck_dim": 32,
            "max_length": TOY_MAX_LEN,
        },
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, token_ids, vocab_size = prepare_real_data()
    
    finetune_toy_model(embeddings, token_ids, vocab_size, 0.0, "testR4_model_control.pt")
    finetune_toy_model(embeddings, token_ids, vocab_size, 1.0, "testR4_model_consistency_reg.pt")

if __name__ == "__main__":
    main()
