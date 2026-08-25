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

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0r.testR2_curvature_reg import prepare_real_data, create_model, ToyConfig, ENCODER_DIM, TOY_MAX_LEN

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

NUM_TRAIN_STEPS = 5000
BATCH_SIZE = 32
LR = 1e-3

class DecoupledELF(nn.Module):
    def __init__(self, denoiser, decoder):
        super().__init__()
        self.denoiser = denoiser
        self.decoder = decoder

    def forward(self, x, t, attention_mask=None, deterministic=True, self_cond_cfg_scale=None, decoder_step_active=None):
        denoiser_out, _ = self.denoiser(x, t, attention_mask, deterministic, self_cond_cfg_scale, decoder_step_active=0.0)
        _, decoder_logits = self.decoder(x, t, attention_mask, deterministic, self_cond_cfg_scale, decoder_step_active=1.0)
        return denoiser_out, decoder_logits

def train_step(model, x0, input_ids, attention_mask, cond_seq_mask, config, device, label_smoothing=0.1):
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

    decoder_logits_float = decoder_logits.to(torch.float32)
    # ce_per_token shape: (batch, seq)
    ce_per_token = F.cross_entropy(
        decoder_logits_float.transpose(1, 2), 
        input_ids, 
        reduction='none', 
        label_smoothing=label_smoothing
    )

    from utils.sampling_utils import net_out_to_v_x
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

def finetune_combined_model(embeddings, token_ids, vocab_size, label_smoothing, ckpt_name):
    print("=" * 60)
    print(f"F2a — Training {ckpt_name} (Decoupled + LS={label_smoothing})")
    print("=" * 60)

    n_sequences = len(embeddings)
    config = ToyConfig()
    
    c4_ckpt = torch.load(Path(__file__).resolve().parent.parent / "phase0c" / "results" / "mini_elf_real_checkpoint.pt", map_location=DEVICE, weights_only=False)
    
    denoiser = create_model(ENCODER_DIM, vocab_size).to(DEVICE)
    denoiser.load_state_dict(c4_ckpt["model_state_dict"])
    
    decoder = create_model(ENCODER_DIM, vocab_size).to(DEVICE)
    decoder.load_state_dict(c4_ckpt["model_state_dict"])
    
    model = DecoupledELF(denoiser, decoder).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_TRAIN_STEPS, eta_min=LR*0.01)

    rng = np.random.default_rng(42)
    l2_losses, ce_losses, total_losses = [], [], []
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
        loss, l2_loss, ce_loss = train_step(
            model, x0, ids, attention_mask, cond_seq_mask, config, DEVICE, label_smoothing
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
            avg_ce = np.mean(ce_losses[-LOG_FREQ:])
            print(f"  Step {step+1:5d}/{NUM_TRAIN_STEPS}: loss={np.mean(total_losses[-LOG_FREQ:]):.4f}, ce={avg_ce:.4f}, l2={np.mean(l2_losses[-LOG_FREQ:]):.4f}")

    ckpt_path = RESULTS_DIR / ckpt_name
    torch.save({
        "denoiser_state_dict": model.denoiser.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
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
    
    finetune_combined_model(embeddings, token_ids, vocab_size, 0.1, "testF2_model_combined.pt")

if __name__ == "__main__":
    main()
