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

def train_step(model, x0, input_ids, attention_mask, cond_seq_mask, config, device, curvature_weight=5.0):
    dtype = next(model.parameters()).dtype
    batch_size, seq_length, encoder_dim = x0.shape

    t = torch.rand((batch_size,), device=device, dtype=dtype)
    noise = torch.randn_like(x0)
    loss_mask = attention_mask * (1 - cond_seq_mask)
    cond_mask_3d = cond_seq_mask.unsqueeze(-1)

    t_exp = t.reshape(-1, 1, 1)
    denoiser_z = t_exp * x0 + (1 - t_exp) * noise * config.denoiser_noise_scale
    denoiser_z = cond_mask_3d * x0 + (1 - cond_mask_3d) * denoiser_z
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_exp, min=config.t_eps)

    decoder_active = torch.zeros((batch_size,), device=device, dtype=dtype)
    t_mixed = t
    z_mixed = denoiser_z

    net_out, decoder_logits = model(
        z_mixed, t_mixed, deterministic=False, decoder_step_active=decoder_active,
    )

    from utils.sampling_utils import net_out_to_v_x
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
    l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

    loss_mask_f = loss_mask.to(dtype)
    l2_mask = loss_mask_f
    
    # Base loss
    base_loss = (l2_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)

    dt = 0.05
    t_prev_val = torch.clamp(t - dt, min=0.0)
    z_prev = denoiser_z - (t - t_prev_val).view(-1, 1, 1) * v_pred.detach()
    net_out_prev, _ = model(z_prev, t_prev_val, deterministic=False)
    v_prev, _ = net_out_to_v_x(net_out_prev, z_prev, t_prev_val, config.t_eps)
    
    dot = (v_pred * v_prev).sum(dim=-1)
    norm_v = torch.linalg.norm(v_pred, dim=-1)
    norm_prev = torch.linalg.norm(v_prev, dim=-1)
    cos_sim = dot / (norm_v * norm_prev + 1e-8)
    cos_sim = torch.clamp(cos_sim, -0.999, 0.999)
    angle = torch.acos(cos_sim)
    curvature_loss = (angle * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)

    total_loss = base_loss + curvature_weight * curvature_loss

    return total_loss, base_loss, curvature_loss, angle

def main():
    embeddings, token_ids, vocab_size = prepare_real_data()
    model = create_model(ENCODER_DIM, vocab_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("=" * 60)
    print("T1 — R2 Sanity Gate")
    print("=" * 60)

    config = ToyConfig()
    rng = np.random.default_rng(42)
    n_sequences = len(embeddings)
    BATCH_SIZE = 32
    curvature_weight = 5.0
    
    base_norms = []
    curv_norms = []
    angles = []
    
    model.train()
    for step in range(100):
        idx = rng.choice(n_sequences, BATCH_SIZE, replace=False)
        x0 = torch.tensor(embeddings[idx], dtype=torch.float32, device=DEVICE)
        ids = torch.tensor(token_ids[idx], dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)
        cond_seq_mask = torch.zeros(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)

        optimizer.zero_grad()
        total_loss, base_loss, curv_loss, angle = train_step(
            model, x0, ids, attention_mask, cond_seq_mask, config, DEVICE, curvature_weight
        )
        
        # Get base grad norm
        base_loss.backward(retain_graph=True)
        base_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        optimizer.zero_grad()
        
        # Get curv grad norm
        (curvature_weight * curv_loss).backward(retain_graph=True)
        curv_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        optimizer.zero_grad()
        
        # Actual update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        base_norms.append(base_norm)
        curv_norms.append(curv_norm)
        angles.append(angle.mean().item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Angle={angle.mean().item():.4f} BaseNorm={base_norm:.4f} CurvNorm={curv_norm:.4f}")
            
    print(f"\nFinal Angle: {angles[-1]:.4f} (Start: {angles[0]:.4f})")
    print(f"Avg Base Grad Norm: {np.mean(base_norms):.4f}")
    print(f"Avg Curv Grad Norm: {np.mean(curv_norms):.4f}")
    
    ratio = np.mean(curv_norms) / (np.mean(base_norms) + 1e-8)
    print(f"Ratio Curv/Base: {ratio:.4f}")

if __name__ == "__main__":
    main()
