#!/usr/bin/env python3
"""L2 — Follow-up Audit.

1. Updated Degeneracy Detector (N=32, L=128)
2. Context-Length PPL Check (N=16, L=1024)
"""

import json
import sys
import math
import time
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent / "real_elf" / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF_models
from utils.sampling_utils import _sde_step, get_sampling_steps
from configs.config import Config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32

N_STEPS = 32
T_EPS = 0.05
SDE_GAMMA = 1.5
SELF_COND_CFG_SCALE = 3.0
DENOISER_NOISE_SCALE = 2.0

COMMON_KWARGS = dict(
    text_encoder_dim=512,
    bottleneck_dim=128,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,
)

config = Config()
config.denoiser_p_mean = -1.5
config.denoiser_p_std = 0.8
config.denoiser_noise_scale = 2.0
config.t_eps = 0.05
config.num_self_cond_cfg_tokens = 4
config.self_cond_prob = 0.5


def load_elf_b(max_length):
    print(f"  Loading ELF-B (max_length={max_length})...")
    hf_repo = "embedded-language-flows/ELF-B-owt-torch"
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    files = list_repo_files(hf_repo)
    ckpt_files = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    if not ckpt_files:
        ckpt_path = snapshot_download(hf_repo)
    else:
        ckpt_path = hf_hub_download(hf_repo, ckpt_files[0])
    
    model = ELF_models["ELF-B"](max_length=max_length, **COMMON_KWARGS)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("ema_params1", ckpt.get("params", ckpt))
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=DEVICE, dtype=DTYPE).eval()


def is_degenerate(token_ids, seq_len, threshold=2):
    if len(set(int(t) for t in token_ids)) <= threshold:
        return True
    
    scale = max(1, seq_len // 128)
    
    if len(token_ids) >= 8:
        ngrams8 = [tuple(token_ids[i:i+8]) for i in range(len(token_ids)-7)]
        if Counter(ngrams8).most_common(1)[0][1] >= 3 * scale:
            return True
            
    if len(token_ids) >= 4:
        ngrams4 = [tuple(token_ids[i:i+4]) for i in range(len(token_ids)-3)]
        if Counter(ngrams4).most_common(1)[0][1] >= 5 * scale:
            return True
            
    return False


def wilson_ci(k, n):
    if n == 0: return 0.0, 0.0
    z = 1.96
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def sde_sample(model, n, seq_len, enc_dim):
    generator = torch.Generator(device=DEVICE if DEVICE != "mps" else "cpu")
    generator.manual_seed(42)

    with torch.no_grad():
        if DEVICE == "cuda":
            z = torch.randn((n, seq_len, enc_dim), dtype=DTYPE, device=DEVICE) * config.denoiser_noise_scale
        else:
            z = (torch.randn((n, seq_len, enc_dim), generator=generator, dtype=DTYPE)
                 .to(DEVICE) * config.denoiser_noise_scale)

        steps = get_sampling_steps(
            n_steps=N_STEPS, time_schedule="logit_normal",
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=DEVICE, dtype=DTYPE
        )

        x_pred_prev = None
        cond_seq = torch.zeros((n, seq_len, enc_dim), dtype=DTYPE, device=DEVICE)
        cond_seq_mask = torch.zeros((n, seq_len), dtype=DTYPE, device=DEVICE)
        
        for i in range(N_STEPS):
            t_curr, t_next = steps[i].item(), steps[i+1].item()
            z, x_pred_prev = _sde_step(
                model=model, z=z, t=t_curr, t_next=t_next, x_pred_prev=x_pred_prev,
                config=config, cfg_scale=1.0, self_cond_cfg_scale=SELF_COND_CFG_SCALE,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, gamma=SDE_GAMMA, generator=generator
            )
    return z


def decode_to_tokens(model, z):
    with torch.no_grad():
        t_f = torch.ones(z.shape[0], dtype=DTYPE, device=DEVICE)
        dec_active = torch.ones(z.shape[0], device=DEVICE)
        sc_scale = torch.full((z.shape[0],), SELF_COND_CFG_SCALE, dtype=DTYPE, device=DEVICE)
        z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_input, t_f, deterministic=True,
                          self_cond_cfg_scale=sc_scale, decoder_step_active=dec_active)
        return logits.argmax(dim=-1).cpu()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("L3 — Full-Context Degeneracy Audit")
    print("=" * 70)
    
    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    enc_dim = COMMON_KWARGS["text_encoder_dim"]
    
    # --- Context Length Check (L=1024, N=32) ---
    print("\n─── Full Context Degeneracy Check (L=1024, N=32) ───")
    model = load_elf_b(max_length=1024)
    n = 32
    seq_len = 1024
    batch_size = 2  # MPS can probably handle batch size 2 for generation, if it OOMs we will scale down. Wait, we used bs=1 previously. Let's use bs=2 to speed it up. Actually, bs=1 is safer for L=1024 on 512 dim. Let's stick to 1 to be absolutely safe from OOM since this script will run unattended.
    batch_size = 1
    
    all_degs, all_texts = [], []
    torch.manual_seed(2024)
    
    for batch_start in range(0, n, batch_size):
        bs = min(batch_size, n - batch_start)
        print(f"  Batch {batch_start//batch_size + 1}/{(n+batch_size-1)//batch_size}...")
        z_final = sde_sample(model, bs, seq_len, enc_dim)
        tids = decode_to_tokens(model, z_final)
        
        for i in range(bs):
            seq = tids[i].tolist()
            deg = is_degenerate(seq, seq_len)
            all_degs.append(deg)
            
            text = t5_tokenizer.decode(seq, skip_special_tokens=True)
            all_texts.append(text)
            
    deg_count = sum(all_degs)
    deg_rate = deg_count / n
    ci_low, ci_high = wilson_ci(deg_count, n)
    
    print(f"\n  Final Degeneracy Rate (L=1024): {100*deg_rate:.1f}% (95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%])")
    
    is_artifact = (deg_rate < 0.15)
    verdict = "PASS: High degeneracy was an L=128 artifact. L=1024 operates cleanly." if is_artifact else \
              f"FAIL: Degeneracy is genuinely high ({100*deg_rate:.1f}%) even at L=1024. Investigation required."
              
    print("\n" + "=" * 70)
    print(f"  Verdict: {verdict}")
    print("=" * 70)
    
    with open(RESULTS_DIR / "testL3_degeneracy_L1024.md", "w") as f:
        f.write("# Phase 0L Follow-Up: L=1024 Degeneracy Audit\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. Full-Context Audit Results\n")
        f.write(f"- Sequences checked: {n} (L={seq_len})\n")
        f.write(f"- Templated/n-gram repetition rate: **{100*deg_rate:.1f}%** ")
        f.write(f"(95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%])\n\n")
        
        f.write(f"## Verdict\n**{verdict}**\n\n")
        
        f.write("## Appendix: Degenerate Samples\n")
        degen_indices = [i for i, d in enumerate(all_degs) if d]
        if not degen_indices:
            f.write("No degenerate samples found.\n")
        else:
            for i in degen_indices:
                f.write(f"\n### Degenerate Sample (Index {i})\n```\n{all_texts[i][:1500]}...\n```\n")

if __name__ == "__main__":
    main()
