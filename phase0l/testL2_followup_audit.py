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


def setup_eval_model():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(DEVICE).eval()
    return model, tokenizer


def score_text(text, model, tokenizer, max_length):
    if not text.strip(): return float('inf')
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(DEVICE)
    if input_ids.shape[1] < 2: return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"part1": {}, "part2": {}}
    
    print("=" * 70)
    print("L2 — Follow-up Audit")
    print("=" * 70)
    
    eval_model, eval_tokenizer = setup_eval_model()
    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    enc_dim = COMMON_KWARGS["text_encoder_dim"]
    
    # --- PART 1: Updated Degeneracy Audit ---
    print("\n─── Part 1: Degeneracy Audit (L=128, N=32) ───")
    model1 = load_elf_b(max_length=128)
    n1 = 32
    seq_len1 = 128
    batch_size1 = 4
    
    all_ppls1, all_degs1, all_texts1 = [], [], []
    raw_ppls1 = []
    torch.manual_seed(42)
    
    for batch_start in range(0, n1, batch_size1):
        bs = min(batch_size1, n1 - batch_start)
        print(f"  Batch {batch_start//batch_size1 + 1}/{(n1+batch_size1-1)//batch_size1}...")
        z_final = sde_sample(model1, bs, seq_len1, enc_dim)
        tids = decode_to_tokens(model1, z_final)
        
        for i in range(bs):
            seq = tids[i].tolist()
            deg = is_degenerate(seq, seq_len1)
            all_degs1.append(deg)
            
            text = t5_tokenizer.decode(seq, skip_special_tokens=True)
            all_texts1.append(text)
            
            ppl = score_text(text, eval_model, eval_tokenizer, 1024)
            if math.isnan(ppl) or math.isinf(ppl):
                ppl = float('inf')
            
            raw_ppls1.append(ppl)
            
            if deg:
                all_ppls1.append(float('inf'))
            else:
                all_ppls1.append(ppl)
                
    degs1_count = sum(all_degs1)
    deg_rate1 = degs1_count / n1
    ci_low, ci_high = wilson_ci(degs1_count, n1)
    
    vp1 = [p for d, p in zip(all_degs1, all_ppls1) if not d]
    mean_ppl1 = float(np.mean(vp1)) if vp1 else float('nan')
    mean_raw_ppl1 = float(np.mean([p for p in raw_ppls1 if not math.isinf(p)]))
    
    print(f"  Updated Degeneracy Rate: {100*deg_rate1:.1f}% (95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%])")
    print(f"  Non-degen PPL (L=128): {mean_ppl1:.1f}")
    print(f"  Raw PPL (all L=128): {mean_raw_ppl1:.1f}")
    
    report["part1"] = {
        "N": n1, "length": seq_len1,
        "deg_count": degs1_count,
        "deg_rate": deg_rate1,
        "ci_low": ci_low, "ci_high": ci_high,
        "mean_ppl": mean_ppl1
    }
    
    del model1
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    
    # --- PART 2: Context Length Check ---
    print("\n─── Part 2: Context Length Check (L=1024, N=8) ───")
    model2 = load_elf_b(max_length=1024)
    n2 = 8
    seq_len2 = 1024
    batch_size2 = 1  # For memory safety on MPS
    
    all_ppls2, all_degs2, all_texts2 = [], [], []
    raw_ppls2 = []
    torch.manual_seed(123)
    
    for batch_start in range(0, n2, batch_size2):
        bs = min(batch_size2, n2 - batch_start)
        print(f"  Batch {batch_start//batch_size2 + 1}/{(n2+batch_size2-1)//batch_size2}...")
        z_final = sde_sample(model2, bs, seq_len2, enc_dim)
        tids = decode_to_tokens(model2, z_final)
        
        for i in range(bs):
            seq = tids[i].tolist()
            deg = is_degenerate(seq, seq_len2)
            all_degs2.append(deg)
            
            text = t5_tokenizer.decode(seq, skip_special_tokens=True)
            all_texts2.append(text)
            
            ppl = score_text(text, eval_model, eval_tokenizer, 1024)
            if math.isnan(ppl) or math.isinf(ppl):
                ppl = float('inf')
                
            raw_ppls2.append(ppl)
            
            if deg:
                all_ppls2.append(float('inf'))
            else:
                all_ppls2.append(ppl)
                
    vp2 = [p for d, p in zip(all_degs2, all_ppls2) if not d]
    mean_ppl2 = float(np.mean(vp2)) if vp2 else float('nan')
    mean_raw_ppl2 = float(np.mean([p for p in raw_ppls2 if not math.isinf(p)]))
    
    print(f"  Non-degen PPL (L=1024): {mean_ppl2:.1f} (Paper=24.1)")
    print(f"  Raw PPL (all L=1024): {mean_raw_ppl2:.1f}")
    
    report["part2"] = {
        "N": n2, "length": seq_len2,
        "deg_count": sum(all_degs2),
        "mean_ppl": mean_ppl2
    }
    
    gap_closed = (abs(mean_raw_ppl2 - 24.1) < 15.0)
    verdict = "PASS: PPL at L=1024 confirms context length was the primary gap." if gap_closed else \
              f"FAIL: PPL at L=1024 is {mean_raw_ppl2:.1f}, still far from 24.1. Config bug remains."
              
    print("\n" + "=" * 70)
    print(f"  Verdict: {verdict}")
    print("=" * 70)
    
    with open(RESULTS_DIR / "testL2_followup_audit.md", "w") as f:
        f.write("# Phase 0L Follow-Up Audit\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. Updated Degeneracy Audit\n")
        f.write(f"- Sequences checked: {n1} (L={seq_len1})\n")
        f.write(f"- Templated/n-gram repetition rate: **{100*deg_rate1:.1f}%** ")
        f.write(f"(95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%])\n")
        f.write(f"- Non-degenerate PPL at L=128: {mean_ppl1:.1f}\n")
        f.write(f"- Raw PPL (all L=128): {mean_raw_ppl1:.1f}\n\n")
        
        f.write("## 2. Context-Length PPL Check\n")
        f.write(f"- Sequences checked: {n2} (L={seq_len2})\n")
        f.write(f"- Non-degenerate PPL at L=1024: **{mean_ppl2:.1f}**\n")
        f.write(f"- Raw PPL (all L=1024): **{mean_raw_ppl2:.1f}**\n")
        f.write(f"- Paper Reference: 24.1\n\n")
        
        f.write(f"## Verdict\n**{verdict}**\n")
        
        f.write("\n## Appendix: Sample L=1024 Texts\n")
        for i, txt in enumerate(all_texts2[:3]):
            st = "DEG" if all_degs2[i] else f"PPL={all_ppls2[i]:.1f}"
            f.write(f"\n### Sample {i+1} [{st}]\n```\n{txt[:800]}...\n```\n")

if __name__ == "__main__":
    main()
