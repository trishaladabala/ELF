#!/usr/bin/env python3
"""L3 — XSum Generalization Check"""

import json
import sys
import math
import time
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from scipy.stats import pearsonr

SRC_DIR = Path(__file__).resolve().parent.parent / "phase0l" / "real_elf" / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF_models
from utils.sampling_utils import _sde_step, get_sampling_steps
from configs.config import Config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32

N_STEPS = 64
SDE_GAMMA = 1.5
SELF_COND_CFG_SCALE = 1.0
CFG_SCALE = 2.0

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


def load_elf_b(model_name, max_length):
    print(f"  Loading {model_name} (max_length={max_length})...")
    hf_repo = f"embedded-language-flows/{model_name}-torch"
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
    token_ids = [t for t in token_ids if t not in (0, 1)]
    if not token_ids:
        return True
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


def sde_sample(model, n, seq_len, enc_dim, cond_seq, cond_seq_mask, use_sde=True):
    generator = torch.Generator(device=DEVICE if DEVICE != "mps" else "cpu")
    generator.manual_seed(42)

    with torch.no_grad():
        if DEVICE == "cuda":
            z = torch.randn((n, seq_len, enc_dim), dtype=DTYPE, device=DEVICE) * config.denoiser_noise_scale
        else:
            z = (torch.randn((n, seq_len, enc_dim), generator=generator, dtype=DTYPE)
                 .to(DEVICE) * config.denoiser_noise_scale)
                 
        z = torch.where(cond_seq_mask.unsqueeze(-1) > 0, cond_seq, z)

        steps = get_sampling_steps(
            n_steps=N_STEPS, time_schedule="logit_normal",
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=DEVICE, dtype=DTYPE
        )

        x_pred_prev = None
        gamma = SDE_GAMMA if use_sde else 0.0
        trajectory = [z.clone()]
        
        for i in range(N_STEPS):
            t_curr, t_next = steps[i].item(), steps[i+1].item()
            z, x_pred_prev = _sde_step(
                model=model, z=z, t=t_curr, t_next=t_next, x_pred_prev=x_pred_prev,
                config=config, cfg_scale=CFG_SCALE, self_cond_cfg_scale=SELF_COND_CFG_SCALE,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, gamma=gamma, generator=generator
            )
            trajectory.append(z.clone())
            
    return z, torch.stack(trajectory, dim=1)


def compute_curvature(trajectories):
    dZ = trajectories[:, 1:] - trajectories[:, :-1]
    angles = []
    for t in range(dZ.shape[1] - 1):
        v1 = dZ[:, t].view(dZ.shape[0], -1)
        v2 = dZ[:, t+1].view(dZ.shape[0], -1)
        v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
        cos_sim = (v1_norm * v2_norm).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cos_sim) * 180 / math.pi
        angles.append(angle)
    if not angles:
        return np.zeros(trajectories.shape[0])
    angles = torch.stack(angles, dim=1)
    return angles.mean(dim=1).cpu().numpy()


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
    print("M2 — XSum Generalization Check")
    print("=" * 70)
    
    from transformers import T5Tokenizer
    from modules.t5_encoder import get_encoder
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    _, t5_encoder = get_encoder("t5-small", dtype=DTYPE)
    t5_encoder = t5_encoder.to(DEVICE).eval()
    
    enc_dim = COMMON_KWARGS["text_encoder_dim"]
    
    n = 32
    c_len = 1024
    t_len = 64
    seq_len = c_len + t_len
    batch_size = 2
    
    print("  Loading Dataset...")
    dataset = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
    samples = dataset.select(range(n))
    doc_texts = [s["document"] for s in samples]
    sum_texts = [s["summary"] for s in samples]
    
    model = load_elf_b("ELF-B-xsum", max_length=seq_len)
    
    dataset_list = []
    for i in range(n):
        dataset_list.append({
            "index": i,
            "input": doc_texts[i],
            "target": sum_texts[i],
            "condition_input_ids": t5_tokenizer(doc_texts[i], add_special_tokens=True)["input_ids"][:c_len],
            "input_ids": t5_tokenizer(sum_texts[i], add_special_tokens=True)["input_ids"][:t_len],
        })
        
    import utils.data_utils as data_utils
    dataloader = data_utils.get_dataloader(
        dataset_list, batch_size=batch_size, shuffle=False, num_workers=0,
        max_seq_length=seq_len, pad_token_id=t5_tokenizer.eos_token_id,
        max_input_seq_length=c_len, distributed=False
    )
    
    from utils.encoder_utils import encode_text
    
    ode_texts, sde_texts = [], []
    ode_degs, sde_degs = [], []
    ode_curvs, sde_curvs = [], []
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"  Batch {batch_idx + 1}/{len(dataloader)}...")
        bsz = batch["input_ids"].shape[0]
        
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(DEVICE).long()
        # Use 2D attention mask for T5 to avoid MPS 3D mask crash
        attention_mask = torch.from_numpy(np.array(batch["attention_mask"])).to(DEVICE).long()
        cond_seq_mask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(DEVICE).float()
        
        cond_seq = encode_text(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder=t5_encoder, latent_mean=0.0, latent_std=0.2,
        ).to(DTYPE)
        
        # ODE
        z_ode, traj_ode = sde_sample(model, bsz, seq_len, enc_dim, cond_seq, cond_seq_mask, use_sde=False)
        tids_ode = decode_to_tokens(model, z_ode)
        curv_ode = compute_curvature(traj_ode)
        
        # SDE
        z_sde, traj_sde = sde_sample(model, bsz, seq_len, enc_dim, cond_seq, cond_seq_mask, use_sde=True)
        tids_sde = decode_to_tokens(model, z_sde)
        curv_sde = compute_curvature(traj_sde)
        
        from utils.generation_utils import shift_left, mask_after_eos
        cond_len_per_sample = cond_seq_mask.to(torch.int32).sum(dim=1)
        
        tids_ode = shift_left(tids_ode, cond_len_per_sample, 0)[:, :t_len]
        tids_ode = mask_after_eos(tids_ode, eos_token_id=t5_tokenizer.eos_token_id, pad_token_id=t5_tokenizer.eos_token_id)
        
        tids_sde = shift_left(tids_sde, cond_len_per_sample, 0)[:, :t_len]
        tids_sde = mask_after_eos(tids_sde, eos_token_id=t5_tokenizer.eos_token_id, pad_token_id=t5_tokenizer.eos_token_id)
        
        for i in range(bsz):
            seq_ode = tids_ode[i].tolist()
            seq_sde = tids_sde[i].tolist()
            
            deg_ode = is_degenerate(seq_ode, t_len)
            deg_sde = is_degenerate(seq_sde, t_len)
            
            text_ode = t5_tokenizer.decode(seq_ode, skip_special_tokens=True)
            text_sde = t5_tokenizer.decode(seq_sde, skip_special_tokens=True)
            
            ode_texts.append(text_ode)
            sde_texts.append(text_sde)
            ode_degs.append(deg_ode)
            sde_degs.append(deg_sde)
            ode_curvs.append(curv_ode[i])
            sde_curvs.append(curv_sde[i])

    from evaluate import load
    rouge = load("rouge")
    rouge_ode = rouge.compute(predictions=ode_texts, references=sum_texts)
    rouge_sde = rouge.compute(predictions=sde_texts, references=sum_texts)
    
    ode_r1, ode_r2, ode_rl = [], [], []
    sde_r1, sde_r2, sde_rl = [], [], []
    
    for h_ode, h_sde, ref in zip(ode_texts, sde_texts, sum_texts):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        score_ode = scorer.score(ref, h_ode)
        score_sde = scorer.score(ref, h_sde)
        
        ode_r1.append(score_ode['rouge1'].fmeasure * 100)
        ode_r2.append(score_ode['rouge2'].fmeasure * 100)
        ode_rl.append(score_ode['rougeL'].fmeasure * 100)
        
        sde_r1.append(score_sde['rouge1'].fmeasure * 100)
        sde_r2.append(score_sde['rouge2'].fmeasure * 100)
        sde_rl.append(score_sde['rougeL'].fmeasure * 100)
        
    rouge_diffs = [s - o for s, o in zip(sde_rl, ode_rl)]
    curv_corr_ode, _ = pearsonr(ode_curvs, rouge_diffs)
    
    # Bootstrap CI for ROUGE diff
    np.random.seed(42)
    diffs = []
    for _ in range(500):
        idx = np.random.choice(n, size=n, replace=True)
        ode_resample = [ode_texts[i] for i in idx]
        sde_resample = [sde_texts[i] for i in idx]
        ref_resample = [sum_texts[i] for i in idx]
        sde_r = rouge.compute(predictions=sde_resample, references=ref_resample)["rougeL"]
        ode_r = rouge.compute(predictions=ode_resample, references=ref_resample)["rougeL"]
        diffs.append(sde_r - ode_r)
    diffs = np.array(diffs)
    rouge_diff_mean = np.mean(diffs)
    rouge_diff_ci_lo = np.percentile(diffs, 2.5)
    rouge_diff_ci_hi = np.percentile(diffs, 97.5)
    
    ode_deg_rate = sum(ode_degs)/n
    sde_deg_rate = sum(sde_degs)/n
    
    print("\n" + "=" * 70)
    print(f"  ODE ROUGE-L: {rouge_ode['rougeL']*100:.1f} (R1: {rouge_ode['rouge1']*100:.1f}, R2: {rouge_ode['rouge2']*100:.1f})")
    print(f"  SDE ROUGE-L: {rouge_sde['rougeL']*100:.1f} (R1: {rouge_sde['rouge1']*100:.1f}, R2: {rouge_sde['rouge2']*100:.1f})")
    print(f"  SDE-ODE ROUGE-L diff: {rouge_diff_mean*100:.2f} 95% CI [{rouge_diff_ci_lo*100:.2f}, {rouge_diff_ci_hi*100:.2f}]")
    print(f"  ODE Degeneracy: {100*ode_deg_rate:.1f}%")
    print(f"  SDE Degeneracy: {100*sde_deg_rate:.1f}%")
    print(f"  Curvature vs ROUGE diff: {curv_corr_ode:.3f}")
    print("=" * 70)
    
    results = {
        "ode_rouge": rouge_ode,
        "sde_rouge": rouge_sde,
        "rouge_diff_mean": rouge_diff_mean,
        "rouge_diff_ci_lo": rouge_diff_ci_lo,
        "rouge_diff_ci_hi": rouge_diff_ci_hi,
        "ode_deg": ode_deg_rate,
        "sde_deg": sde_deg_rate,
        "curv_corr": curv_corr_ode
    }
    
    with open(RESULTS_DIR / "xsum_results.json", "w") as f:
        json.dump(results, f)
        
    with open(RESULTS_DIR / "xsum_gens.json", "w") as f:
        json.dump({"ode": ode_texts, "sde": sde_texts, "ref": sum_texts}, f, indent=2)

if __name__ == "__main__":
    main()
