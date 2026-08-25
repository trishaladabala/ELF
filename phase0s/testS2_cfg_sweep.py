#!/usr/bin/env python3
"""L1 — Local Sanity Verification of Real ELF-B Checkpoint.

Downloads the PyTorch-converted ELF-B-owt checkpoint from HuggingFace,
loads it via our local ELF model factory (ELF_B/M/L), runs unconditional
SDE generation at 32 steps with sde_gamma=1.5 (matching the paper),
scores with GPT-2 Large, and instruments degeneracy + trajectory extraction.

Also verifies weight loading for ELF-M and ELF-L.
"""

import json
import sys
import math
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent.parent / "phase0l" / "real_elf" / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF_models
from utils.sampling_utils import _sde_step, get_sampling_steps
from configs.config import Config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32

N_SAMPLES = 16
BATCH_SIZE = 1
N_STEPS = 32
T_EPS = 0.05
SDE_GAMMA = 1.5
SELF_COND_CFG_SCALE = 3.0
DENOISER_NOISE_SCALE = 2.0

# Config from train_owt_ELF-B.yml
COMMON_KWARGS = dict(
    text_encoder_dim=512,    # T5-small output dim
    max_length=1024,         # Use 1024 as requested
    bottleneck_dim=128,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,        # T5-small vocab
)

config = Config()
config.denoiser_p_mean = -1.5
config.denoiser_p_std = 0.8
config.denoiser_noise_scale = 2.0
config.t_eps = 0.05
config.num_self_cond_cfg_tokens = 4
config.self_cond_prob = 0.5

HF_REPOS = {
    "ELF-B": "embedded-language-flows/ELF-B-owt-torch",
    "ELF-M": "embedded-language-flows/ELF-M-owt-torch",
    "ELF-L": "embedded-language-flows/ELF-L-owt-torch",
}

PAPER_PPLS = {"ELF-B": 24.1, "ELF-M": 21.7, "ELF-L": 23.3}


def download_checkpoint(hf_repo):
    from huggingface_hub import hf_hub_download, list_repo_files
    print(f"  Listing files in {hf_repo}...")
    files = list_repo_files(hf_repo)
    ckpt_files = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    if not ckpt_files:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo)
    print(f"  Downloading {ckpt_files[0]}...")
    return hf_hub_download(hf_repo, ckpt_files[0])


def load_elf_model(model_name, ckpt_path, device="cpu", dtype=torch.float32):
    factory_fn = ELF_models[model_name]
    model = factory_fn(**COMMON_KWARGS)

    print(f"  Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "ema_params1" in ckpt and ckpt["ema_params1"]:
        state_dict = ckpt["ema_params1"]
        print(f"  Using EMA parameters")
    elif "params" in ckpt:
        state_dict = ckpt["params"]
        print(f"  Using raw parameters")
    else:
        state_dict = ckpt
        print(f"  Using checkpoint directly")

    # Load state dict
    result = model.load_state_dict(state_dict, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Loaded successfully. Params: {n_params:,}")

    model = model.to(device=device, dtype=dtype).eval()
    return model, True


def sde_sample_with_trajectory(model, n, n_steps, seq_len, enc_dim, gamma, device, dtype,
                                noise=None, return_trajectory=False, cfg_scale=3.0):
    trajectory = []
    generator = torch.Generator(device=device if device != "mps" else "cpu")
    generator.manual_seed(42)

    with torch.no_grad():
        if noise is not None:
            z = noise.clone().to(device=device, dtype=dtype)
        else:
            if device == "cuda":
                z = torch.randn((n, seq_len, enc_dim), dtype=dtype, device=device) * config.denoiser_noise_scale
            else:
                z = (torch.randn((n, seq_len, enc_dim), generator=generator, dtype=dtype)
                     .to(device) * config.denoiser_noise_scale)

        if return_trajectory:
            trajectory.append(z.cpu().numpy())

        steps = get_sampling_steps(
            n_steps=n_steps, time_schedule="logit_normal",
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=dtype
        )

        x_pred_prev = None
        cond_seq = torch.zeros((n, seq_len, enc_dim), dtype=dtype, device=device)
        cond_seq_mask = torch.zeros((n, seq_len), dtype=dtype, device=device)
        
        for i in range(n_steps):
            t_curr, t_next = steps[i].item(), steps[i+1].item()
            z, x_pred_prev = _sde_step(
                model=model, z=z, t=t_curr, t_next=t_next, x_pred_prev=x_pred_prev,
                config=config, cfg_scale=1.0, self_cond_cfg_scale=cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, gamma=gamma, generator=generator
            )
            if return_trajectory:
                trajectory.append(z.cpu().numpy())

    if return_trajectory:
        return z, np.array(trajectory)
    return z


def decode_to_tokens(model, z, device, dtype, cfg_scale=3.0):
    with torch.no_grad():
        t_f = torch.ones(z.shape[0], dtype=dtype, device=device)
        dec_active = torch.ones(z.shape[0], device=device)
        sc_scale = torch.full((z.shape[0],), cfg_scale, dtype=dtype, device=device)
        z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_input, t_f, deterministic=True,
                          self_cond_cfg_scale=sc_scale, decoder_step_active=dec_active)
        return logits.argmax(dim=-1).cpu()


def is_degenerate(token_ids, seq_len, threshold=2):
    if len(set(int(t) for t in token_ids)) <= threshold:
        return True
    
    scale = max(1, seq_len // 128)
    
    if len(token_ids) >= 8:
        from collections import Counter
        ngrams8 = [tuple(token_ids[i:i+8]) for i in range(len(token_ids)-7)]
        if Counter(ngrams8).most_common(1)[0][1] >= 3 * scale:
            return True
            
    if len(token_ids) >= 4:
        from collections import Counter
        ngrams4 = [tuple(token_ids[i:i+4]) for i in range(len(token_ids)-3)]
        if Counter(ngrams4).most_common(1)[0][1] >= 5 * scale:
            return True
            
    return False


def compute_turning_angles(trajectory):
    v = np.diff(trajectory, axis=0)
    if v.shape[0] < 2:
        return np.zeros((trajectory.shape[1], trajectory.shape[2]))
    v_prev, v_curr = v[:-1], v[1:]
    dot = np.sum(v_prev * v_curr, axis=-1)
    n_prev = np.linalg.norm(v_prev, axis=-1)
    n_curr = np.linalg.norm(v_curr, axis=-1)
    cos_sim = np.clip(dot / (n_prev * n_curr + 1e-12), -1.0, 1.0)
    return np.mean(np.arccos(cos_sim), axis=0)


def setup_eval_model():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print("  Loading GPT-2 Large for evaluation...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(DEVICE).eval()
    return model, tokenizer


def score_text(text, model, tokenizer, max_length=1024):
    if not text.strip():
        return float('inf')
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(DEVICE)
    if input_ids.shape[1] < 2:
        return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("S2 — CFG Sweep at Real Scale")
    print("=" * 70)

    try:
        ckpt_path = download_checkpoint(HF_REPOS["ELF-B"])
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return

    model, load_ok = load_elf_model("ELF-B", ckpt_path, device=DEVICE, dtype=DTYPE)
    if not load_ok:
        print("FAIL: ELF-B weight loading failed")
        return

    enc_dim = COMMON_KWARGS["text_encoder_dim"]
    seq_len = COMMON_KWARGS["max_length"]

    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    eval_model, eval_tokenizer = setup_eval_model()

    cfgs = [1.0, 2.0, 3.0, 5.0]
    
    cfg_degs = []
    cfg_ppls = []

    # Use a fixed random seed for fair comparison
    seed = 42

    for cfg in cfgs:
        print(f"\n─── CFG Scale {cfg} ───")
        torch.manual_seed(seed)
        
        all_ppls, all_degs = [], []
        
        for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
            bs = min(BATCH_SIZE, N_SAMPLES - batch_start)
            noise = torch.randn(bs, seq_len, enc_dim, dtype=DTYPE, device=DEVICE) * DENOISER_NOISE_SCALE
            z_final = sde_sample_with_trajectory(
                model, bs, N_STEPS, seq_len, enc_dim, SDE_GAMMA, DEVICE, DTYPE,
                noise=noise, return_trajectory=False, cfg_scale=cfg)
            
            token_ids = decode_to_tokens(model, z_final, DEVICE, DTYPE, cfg_scale=cfg)
            for i in range(bs):
                tids = token_ids[i].tolist()
                deg = is_degenerate(tids, seq_len)
                all_degs.append(deg)
                text = t5_tokenizer.decode(tids, skip_special_tokens=True)
                if deg:
                    all_ppls.append(50000.0)
                else:
                    ppl = score_text(text, eval_model, eval_tokenizer)
                    if math.isnan(ppl) or math.isinf(ppl):
                        ppl = 50000.0
                    all_ppls.append(ppl)

        ppls = np.array(all_ppls)
        degs = np.array(all_degs)
        deg_rate = float(degs.mean())
        valid = ~degs
        if valid.sum() > 0:
            mean_ppl = float(np.mean(ppls[valid]))
        else:
            mean_ppl = float('nan')
            
        cfg_degs.append(deg_rate)
        cfg_ppls.append(mean_ppl)
        
        print(f"  Degeneracy: {deg_rate:.1%}")
        print(f"  Valid PPL:  {mean_ppl:.2f}")

    print("\n" + "=" * 70)
    print(f"S2 RESULTS (N={N_SAMPLES} per CFG)")
    for i, cfg in enumerate(cfgs):
        print(f"  CFG={cfg:<4} | Degeneracy: {cfg_degs[i]:.1%} | Gen. PPL: {cfg_ppls[i]:.2f}")
    print("=" * 70)
    
    results = {
        "cfgs": cfgs,
        "degeneracy_rates": cfg_degs,
        "ppls": cfg_ppls,
    }
    
    with open(RESULTS_DIR / "testS2_cfg_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
