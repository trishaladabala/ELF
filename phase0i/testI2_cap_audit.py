#!/usr/bin/env python3
"""Part 1: Cap-audit the I2 log-PPL comparison."""

import json
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import fisher_exact

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF

RESULTS_DIR = Path(__file__).resolve().parent / "results"
E_RESULTS = Path(__file__).resolve().parent.parent / "phase0e" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_BOOTSTRAP = 1000

def load_detokenizer():
    with open(E_RESULTS / "detokenizer_map.json", "r") as f:
        d_map = json.load(f)
    return {int(k): v for k, v in d_map.items()}

def decode_tokens_to_text(token_ids, detok_map):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid in detok_map:
            word = detok_map[tid]
            if word.startswith(' '): words.append(' ' + word[1:])
            else: words.append(word)
    return "".join(words).replace("  ", " ").strip()

def setup_external_lm():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE).eval()
    return model, tokenizer

def score_text(text, model, tokenizer, device=DEVICE):
    if not text.strip(): return float('inf')
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2: return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

def load_real_checkpoint():
    ckpt_path = Path(__file__).resolve().parent.parent / "phase0c" / "results" / "mini_elf_real_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ELF(
        text_encoder_dim=cfg["encoder_dim"], max_length=cfg["max_length"],
        hidden_size=cfg["hidden_size"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        bottleneck_dim=cfg["bottleneck_dim"], num_time_tokens=2,
        num_self_cond_cfg_tokens=0, num_model_mode_tokens=2,
        vocab_size=cfg["vocab_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model

def get_tokens(model, z, device):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        tokens = logits.argmax(dim=-1).cpu().numpy()
    return tokens

def is_degenerate(token_ids):
    return len(set(int(t) for t in token_ids)) <= 2

def bootstrap_paired_ci(log_ode, log_sde, n_boot=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(log_ode)
    diffs = np.array(log_sde) - np.array(log_ode)
    boot_diffs = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)

def main():
    print("=" * 60)
    print("Part 1: Cap-audit for the I2 log-PPL comparison")
    print("=" * 60)

    # 1. Load Data
    ode_traj = np.load(RESULTS_DIR / "testI2_ode_trajectories.npy")
    sde_traj = np.load(RESULTS_DIR / "testI2_sde_trajectories.npy")
    z_ode = torch.tensor(ode_traj[-1], dtype=torch.float32, device=DEVICE)
    z_sde = torch.tensor(sde_traj[-1], dtype=torch.float32, device=DEVICE)
    
    elf_model = load_real_checkpoint()
    detok_map = load_detokenizer()
    lm_model, lm_tokenizer = setup_external_lm()
    
    ode_ppls, sde_ppls = [], []
    N = z_ode.shape[0]
    BATCH_SIZE = 64
    for batch_start in range(0, N, BATCH_SIZE):
        bs = min(BATCH_SIZE, N - batch_start)
        to = get_tokens(elf_model, z_ode[batch_start:batch_start+bs], DEVICE)
        ts = get_tokens(elf_model, z_sde[batch_start:batch_start+bs], DEVICE)
        for i in range(bs):
            po = 50000.0 if is_degenerate(to[i]) else score_text(decode_tokens_to_text(to[i], detok_map), lm_model, lm_tokenizer, DEVICE)
            ps = 50000.0 if is_degenerate(ts[i]) else score_text(decode_tokens_to_text(ts[i], detok_map), lm_model, lm_tokenizer, DEVICE)
            if math.isnan(po) or math.isinf(po): po = 50000.0
            if math.isnan(ps) or math.isinf(ps): ps = 50000.0
            ode_ppls.append(po)
            sde_ppls.append(ps)
            
    ode_ppls = np.array(ode_ppls)
    sde_ppls = np.array(sde_ppls)
    
    # Identify degenerate
    ode_deg = (ode_ppls == 50000.0)
    sde_deg = (sde_ppls == 50000.0)
    
    ode_deg_count = int(ode_deg.sum())
    sde_deg_count = int(sde_deg.sum())
    
    print("\n--- Degeneracy Rates ---")
    print(f"ODE Degeneracy: {ode_deg_count}/{N} ({100*ode_deg_count/N:.1f}%)")
    print(f"SDE Degeneracy: {sde_deg_count}/{N} ({100*sde_deg_count/N:.1f}%)")
    
    # Fisher Exact Test
    table = [[ode_deg_count, N - ode_deg_count], 
             [sde_deg_count, N - sde_deg_count]]
    oddsr, p_val = fisher_exact(table)
    print(f"Fisher's Exact Test p-value: {p_val:.2e}")
    if p_val < 0.05:
        print("✅ The difference in degeneracy rates is statistically significant.")
    else:
        print("❌ The difference is not statistically significant.")

    # Non-degenerate subset
    both_valid = ~ode_deg & ~sde_deg
    n_valid = int(both_valid.sum())
    
    print("\n--- Log-PPL on Non-Degenerate Subset ---")
    print(f"Excluding {N - n_valid} paired samples where either ODE or SDE degenerated.")
    
    log_ode_valid = np.log(ode_ppls[both_valid])
    log_sde_valid = np.log(sde_ppls[both_valid])
    
    diff_mean, diff_lo, diff_hi = bootstrap_paired_ci(log_ode_valid, log_sde_valid)
    red_mean = (1 - math.exp(diff_mean)) * 100
    red_hi = (1 - math.exp(diff_lo)) * 100
    red_lo = (1 - math.exp(diff_hi)) * 100
    excludes_zero = (diff_lo > 0) or (diff_hi < 0)
    
    print(f"Valid ODE log-PPL mean: {np.mean(log_ode_valid):.3f}")
    print(f"Valid SDE log-PPL mean: {np.mean(log_sde_valid):.3f}")
    print(f"Log Paired Δ: {diff_mean:+.3f} [{diff_lo:+.3f}, {diff_hi:+.3f}]")
    print(f"Excludes 0: {excludes_zero}")
    print(f"Multiplicative: SDE reduces PPL by {red_mean:.1f}% [{red_lo:.1f}%, {red_hi:.1f}%]")
    
    if excludes_zero and diff_mean < 0:
        verdict = "The quality advantage for SDE survives on non-degenerate samples alone, though it is smaller than the 19.4% originally reported."
    elif excludes_zero and diff_mean > 0:
        verdict = "On non-degenerate samples, the SDE is significantly WORSE than ODE. The original 19.4% advantage was entirely driven by the degeneracy-rate difference."
    else:
        verdict = "No significant quality advantage for SDE survives when excluding degenerate samples. The original 19.4% advantage was entirely driven by eliminating degeneracy."
    print(f"\nConclusion: {verdict}")
    
    results = {
        "N": N,
        "ode_degenerate_count": ode_deg_count,
        "sde_degenerate_count": sde_deg_count,
        "fisher_exact_p": float(p_val),
        "subset_N": n_valid,
        "subset_log_ode_mean": float(np.mean(log_ode_valid)),
        "subset_log_sde_mean": float(np.mean(log_sde_valid)),
        "subset_log_diff_mean": diff_mean,
        "subset_log_diff_ci_lo": diff_lo,
        "subset_log_diff_ci_hi": diff_hi,
        "subset_excludes_zero": bool(excludes_zero),
        "subset_reduction_pct_mean": red_mean,
        "subset_reduction_pct_ci_lo": red_lo,
        "subset_reduction_pct_ci_hi": red_hi,
        "verdict": verdict
    }
    
    with open(RESULTS_DIR / "testI2_cap_audit.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
