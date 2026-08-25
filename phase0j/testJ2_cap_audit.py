#!/usr/bin/env python3
"""Part 2: Cap-audit for the Idea 2 curvature correlation."""

import json
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF

RESULTS_DIR = Path(__file__).resolve().parent / "results"
I_RESULTS = Path(__file__).resolve().parent.parent / "phase0i" / "results"
E_RESULTS = Path(__file__).resolve().parent.parent / "phase0e" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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


def main():
    print("=" * 60)
    print("Part 2: Cap-audit for Idea 2 curvature correlation")
    print("=" * 60)

    # Load Curvature
    with open(RESULTS_DIR / "testJ1_trajectory_curvature.json", "r") as f:
        j1_data = json.load(f)
    ode_curvature = np.array(j1_data["ode_curvature"])

    # Load trajectories and score
    ode_traj = np.load(I_RESULTS / "testI2_ode_trajectories.npy")
    sde_traj = np.load(I_RESULTS / "testI2_sde_trajectories.npy")
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
    
    # 1. Identify Capped Values
    # Capped points are where either ODE or SDE hit 50000.0
    cap_val = 50000.0
    ode_capped_idx = (ode_ppls == cap_val)
    sde_capped_idx = (sde_ppls == cap_val)
    both_capped_idx = ode_capped_idx & sde_capped_idx
    any_capped_idx = ode_capped_idx | sde_capped_idx
    uncapped_idx = ~any_capped_idx
    
    # Gap: PPL_ODE - PPL_SDE
    gap = ode_ppls - sde_ppls
    
    print(f"Total N: {N}")
    print(f"ODE capped (PPL=50000): {ode_capped_idx.sum()} ({100*ode_capped_idx.sum()/N:.1f}%)")
    print(f"SDE capped (PPL=50000): {sde_capped_idx.sum()} ({100*sde_capped_idx.sum()/N:.1f}%)")
    print(f"Both capped: {both_capped_idx.sum()} ({100*both_capped_idx.sum()/N:.1f}%)")
    print(f"Any capped: {any_capped_idx.sum()} ({100*any_capped_idx.sum()/N:.1f}%)")
    
    # 3. Compute correlations three ways
    # a) All points
    gap_clipped_orig = np.clip(gap, np.percentile(gap, 5), np.percentile(gap, 95))
    sp_all, p_sp_all = spearmanr(ode_curvature, gap)
    pe_all, p_pe_all = pearsonr(ode_curvature, gap_clipped_orig)
    
    # b) Exclude capped points entirely
    sp_excl, p_sp_excl = spearmanr(ode_curvature[uncapped_idx], gap[uncapped_idx])
    gap_excl_clipped = np.clip(gap[uncapped_idx], np.percentile(gap[uncapped_idx], 5), np.percentile(gap[uncapped_idx], 95))
    pe_excl, p_pe_excl = pearsonr(ode_curvature[uncapped_idx], gap_excl_clipped)
    
    # c) Winsorize capped points
    max_uncapped = gap[uncapped_idx].max()
    min_uncapped = gap[uncapped_idx].min()
    gap_win = gap.copy()
    # Positive cap: ODE was 50000
    pos_cap_idx = ode_capped_idx & ~sde_capped_idx
    gap_win[pos_cap_idx] = max_uncapped
    # Negative cap: SDE was 50000
    neg_cap_idx = sde_capped_idx & ~ode_capped_idx
    gap_win[neg_cap_idx] = min_uncapped
    # Both capped: gap is 0, which is fine
    gap_win[both_capped_idx] = 0
    
    sp_win, p_sp_win = spearmanr(ode_curvature, gap_win)
    gap_win_clipped = np.clip(gap_win, np.percentile(gap_win, 5), np.percentile(gap_win, 95))
    pe_win, p_pe_win = pearsonr(ode_curvature, gap_win_clipped)
    
    print("\n--- Correlations ---")
    print(f"a) All points (N={N}):")
    print(f"   Spearman: r={sp_all:.3f}, p={p_sp_all:.1e}")
    print(f"   Pearson:  r={pe_all:.3f}, p={p_pe_all:.1e}")
    print(f"b) Exclude capped (N={uncapped_idx.sum()}):")
    print(f"   Spearman: r={sp_excl:.3f}, p={p_sp_excl:.1e}")
    print(f"   Pearson:  r={pe_excl:.3f}, p={p_pe_excl:.1e}")
    print(f"c) Winsorize capped (N={N}):")
    print(f"   Spearman: r={sp_win:.3f}, p={p_sp_win:.1e}")
    print(f"   Pearson:  r={pe_win:.3f}, p={p_pe_win:.1e}")
    
    # 4. State explicitly
    if p_sp_excl > 0.05 and p_pe_excl > 0.05:
        verdict = "The negative correlation vanishes entirely when capped points are removed. The effect was completely driven by the artificial 50000 PPL cap for degenerate sequences."
    elif (sp_excl > -0.1 and pe_excl > -0.1):
        verdict = "The negative correlation drops below meaningful effect size when capped points are removed. The effect was substantially driven by the capped points."
    else:
        verdict = "The negative correlation direction and statistical significance survive even when capped points are removed or winsorized."
        
    print(f"\nConclusion: {verdict}")
    
    results = {
        "fraction_capped": float(any_capped_idx.sum() / N),
        "fraction_ode_capped": float(ode_capped_idx.sum() / N),
        "fraction_sde_capped": float(sde_capped_idx.sum() / N),
        "fraction_both_capped": float(both_capped_idx.sum() / N),
        "all_points": {"spearman_r": sp_all, "spearman_p": p_sp_all, "pearson_r": pe_all, "pearson_p": p_pe_all},
        "exclude_capped": {"spearman_r": sp_excl, "spearman_p": p_sp_excl, "pearson_r": pe_excl, "pearson_p": p_pe_excl},
        "winsorize_capped": {"spearman_r": sp_win, "spearman_p": p_sp_win, "pearson_r": pe_win, "pearson_p": p_pe_win},
        "verdict": verdict
    }
    
    with open(RESULTS_DIR / "testJ2_cap_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: All points with capped highlighted
    ax = axes[0]
    ax.scatter(ode_curvature[uncapped_idx], gap_clipped_orig[uncapped_idx], color="#4C72B0", alpha=0.5, label="Uncapped", s=15)
    ax.scatter(ode_curvature[any_capped_idx], gap_clipped_orig[any_capped_idx], color="#C44E52", alpha=0.8, marker='x', label="Capped (Degenerate)", s=25)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("ODE Trajectory Curvature")
    ax.set_ylabel("SDE Advantage (ODE - SDE PPL)")
    ax.set_title(f"Original Data (All Points)\nSpearman r={sp_all:.3f}", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Winsorized
    ax = axes[1]
    ax.scatter(ode_curvature[uncapped_idx], gap_win_clipped[uncapped_idx], color="#4C72B0", alpha=0.5, label="Uncapped", s=15)
    ax.scatter(ode_curvature[any_capped_idx], gap_win_clipped[any_capped_idx], color="#55A868", alpha=0.8, marker='s', label="Winsorized", s=20)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("ODE Trajectory Curvature")
    ax.set_ylabel("SDE Advantage (Winsorized)")
    ax.set_title(f"Winsorized Data\nSpearman r={sp_win:.3f}", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testJ2_cap_audit_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
if __name__ == "__main__":
    main()
