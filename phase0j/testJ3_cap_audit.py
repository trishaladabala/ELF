#!/usr/bin/env python3
"""Part 2: Cap-audit the J3 entropy-advantage correlation."""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF

RESULTS_DIR = Path(__file__).resolve().parent / "results"
I_RESULTS = Path(__file__).resolve().parent.parent / "phase0i" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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

def decode_entropy(model, z, device):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return entropy.cpu().numpy()

def get_tokens(model, z, device):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        tokens = logits.argmax(dim=-1).cpu().numpy()
    return tokens

def compute_turning_angles_per_token(trajectory):
    v = np.diff(trajectory, axis=0)
    steps, N, seq_len, dim = v.shape
    if steps < 2: return np.zeros((N, seq_len))
    
    v_prev = v[:-1]
    v_curr = v[1:]
    
    dot = np.sum(v_prev * v_curr, axis=-1)
    norm_prev = np.linalg.norm(v_prev, axis=-1)
    norm_curr = np.linalg.norm(v_curr, axis=-1)
    
    cos_sim = dot / (norm_prev * norm_curr + 1e-12)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles = np.arccos(cos_sim)
    return np.mean(angles, axis=0)

def is_degenerate(token_ids):
    return len(set(int(t) for t in token_ids)) <= 2

def main():
    print("=" * 60)
    print("Part 2: Cap-audit the J3 entropy-advantage correlation")
    print("=" * 60)

    ode_traj = np.load(I_RESULTS / "testI2_ode_trajectories.npy")
    sde_traj = np.load(I_RESULTS / "testI2_sde_trajectories.npy")
    z_ode = torch.tensor(ode_traj[-1], dtype=torch.float32, device=DEVICE)
    z_sde = torch.tensor(sde_traj[-1], dtype=torch.float32, device=DEVICE)
    N, seq_len, dim = z_ode.shape

    ode_curv_token = compute_turning_angles_per_token(ode_traj)
    
    elf_model = load_real_checkpoint()
    
    ent_ode_all, ent_sde_all = [], []
    ode_deg_all, sde_deg_all = [], []
    
    BATCH_SIZE = 64
    for batch_start in range(0, N, BATCH_SIZE):
        bs = min(BATCH_SIZE, N - batch_start)
        z_o = z_ode[batch_start:batch_start+bs]
        z_s = z_sde[batch_start:batch_start+bs]
        
        eo = decode_entropy(elf_model, z_o, DEVICE)
        es = decode_entropy(elf_model, z_s, DEVICE)
        ent_ode_all.append(eo)
        ent_sde_all.append(es)
        
        to = get_tokens(elf_model, z_o, DEVICE)
        ts = get_tokens(elf_model, z_s, DEVICE)
        for i in range(bs):
            ode_deg_all.append(is_degenerate(to[i]))
            sde_deg_all.append(is_degenerate(ts[i]))
            
    ent_ode = np.concatenate(ent_ode_all, axis=0)
    ent_sde = np.concatenate(ent_sde_all, axis=0)
    ode_deg = np.array(ode_deg_all)
    sde_deg = np.array(sde_deg_all)
    
    # Identify capped/degenerate
    any_capped_seq_idx = ode_deg | sde_deg
    uncapped_seq_idx = ~any_capped_seq_idx
    
    # Broadcast to token level
    # gap > 0 means SDE has lower entropy / more confident
    gap = ent_ode - ent_sde
    
    any_capped_token_mask = np.repeat(any_capped_seq_idx[:, None], seq_len, axis=1)
    uncapped_token_mask = ~any_capped_token_mask
    
    print(f"Total tokens: {N * seq_len}")
    print(f"Tokens in degenerate sequences: {any_capped_token_mask.sum()} ({(100*any_capped_token_mask.sum()/(N*seq_len)):.1f}%)")
    
    # a) All tokens
    curv_flat = ode_curv_token.flatten()
    gap_flat = gap.flatten()
    sp_all, p_all = spearmanr(curv_flat, gap_flat)
    
    # b) Exclude tokens in degenerate sequences
    curv_excl = ode_curv_token[uncapped_token_mask]
    gap_excl = gap[uncapped_token_mask]
    sp_excl, p_excl = spearmanr(curv_excl, gap_excl)
    
    # c) Winsorize
    # Here, "winsorize" might not perfectly map since entropy isn't explicitly capped at 50000,
    # but the sequence output degenerates, meaning entropy is very low (confident repetition).
    # We will winsorize the entropy gap to the min/max of the non-degenerate subset.
    max_uncapped = gap_excl.max()
    min_uncapped = gap_excl.min()
    
    gap_win = gap.copy()
    pos_cap_idx = np.repeat((ode_deg & ~sde_deg)[:, None], seq_len, axis=1)
    # If ODE degenerated, ODE entropy is very low. So gap = ent_ode - ent_sde is negative.
    # We winsorize to min_uncapped.
    gap_win[pos_cap_idx] = min_uncapped
    
    neg_cap_idx = np.repeat((sde_deg & ~ode_deg)[:, None], seq_len, axis=1)
    gap_win[neg_cap_idx] = max_uncapped
    
    sp_win, p_win = spearmanr(curv_flat, gap_win.flatten())
    
    print("\n--- Correlations (Token-Level) ---")
    print(f"a) All tokens (N={N*seq_len}):")
    print(f"   Spearman: r={sp_all:.3f}, p={p_all:.1e}")
    print(f"b) Exclude degenerate sequences (N={uncapped_token_mask.sum()}):")
    print(f"   Spearman: r={sp_excl:.3f}, p={p_excl:.1e}")
    print(f"c) Winsorize degenerate sequences (N={N*seq_len}):")
    print(f"   Spearman: r={sp_win:.3f}, p={p_win:.1e}")
    
    if sp_excl > 0.1 and p_excl < 0.05:
        verdict = "The strong positive correlation (higher curvature -> greater SDE overconfidence) survives exclusion of degenerate sequences. The mechanism holds generally, not just for catastrophic collapses."
    else:
        verdict = "The correlation vanishes or drops substantially when excluding degenerate sequences. The overconfidence effect is primarily a feature of the sequence collapses."
        
    print(f"\nConclusion: {verdict}")
    
    results = {
        "all_tokens": {"spearman_r": sp_all, "spearman_p": p_all},
        "exclude_capped": {"spearman_r": sp_excl, "spearman_p": p_excl},
        "winsorize_capped": {"spearman_r": sp_win, "spearman_p": p_win},
        "verdict": verdict
    }
    
    with open(RESULTS_DIR / "testJ3_cap_audit.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
