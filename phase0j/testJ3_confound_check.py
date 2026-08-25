#!/usr/bin/env python3
"""J3 — Rule Out Confounds (Position in Sequence).

Since J2 was run entirely within a single step-count bucket (8 steps),
the step-count confound is perfectly controlled by design. 

To satisfy J3's requirement for robustness, we check the sequence-position confound:
We regress per-token curvature against per-token quality gap (using entropy as
the token-level proxy for quality) and verify that the correlation holds
WITHIN individual sequence positions, not just pooled across them.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

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
    
    # Average across steps only, keep per-token (N, seq_len)
    curvature = np.mean(angles, axis=0)
    return curvature

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("J3 — Rule Out Confounds (Position in Sequence)")
    print("=" * 60)

    # 1. Load Trajectories
    ode_traj = np.load(I_RESULTS / "testI2_ode_trajectories.npy")
    sde_traj = np.load(I_RESULTS / "testI2_sde_trajectories.npy")
    
    z_ode = torch.tensor(ode_traj[-1], dtype=torch.float32, device=DEVICE)
    z_sde = torch.tensor(sde_traj[-1], dtype=torch.float32, device=DEVICE)
    N, seq_len, dim = z_ode.shape

    # 2. Get per-token curvature
    ode_curv_token = compute_turning_angles_per_token(ode_traj) # (N, seq_len)
    
    # 3. Get per-token entropy
    elf_model = load_real_checkpoint()
    
    ent_ode_all = []
    ent_sde_all = []
    
    print("Scoring tokens...")
    BATCH_SIZE = 64
    for batch_start in range(0, N, BATCH_SIZE):
        bs = min(BATCH_SIZE, N - batch_start)
        eo = decode_entropy(elf_model, z_ode[batch_start:batch_start+bs], DEVICE)
        es = decode_entropy(elf_model, z_sde[batch_start:batch_start+bs], DEVICE)
        ent_ode_all.append(eo)
        ent_sde_all.append(es)
        
    ent_ode = np.concatenate(ent_ode_all, axis=0) # (N, seq_len)
    ent_sde = np.concatenate(ent_sde_all, axis=0) # (N, seq_len)
    
    # Gap > 0 means SDE is better (has lower entropy / more confidence)
    gap = ent_ode - ent_sde 
    
    # 4. Check correlation overall and per-position
    overall_corr, overall_p = spearmanr(ode_curv_token.flatten(), gap.flatten())
    print(f"\n  Overall per-token correlation: r={overall_corr:.3f}, p={overall_p:.1e}")
    
    pos_corrs = []
    sig_positions = 0
    for pos in range(seq_len):
        c = ode_curv_token[:, pos]
        g = gap[:, pos]
        r, p = spearmanr(c, g)
        pos_corrs.append(r)
        if r > 0.05 and p < 0.05:
            sig_positions += 1
        print(f"    Pos {pos:02d}: r={r:.3f}, p={p:.1e}")

    print(f"\n  Significant positive correlation in {sig_positions}/{seq_len} positions.")
    
    # We require the correlation to hold within positions (at least 50% of them)
    if sig_positions >= seq_len // 2:
        print("✅ PASS: The correlation holds robustly within individual sequence positions.")
        print("   It is not a spurious confound driven by position.")
        verdict = "PASS"
    else:
        print("❌ FAIL: The correlation vanishes when controlling for position.")
        print("   It was likely a spurious confound.")
        verdict = "FAIL"

    with open(RESULTS_DIR / "testJ3_confounds.json", "w") as f:
        json.dump({
            "overall_corr": float(overall_corr),
            "sig_positions": sig_positions,
            "seq_len": seq_len,
            "verdict": verdict,
            "controlled_for_step_count": True, # By design in I2
            "position_correlations": pos_corrs
        }, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
