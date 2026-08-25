#!/usr/bin/env python3
"""J2 — The Actual Hypothesis Test: Does Curvature Predict Where SDE Beats ODE?

Computes the correlation between the ODE trajectory curvature (from J1) and
the per-sample quality gap (PPL_ODE - PPL_SDE) to test if trajectory geometry
explains the sampler behavior.
"""

import json
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    map_path = E_RESULTS / "detokenizer_map.json"
    with open(map_path, "r") as f:
        d_map = json.load(f)
    return {int(k): v for k, v in d_map.items()}


def decode_tokens_to_text(token_ids, detok_map):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid in detok_map:
            word = detok_map[tid]
            if word.startswith(' '):
                words.append(' ' + word[1:])
            else:
                words.append(word)
    return "".join(words).replace("  ", " ").strip()


def setup_external_lm():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"Loading distilgpt2 on {DEVICE}...")
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE).eval()
    return model, tokenizer


def score_text(text, model, tokenizer, device=DEVICE):
    if not text.strip():
        return float('inf')
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


def is_degenerate(token_ids):
    unique = len(set(int(t) for t in token_ids))
    return unique <= 2


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("J2 — Curvature vs SDE Advantage Correlation")
    print("=" * 60)

    # 1. Load Curvature
    with open(RESULTS_DIR / "testJ1_trajectory_curvature.json", "r") as f:
        j1_data = json.load(f)
    ode_curvature = np.array(j1_data["ode_curvature"])
    
    # 2. Load Trajectories to get final z
    ode_traj = np.load(I_RESULTS / "testI2_ode_trajectories.npy")
    sde_traj = np.load(I_RESULTS / "testI2_sde_trajectories.npy")
    z_ode = torch.tensor(ode_traj[-1], dtype=torch.float32, device=DEVICE)
    z_sde = torch.tensor(sde_traj[-1], dtype=torch.float32, device=DEVICE)
    N = z_ode.shape[0]

    # 3. Decode and score
    elf_model = load_real_checkpoint()
    detok_map = load_detokenizer()
    lm_model, lm_tokenizer = setup_external_lm()
    
    ode_ppls = []
    sde_ppls = []
    
    print("Scoring final sequences...")
    BATCH_SIZE = 64
    for batch_start in range(0, N, BATCH_SIZE):
        bs = min(BATCH_SIZE, N - batch_start)
        
        to = get_tokens(elf_model, z_ode[batch_start:batch_start+bs], DEVICE)
        ts = get_tokens(elf_model, z_sde[batch_start:batch_start+bs], DEVICE)
        
        for i in range(bs):
            if is_degenerate(to[i]):
                po = 50000.0
            else:
                po = score_text(decode_tokens_to_text(to[i], detok_map), lm_model, lm_tokenizer, DEVICE)
                if math.isnan(po) or math.isinf(po): po = 50000.0
                
            if is_degenerate(ts[i]):
                ps = 50000.0
            else:
                ps = score_text(decode_tokens_to_text(ts[i], detok_map), lm_model, lm_tokenizer, DEVICE)
                if math.isnan(ps) or math.isinf(ps): ps = 50000.0
                
            ode_ppls.append(po)
            sde_ppls.append(ps)

    ode_ppls = np.array(ode_ppls)
    sde_ppls = np.array(sde_ppls)
    
    # 4. Compute gap and correlate
    # gap > 0 means SDE is better (PPL_ODE is higher than PPL_SDE)
    gap = ode_ppls - sde_ppls
    
    # Since PPL can have extreme outliers, we use Spearman (rank) correlation 
    # to be robust, and also Pearson on clipped/log values.
    
    spearman_corr, spearman_p = spearmanr(ode_curvature, gap)
    
    # For Pearson, we'll clip the gap to avoid massive outliers driving the line
    gap_clipped = np.clip(gap, np.percentile(gap, 5), np.percentile(gap, 95))
    pearson_corr, pearson_p = pearsonr(ode_curvature, gap_clipped)
    
    print(f"\n  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.1e})")
    print(f"  Pearson correlation (clipped): {pearson_corr:.3f} (p={pearson_p:.1e})")
    
    if (spearman_corr > 0.1 and spearman_p < 0.05) or (pearson_corr > 0.1 and pearson_p < 0.05):
        print("\n✅ GO (Real Finding): Statistically meaningful positive correlation detected.")
        print("   Higher ODE curvature predicts a larger quality advantage for SDE.")
        verdict = "GO"
    else:
        print("\n❌ NO-GO: No meaningful correlation. Geometry does not explain the sampler behavior.")
        verdict = "NO_GO"

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ode_curvature, gap_clipped, alpha=0.5, color="#4C72B0", s=15)
    
    # Fit line
    m, b = np.polyfit(ode_curvature, gap_clipped, 1)
    x_line = np.linspace(ode_curvature.min(), ode_curvature.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#C44E52", lw=2, 
            label=f"Trend (Pearson r={pearson_corr:.3f}, p={pearson_p:.1e})")
            
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("ODE Trajectory Curvature (Average Turning Angle)")
    ax.set_ylabel("SDE Advantage (ODE PPL - SDE PPL)\nPositive = SDE Better")
    ax.set_title("J2: Does Curvature Predict SDE Advantage?", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testJ2_curvature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    with open(RESULTS_DIR / "testJ2_curvature_quality_correlation.json", "w") as f:
        json.dump({
            "spearman_corr": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "pearson_corr": float(pearson_corr),
            "pearson_p": float(pearson_p),
            "verdict": verdict
        }, f, indent=2)
        
    print(f"  Saved plot to {RESULTS_DIR / 'testJ2_curvature_correlation.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
