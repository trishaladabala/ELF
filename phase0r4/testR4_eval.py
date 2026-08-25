#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
import math
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0r.testR1_eval import generate_and_track, detect_degeneracy, proportion_confint, compute_ppl_distilgpt2, bootstrap_ci, bootstrap_ci_paired_diff, DummyConfig
from phase0j.testJ1_trajectory_curvature import compute_turning_angles

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

def evaluate_model(ckpt_name, label, n_samples=768):
    print(f"\n{'='*40}")
    print(f"Evaluating {label} ({ckpt_name})")
    print(f"{'='*40}")
    
    ckpt = torch.load(RESULTS_DIR / ckpt_name, map_location=DEVICE, weights_only=False)
    c = ckpt["config"]
    from modules.model import ELF
    model = ELF(
        text_encoder_dim=c["encoder_dim"],
        max_length=c["max_length"],
        hidden_size=c["hidden_size"],
        depth=c["depth"],
        num_heads=c["num_heads"],
        mlp_ratio=c["mlp_ratio"],
        bottleneck_dim=c["bottleneck_dim"],
        num_time_tokens=2,
        num_self_cond_cfg_tokens=0,
        num_model_mode_tokens=2,
        vocab_size=c["vocab_size"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("Generating ODE samples...")
    torch.manual_seed(42)
    ode_texts, ode_traj = generate_and_track(model, n_samples=n_samples, use_sde=False)
    
    print("Generating SDE samples...")
    torch.manual_seed(42)
    sde_texts, sde_traj = generate_and_track(model, n_samples=n_samples, use_sde=True)
    
    ode_deg = detect_degeneracy(ode_texts)
    sde_deg = detect_degeneracy(sde_texts)
    
    ode_deg_rate = ode_deg.mean()
    sde_deg_rate = sde_deg.mean()
    
    print(f"ODE Degeneracy: {ode_deg_rate:.1%} CI: {proportion_confint(ode_deg.sum(), n_samples)}")
    print(f"SDE Degeneracy: {sde_deg_rate:.1%} CI: {proportion_confint(sde_deg.sum(), n_samples)}")
    
    print("Computing PPL...")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    eval_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    eval_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE)
    eval_model.eval()
    
    ode_ppls = compute_ppl_distilgpt2(ode_texts, eval_model, eval_tokenizer, DEVICE)
    sde_ppls = compute_ppl_distilgpt2(sde_texts, eval_model, eval_tokenizer, DEVICE)
    
    valid = ~(ode_deg | sde_deg)
    
    # Compute curvature as a diagnostic
    ode_curv = compute_turning_angles(ode_traj)
    
    if valid.sum() > 10:
        valid_ode_ppls = ode_ppls[valid]
        valid_sde_ppls = sde_ppls[valid]
        
        log_ode = np.log(valid_ode_ppls)
        log_sde = np.log(valid_sde_ppls)
        
        diff = log_sde - log_ode
        diff_ci = bootstrap_ci(diff)
        
        print(f"Valid samples (neither degenerate): {valid.sum()}/{n_samples}")
        print(f"ODE log-PPL: {log_ode.mean():.4f}")
        print(f"SDE log-PPL: {log_sde.mean():.4f}")
        print(f"SDE quality penalty (log-PPL diff): {diff.mean():.4f} CI: {diff_ci}")
    else:
        diff_ci = [0, 0]
        print("Too few valid samples for PPL comparison.")
        
    return {
        "ode_deg_rate": float(ode_deg_rate),
        "sde_deg_rate": float(sde_deg_rate),
        "ode_log_ppl": float(log_ode.mean()) if valid.sum() > 10 else None,
        "sde_log_ppl": float(log_sde.mean()) if valid.sum() > 10 else None,
        "sde_penalty": float(diff.mean()) if valid.sum() > 10 else None,
        "sde_penalty_ci": [float(x) for x in diff_ci],
        "avg_ode_curvature": float(ode_curv.mean()),
        "ode_log_ppls": log_ode if valid.sum() > 10 else None,
        "sde_log_ppls": log_sde if valid.sum() > 10 else None
    }

def main():
    res_ctrl = evaluate_model("testR4_model_control.pt", "Control Model", n_samples=768)
    res_reg = evaluate_model("testR4_model_consistency_reg.pt", "Consistency-Smoothed", n_samples=768)
    
    out_path = RESULTS_DIR / "testR4_consistency_smoothing.json"
    save_dict = {
        "control": {k: v for k, v in res_ctrl.items() if k not in ["ode_log_ppls", "sde_log_ppls"]},
        "consistency": {k: v for k, v in res_reg.items() if k not in ["ode_log_ppls", "sde_log_ppls"]},
    }
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
        
    print(f"\nSaved R4 evaluation to {out_path}")
    
    diffs_ctrl = res_ctrl["sde_log_ppls"] - res_ctrl["ode_log_ppls"]
    diffs_reg = res_reg["sde_log_ppls"] - res_reg["ode_log_ppls"]
    
    mean_diff, lo, hi = bootstrap_ci_paired_diff(diffs_reg, diffs_ctrl)
    
    print("\n--- Diagnostic: Curvature ---")
    print(f"Control Curvature: {res_ctrl['avg_ode_curvature']:.4f}")
    print(f"Consistency Curvature: {res_reg['avg_ode_curvature']:.4f}")
    
    print("\n--- Go/No-Go Criterion Check ---")
    print(f"Control SDE-ODE Gap: {res_ctrl['sde_penalty']:.4f}")
    print(f"Consistency SDE-ODE Gap: {res_reg['sde_penalty']:.4f}")
    print(f"Difference (Cons - Control): {mean_diff:.4f} CI: ({lo:.4f}, {hi:.4f})")

if __name__ == "__main__":
    main()
