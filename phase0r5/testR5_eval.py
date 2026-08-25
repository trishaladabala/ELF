#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
import math
import torch.nn as nn
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0r.testR1_eval import generate_and_track, detect_degeneracy, proportion_confint, compute_ppl_distilgpt2, bootstrap_ci, bootstrap_ci_paired_diff, DummyConfig
from modules.model import ELF

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

class DecoupledELF(nn.Module):
    def __init__(self, denoiser, decoder):
        super().__init__()
        self.denoiser = denoiser
        self.decoder = decoder

    def forward(self, x, t, attention_mask=None, deterministic=True, self_cond_cfg_scale=None, decoder_step_active=None):
        denoiser_out, _ = self.denoiser(x, t, attention_mask, deterministic, self_cond_cfg_scale, decoder_step_active=0.0)
        _, decoder_logits = self.decoder(x, t, attention_mask, deterministic, self_cond_cfg_scale, decoder_step_active=1.0)
        return denoiser_out, decoder_logits

def load_decoupled_model(ckpt_name):
    ckpt = torch.load(RESULTS_DIR / ckpt_name, map_location=DEVICE, weights_only=False)
    c = ckpt["config"]
    denoiser = ELF(
        text_encoder_dim=c["encoder_dim"], max_length=c["max_length"], hidden_size=c["hidden_size"],
        depth=c["depth"], num_heads=c["num_heads"], mlp_ratio=c["mlp_ratio"], bottleneck_dim=c["bottleneck_dim"],
        num_time_tokens=2, num_self_cond_cfg_tokens=0, num_model_mode_tokens=2, vocab_size=c["vocab_size"],
    ).to(DEVICE)
    decoder = ELF(
        text_encoder_dim=c["encoder_dim"], max_length=c["max_length"], hidden_size=c["hidden_size"],
        depth=c["depth"], num_heads=c["num_heads"], mlp_ratio=c["mlp_ratio"], bottleneck_dim=c["bottleneck_dim"],
        num_time_tokens=2, num_self_cond_cfg_tokens=0, num_model_mode_tokens=2, vocab_size=c["vocab_size"],
    ).to(DEVICE)
    
    denoiser.load_state_dict(ckpt["denoiser_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    
    model = DecoupledELF(denoiser, decoder)
    model.eval()
    return model

def load_shared_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    c = ckpt["config"]
    model = ELF(
        text_encoder_dim=c["encoder_dim"], max_length=c["max_length"], hidden_size=c["hidden_size"],
        depth=c["depth"], num_heads=c["num_heads"], mlp_ratio=c["mlp_ratio"], bottleneck_dim=c["bottleneck_dim"],
        num_time_tokens=2, num_self_cond_cfg_tokens=0, num_model_mode_tokens=2, vocab_size=c["vocab_size"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def evaluate_model(model, label, n_samples=768):
    print(f"\n{'='*40}")
    print(f"Evaluating {label}")
    print(f"{'='*40}")
    
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
        "ode_log_ppls": log_ode if valid.sum() > 10 else None,
        "sde_log_ppls": log_sde if valid.sum() > 10 else None
    }

def main():
    n_samples = 768
    
    # 1. Evaluate Control Model (Shared weights)
    control_path = Path(__file__).resolve().parent.parent / "phase0r" / "results" / "testR2_model_control.pt"
    control_model = load_shared_model(control_path)
    res_ctrl = evaluate_model(control_model, "Control Model (Shared)", n_samples=n_samples)
    
    # 2. Evaluate Decoupled Model
    decoupled_model = load_decoupled_model("testR5_model_decoupled.pt")
    res_dec = evaluate_model(decoupled_model, "Decoupled Model", n_samples=n_samples)
    
    # Save results
    out_path = RESULTS_DIR / "testR5_decoupled_architecture.json"
    save_dict = {
        "control": {k: v for k, v in res_ctrl.items() if k not in ["ode_log_ppls", "sde_log_ppls"]},
        "decoupled": {k: v for k, v in res_dec.items() if k not in ["ode_log_ppls", "sde_log_ppls"]},
    }
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
        
    print(f"\nSaved R5 evaluation to {out_path}")
    
    diffs_ctrl = res_ctrl["sde_log_ppls"] - res_ctrl["ode_log_ppls"]
    diffs_dec = res_dec["sde_log_ppls"] - res_dec["ode_log_ppls"]
    
    mean_diff, lo, hi = bootstrap_ci_paired_diff(diffs_dec, diffs_ctrl)
    
    print("\n--- Unconditional Quality Check ---")
    print(f"Control ODE log-PPL: {res_ctrl['ode_log_ppl']:.4f}")
    print(f"Decoupled ODE log-PPL: {res_dec['ode_log_ppl']:.4f}")
    print("Sanity Check: Does the decoupled model cost some raw quality (higher log-PPL)?")
    
    print("\n--- Go/No-Go Criterion Check ---")
    print(f"Control SDE-ODE Gap: {res_ctrl['sde_penalty']:.4f}")
    print(f"Decoupled SDE-ODE Gap: {res_dec['sde_penalty']:.4f}")
    print(f"Difference (Decoupled - Control): {mean_diff:.4f} CI: ({lo:.4f}, {hi:.4f})")
    
    excludes_zero = (lo > 0) or (hi < 0)
    if mean_diff < 0 and excludes_zero:
        print("GO: Decoupling improved the SDE vs ODE gap significantly! The trade-off is architectural.")
    else:
        print("NO-GO: Decoupling did not significantly improve the SDE vs ODE gap.")

if __name__ == "__main__":
    main()
