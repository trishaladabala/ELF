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

from phase0r.testR1_eval import detect_degeneracy, proportion_confint, load_toy_model
from phase0r5.testR5_eval import DecoupledELF
from phase0i.testI2_final_result import ode_sample, get_tokens
from modules.model import ELF

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_DIR = Path(__file__).resolve().parent.parent

def load_decoupled_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    return load_toy_model(ckpt_path)

def generate_tokens(model, n_samples=256, n_steps=8, seq_len=16, time_schedule="logit_normal"):
    encoder_dim = 128
    batch_size = 64
    all_tokens = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            bsz = min(batch_size, n_samples - start_idx)
            torch.manual_seed(42 + start_idx)
            noise = torch.randn(bsz, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)
            
            z_ode = ode_sample(model, bsz, n_steps, encoder_dim, seq_len,
                               time_schedule=time_schedule, device=DEVICE, initial_noise=noise)
            
            tokens = get_tokens(model, z_ode, DEVICE)
            for i in range(bsz):
                all_tokens.append(tokens[i])
                
    return all_tokens

def evaluate_degeneracy(model, label, n_samples=256):
    print(f"\nEvaluating {label}...")
    tokens = generate_tokens(model, n_samples=n_samples, n_steps=8, seq_len=16, time_schedule="logit_normal")
    is_deg = np.array([len(set(int(t) for t in tok)) <= 2 for tok in tokens])
    rate = is_deg.mean()
    ci = proportion_confint(is_deg.sum(), n_samples)
    print(f"  ODE Degeneracy: {rate:.2%} [{ci[0]:.2%}, {ci[1]:.2%}] (count: {is_deg.sum()}/{n_samples})")
    return rate, ci

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "testT5_rseries_indistribution.md"
    n_samples = 256
    
    print("="*60)
    print("T5 — R-Series Re-evaluation In-Distribution")
    print("Regime: ODE, logit_normal schedule, 8 steps, L=16")
    print(f"N = {n_samples}")
    print("="*60)
    
    models = {
        "Control": (BASE_DIR / "phase0r" / "results" / "testR2_model_control.pt", False),
        "R1 (Density Reweighting)": (BASE_DIR / "phase0r" / "results" / "testR1_model_over.pt", False),
        "R3 (Label Smoothing)": (BASE_DIR / "phase0r3" / "results" / "testR3_model_label_smooth.pt", False),
        "R3 (Entropy Regularization)": (BASE_DIR / "phase0r3" / "results" / "testR3_model_entropy_reg.pt", False),
        "R5 (Decoupled Architecture)": (BASE_DIR / "phase0r5" / "results" / "testR5_model_decoupled.pt", True)
    }
    
    results = {}
    control_rate = None
    
    for label, (path, is_decoupled) in models.items():
        if is_decoupled:
            model = load_decoupled_model(path)
        else:
            model = load_shared_model(path)
            
        rate, ci = evaluate_degeneracy(model, label, n_samples)
        results[label] = {"rate": rate, "ci": ci}
        if label == "Control":
            control_rate = rate
            
    out_lines = [
        "# Part T5: R-Series Re-evaluation In-Distribution",
        "",
        "**Goal:** Re-run the R-series degeneracy comparison at `L=16` (the sequence length the checkpoints were actually fine-tuned for), but under the unstable `logit_normal`/8-step schedule.",
        "",
        "## Results (N=256)",
        "| Intervention | ODE Degeneracy | 95% CI | vs Control |",
        "| :--- | :--- | :--- | :--- |"
    ]
    
    for label, data in results.items():
        rate = data["rate"]
        ci = data["ci"]
        vs_ctrl = "N/A"
        if label != "Control":
            if rate < control_rate and ci[1] < results["Control"]["ci"][0]:
                vs_ctrl = "**REDUCED**"
            else:
                vs_ctrl = "No Significant Reduction"
                
        out_lines.append(f"| {label} | {rate:.2%} | [{ci[0]:.2%}, {ci[1]:.2%}] | {vs_ctrl} |")
        
    out_lines.extend([
        "",
        "## Conclusion",
    ])
    
    reduced_any = any(v["rate"] < control_rate and v["ci"][1] < results["Control"]["ci"][0] for k, v in results.items() if k != "Control")
    all_zero = all(v["rate"] == 0 for v in results.values())
    all_hundred = all(v["rate"] == 1.0 for v in results.values())
    
    if all_zero or all_hundred:
        out_lines.append("The evaluations are completely saturated at one end (0% or 100%) even in-distribution.")
        out_lines.append("This points to `L=16` being inherently incapable of resolving partial degeneracy under `logit_normal`/8-steps. The toy setup cannot resolve the degeneracy question for R1/R3/R5, full stop.")
    elif reduced_any:
        out_lines.append("At least one intervention successfully reduced degeneracy at the boundary!")
    else:
        out_lines.append("The eval resolved partial degeneracy rates across conditions, but none of the interventions significantly reduced ODE degeneracy compared to the control.")
        
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        
    print(f"\nSaved report to {out_path}")

if __name__ == "__main__":
    main()
