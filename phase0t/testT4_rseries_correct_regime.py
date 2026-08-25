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
        text_encoder_dim=c["encoder_dim"], max_length=128, hidden_size=c["hidden_size"],
        depth=c["depth"], num_heads=c["num_heads"], mlp_ratio=c["mlp_ratio"], bottleneck_dim=c["bottleneck_dim"],
        num_time_tokens=2, num_self_cond_cfg_tokens=0, num_model_mode_tokens=2, vocab_size=c["vocab_size"],
    ).to(DEVICE)
    decoder = ELF(
        text_encoder_dim=c["encoder_dim"], max_length=128, hidden_size=c["hidden_size"],
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
        text_encoder_dim=c["encoder_dim"], max_length=128, hidden_size=c["hidden_size"],
        depth=c["depth"], num_heads=c["num_heads"], mlp_ratio=c["mlp_ratio"], bottleneck_dim=c["bottleneck_dim"],
        num_time_tokens=2, num_self_cond_cfg_tokens=0, num_model_mode_tokens=2, vocab_size=c["vocab_size"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def generate_texts(model, n_samples=256, n_steps=8, seq_len=128, time_schedule="logit_normal"):
    encoder_dim = 128
    batch_size = 64
    all_texts = []
    
    detok_map_path = BASE_DIR / "phase0e" / "results" / "detokenizer_map.json"
    with open(detok_map_path, "r") as f:
        d_map = json.load(f)
    detok_map = {int(k): v for k, v in d_map.items()}
    
    def decode_tokens_to_text(token_ids):
        words = []
        for tid in token_ids:
            tid = int(tid)
            if tid in detok_map:
                word = detok_map[tid]
                if word.startswith(' '): words.append(' ' + word[1:])
                else: words.append(word)
        return "".join(words).replace("  ", " ").strip()
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            bsz = min(batch_size, n_samples - start_idx)
            torch.manual_seed(42 + start_idx)
            noise = torch.randn(bsz, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)
            
            z_ode = ode_sample(model, bsz, n_steps, encoder_dim, seq_len,
                               time_schedule=time_schedule, device=DEVICE, initial_noise=noise)
            
            tokens = get_tokens(model, z_ode, DEVICE)
            for i in range(bsz):
                all_texts.append(decode_tokens_to_text(tokens[i]))
                
    return all_texts

def evaluate_degeneracy(model, label, n_samples=256):
    print(f"\nEvaluating {label}...")
    texts = generate_texts(model, n_samples=n_samples, n_steps=8, seq_len=128, time_schedule="logit_normal")
    is_deg = detect_degeneracy(texts)
    rate = is_deg.mean()
    ci = proportion_confint(is_deg.sum(), n_samples)
    print(f"  ODE Degeneracy: {rate:.2%} [{ci[0]:.2%}, {ci[1]:.2%}] (count: {is_deg.sum()}/{n_samples})")
    return rate, ci

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "testT4_rseries_correct_regime.md"
    n_samples = 256
    
    print("="*60)
    print("T4 — R-Series Re-evaluation in Degenerate Regime")
    print("Regime: ODE, logit_normal schedule, 8 steps, L=128")
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
        "# Part T4: R-Series Re-evaluation in Degenerate Regime",
        "",
        "**Goal:** Determine whether the R1, R3, and R5 interventions reduce ODE degeneracy when evaluated under the regime where it actually occurs (`logit_normal`, 8 steps, `L=128`).",
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
    if reduced_any:
        out_lines.append("At least one intervention successfully reduced degeneracy at the boundary!")
    else:
        out_lines.append("None of the training-time interventions (R1, R3, R5) successfully reduced ODE degeneracy compared to the control under the unstable boundary regime. This confirms that these training interventions do not fundamentally solve the degeneracy collapse.")
        
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        
    print(f"\nSaved report to {out_path}")

if __name__ == "__main__":
    main()
