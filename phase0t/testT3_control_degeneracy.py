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

from phase0r.testR1_eval import generate_and_track, detect_degeneracy, proportion_confint, DummyConfig
from modules.model import ELF

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

def load_toy_model(ckpt_path):
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

def main():
    n_samples = 2048 # High power to definitively rule out Wilson interval artifact
    control_path = Path(__file__).resolve().parent.parent / "phase0r" / "results" / "testR2_model_control.pt"
    
    print(f"Rerunning ODE Degeneracy on Control Model: {control_path}")
    print(f"N={n_samples} samples")
    
    model = load_toy_model(control_path)
    
    torch.manual_seed(1234)
    # Using the R1-R5 eval config: uniform, 32 steps, len=16
    ode_texts, _ = generate_and_track(model, n_samples=n_samples, n_steps=32, use_sde=False)
    
    ode_deg = detect_degeneracy(ode_texts)
    ode_deg_rate = ode_deg.mean()
    
    print(f"ODE Degeneracy: {ode_deg_rate:.2%} (count: {ode_deg.sum()}/{n_samples})")
    ci = proportion_confint(ode_deg.sum(), n_samples)
    print(f"95% CI: ({ci[0]:.4%}, {ci[1]:.4%})")

if __name__ == "__main__":
    main()
