#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0t.testT5_rseries_indistribution import generate_tokens, load_shared_model, load_decoupled_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_DIR = Path(__file__).resolve().parent.parent

def get_degenerate_indices(tokens):
    is_deg = np.array([len(set(int(t) for t in tok)) <= 2 for tok in tokens])
    return set(np.where(is_deg)[0].tolist())

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_samples = 256
    
    print("Extracting degenerate indices for Control, R3, and R5...")
    
    ctrl_path = BASE_DIR / "phase0r" / "results" / "testR2_model_control.pt"
    r3_path = BASE_DIR / "phase0r3" / "results" / "testR3_model_label_smooth.pt"
    r5_path = BASE_DIR / "phase0r5" / "results" / "testR5_model_decoupled.pt"
    
    ctrl_model = load_shared_model(ctrl_path)
    r3_model = load_shared_model(r3_path)
    r5_model = load_decoupled_model(r5_path)
    
    ctrl_tokens = generate_tokens(ctrl_model, n_samples=n_samples, n_steps=8, seq_len=16, time_schedule="logit_normal")
    r3_tokens = generate_tokens(r3_model, n_samples=n_samples, n_steps=8, seq_len=16, time_schedule="logit_normal")
    r5_tokens = generate_tokens(r5_model, n_samples=n_samples, n_steps=8, seq_len=16, time_schedule="logit_normal")
    
    ctrl_ind = get_degenerate_indices(ctrl_tokens)
    r3_ind = get_degenerate_indices(r3_tokens)
    r5_ind = get_degenerate_indices(r5_tokens)
    
    always_fails = r3_ind.intersection(r5_ind)
    r3_only = r3_ind - r5_ind
    r5_only = r5_ind - r3_ind
    sometimes_fails = r3_only.union(r5_only)
    
    all_indices = set(range(n_samples))
    never_fails = all_indices - ctrl_ind - r3_ind - r5_ind
    
    out_data = {
        "always_fails": sorted(list(always_fails)),
        "sometimes_fails": sorted(list(sometimes_fails)),
        "never_fails": sorted(list(never_fails)),
        "control_fails": sorted(list(ctrl_ind)),
        "r3_fails": sorted(list(r3_ind)),
        "r5_fails": sorted(list(r5_ind))
    }
    
    out_path = RESULTS_DIR / "testF1a_seed_groups.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Saved to {out_path}")
    print(f"  Always-fails: {len(always_fails)}")
    print(f"  Sometimes-fails: {len(sometimes_fails)}")
    print(f"  Never-fails: {len(never_fails)}")

if __name__ == "__main__":
    main()
