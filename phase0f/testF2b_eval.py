#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
import hashlib
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0t.testT5_rseries_indistribution import generate_tokens, load_decoupled_model, load_shared_model
from phase0r.testR1_eval import proportion_confint
from phase0t.testT6_r3r5_independence_check import compute_checksum

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_DIR = Path(__file__).resolve().parent.parent

def evaluate_degeneracy(model, n_samples=256):
    tokens = generate_tokens(model, n_samples=n_samples, n_steps=8, seq_len=16, time_schedule="logit_normal")
    is_deg = np.array([len(set(int(t) for t in tok)) <= 2 for tok in tokens])
    rate = is_deg.mean()
    ci = proportion_confint(is_deg.sum(), n_samples)
    return rate, ci, tokens

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "testF2b_eval.md"
    n_samples = 256
    
    print("="*60)
    print("F2b — Combined R3+R5 Model Evaluation")
    print("="*60)
    
    combined_path = RESULTS_DIR / "testF2_model_combined.pt"
    r3_path = BASE_DIR / "phase0r3" / "results" / "testR3_model_label_smooth.pt"
    r5_path = BASE_DIR / "phase0r5" / "results" / "testR5_model_decoupled.pt"
    
    print("Loading models and checking checksums...")
    comb_model = load_decoupled_model(combined_path)
    r3_model = load_shared_model(r3_path)
    r5_model = load_decoupled_model(r5_path)
    
    comb_chk = compute_checksum(comb_model)
    r3_chk = compute_checksum(r3_model)
    r5_chk = compute_checksum(r5_model)
    
    print(f"Combined Checksum: {comb_chk}")
    print(f"R3 Checksum: {r3_chk}")
    print(f"R5 Checksum: {r5_chk}")
    
    assert comb_chk != r3_chk, "Combined model is identical to R3!"
    assert comb_chk != r5_chk, "Combined model is identical to R5!"
    
    print("\nEvaluating combined model...")
    rate, ci, tokens = evaluate_degeneracy(comb_model, n_samples)
    print(f"Combined ODE Degeneracy: {rate:.2%} [{ci[0]:.2%}, {ci[1]:.2%}]")
    
    # Extract degenerate indices for F2c
    is_deg = np.array([len(set(int(t) for t in tok)) <= 2 for tok in tokens])
    deg_indices = np.where(is_deg)[0].tolist()
    
    with open(RESULTS_DIR / "testF2_combined_fails.json", "w") as f:
        json.dump({"combined_fails": deg_indices}, f, indent=2)
        
    # The previous results for comparison
    ctrl_rate, ctrl_ci = 0.2031, [0.1584, 0.2566]
    r3_rate, r3_ci = 0.0977, [0.0670, 0.1402]
    r5_rate, r5_ci = 0.0977, [0.0670, 0.1402]
    
    out_lines = [
        "# Part F2b: Combined R3+R5 Evaluation",
        "",
        "## Independence Verification",
        f"- **Combined Model Checksum:** `{comb_chk}`",
        f"- **R3 (Label Smooth) Checksum:** `{r3_chk}`",
        f"- **R5 (Decoupled) Checksum:** `{r5_chk}`",
        "**Conclusion:** The combined model is genuinely distinct from both individual models.",
        "",
        "## Results (N=256, L=16, 8-steps logit_normal)",
        "| Model | ODE Degeneracy | 95% CI |",
        "| :--- | :--- | :--- |",
        f"| Control | {ctrl_rate:.2%} | [{ctrl_ci[0]:.2%}, {ctrl_ci[1]:.2%}] |",
        f"| R3 (Label Smoothing) | {r3_rate:.2%} | [{r3_ci[0]:.2%}, {r3_ci[1]:.2%}] |",
        f"| R5 (Decoupled) | {r5_rate:.2%} | [{r5_ci[0]:.2%}, {r5_ci[1]:.2%}] |",
        f"| **Combined (R3 + R5)** | **{rate:.2%}** | **[{ci[0]:.2%}, {ci[1]:.2%}]** |",
        ""
    ]
    
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
        
    print(f"\nSaved report to {out_path}")
    print(f"Saved indices to {RESULTS_DIR / 'testF2_combined_fails.json'}")

if __name__ == "__main__":
    main()
