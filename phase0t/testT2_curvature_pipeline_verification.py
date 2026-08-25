#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
import math
from pathlib import Path
import hashlib
import os

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0r.testR1_eval import generate_and_track
from phase0j.testJ1_trajectory_curvature import compute_turning_angles

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_toy_model_from_path(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    return model

def compute_model_checksum(model):
    """Compute a simple checksum of model weights to prove they differ."""
    total = 0.0
    for p in model.parameters():
        total += float(p.data.sum())
    return f"{total:.10f}"

def evaluate_curvature(ckpt_path, label, n_samples=256):
    print(f"\n{'='*60}")
    print(f"Evaluating {label}")
    print(f"Path: {ckpt_path}")
    
    model = load_toy_model_from_path(ckpt_path)
    checksum = compute_model_checksum(model)
    print(f"Model Parameter Checksum (Sum): {checksum}")
    
    # We verify that we are indeed running generation on THIS model:
    torch.manual_seed(42)
    print("Generating ODE samples to trace trajectory (not cached)...")
    ode_texts, ode_traj = generate_and_track(model, n_samples=n_samples, use_sde=False, n_steps=32)
    
    # Compute curvature
    ode_curv = compute_turning_angles(ode_traj)
    
    mean_curv = float(ode_curv.mean())
    std_curv = float(ode_curv.std())
    
    print(f"Curvature (high precision): {mean_curv:.10f} +- {std_curv:.10f}")
    
    return {
        "checksum": checksum,
        "mean_curv": mean_curv,
        "std_curv": std_curv,
    }

def main():
    base_dir = Path(__file__).resolve().parent.parent
    
    # Ensure R2 w=5.0 exists
    w5_path = base_dir / "phase0r" / "results" / "testR2_model_reg_5.0.pt"
    if not w5_path.exists():
        print("Training R2 model with weight 5.0 to fulfill 'from Part T' requirement...")
        from phase0r.testR2_curvature_reg import finetune_toy_model, prepare_real_data
        embeddings, token_ids, vocab_size = prepare_real_data()
        finetune_toy_model(embeddings, token_ids, vocab_size, 5.0, "testR2_model_reg_5.0.pt")
    
    models_to_eval = [
        ("Control Model", base_dir / "phase0r" / "results" / "testR2_model_control.pt"),
        ("R2 Curvature (Weight=5.0)", base_dir / "phase0r" / "results" / "testR2_model_reg_5.0.pt"),
        ("R4 Consistency", base_dir / "phase0r4" / "results" / "testR4_model_consistency_reg.pt")
    ]
    
    out_lines = [
        "# Part T2: Curvature Pipeline Verification",
        "",
        "This diagnostic proves whether the trajectory curvature metric actually reflects the unique geometries of the fine-tuned models, or if it is invariant/cached.",
        ""
    ]
    
    for label, path in models_to_eval:
        res = evaluate_curvature(path, label, n_samples=256)
        out_lines.extend([
            f"### {label}",
            f"- **Weights Checksum**: `{res['checksum']}`",
            f"- **Trajectory Mean Curvature**: `{res['mean_curv']:.10f}`",
            ""
        ])
        
    out_md = base_dir / "phase0t" / "results" / "testT2_curvature_pipeline_verification.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(out_lines))
        
    print(f"\nSaved report to {out_md}")

if __name__ == "__main__":
    main()
