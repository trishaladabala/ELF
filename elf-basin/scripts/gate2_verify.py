#!/usr/bin/env python3
"""Gate 2 verification: Analyzes harvested trajectory states.
Checks:
- Per-phase norm distribution of r_s
- Top principal directions of r_s
- Cosine similarity with isotropic noise and decoder gradients
"""
import sys
import os
import torch
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.sampling.sensitivity import ProbeDirectionGenerator

def decode_with_grad(wrapper, x):
    B = x.shape[0]
    t_final = torch.ones((B,), dtype=x.dtype, device=x.device)
    sc_batch = (
        torch.full((B,), 1.0, dtype=x.dtype, device=x.device)
        if wrapper.config.num_self_cond_cfg_tokens > 0 else None
    )
    if wrapper.config.self_cond_prob > 0 or wrapper.config.num_self_cond_cfg_tokens > 0:
        x_input = torch.cat([x, torch.zeros_like(x)], dim=-1)
    else:
        x_input = x
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=wrapper.use_bf16):
        _, decoder_logits = wrapper.model(
            x_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    return decoder_logits.float()

def analyze_residuals(step, r_data, x_hat_data, wrapper, num_samples=256):
    """Analyze the residuals for a single step."""
    # Subsample for speed
    B, L, D = r_data.shape
    b_idx = min(B, num_samples)
    r = r_data[:b_idx].reshape(-1, D).astype(np.float32)
    x_hat = x_hat_data[:b_idx]
    
    # 1. Norm distribution
    norms = np.linalg.norm(r, axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    
    # 2. PCA
    r_centered = r - r.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(r_centered, full_matrices=False)
    # Effective rank (entropy of normalized S spectrum)
    S_norm = S / (S.sum() + 1e-12)
    entropy = float(-(S_norm * np.log(S_norm + 1e-12)).sum())
    eff_rank = float(np.exp(entropy))
    
    # 3. Cosines
    # (a) Isotropic
    iso = np.random.randn(*r.shape).astype(np.float32)
    iso = iso / np.linalg.norm(iso, axis=1, keepdims=True)
    r_normed = r / (norms[:, None] + 1e-12)
    cos_iso = np.abs((r_normed * iso).sum(axis=1))
    
    # (b) Decoder gradient
    x_hat_t = torch.from_numpy(x_hat.astype(np.float32)).to(wrapper.device)
    
    # Process in batches to avoid OOM
    batch_size = 16
    grad_probes_list = []
    for i in range(0, b_idx, batch_size):
        batch_x = x_hat_t[i:i+batch_size]
        grad_batch = ProbeDirectionGenerator.decoder_grad(
            batch_x, lambda x: decode_with_grad(wrapper, x)
        )[0].cpu().numpy().reshape(-1, D)
        grad_probes_list.append(grad_batch)
    
    grad_probes = np.concatenate(grad_probes_list, axis=0)
    
    cos_grad = np.abs((r_normed * grad_probes).sum(axis=1))
    
    return {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "eff_rank": eff_rank,
        "cos_iso_mean": float(cos_iso.mean()),
        "cos_grad_mean": float(cos_grad.mean())
    }

def main():
    print("="*70)
    print("  Gate 2: Trajectory State Harvesting Verification")
    print("="*70)
    
    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_deen"
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist. Run harvest_deen.py first.")
        return 1
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELFWrapper("ELF-B-de-en", device=device)
    
    steps = [8, 16, 24, 32, 40, 48, 56, 62]
    results = {}
    
    shape = (3000, 128, 512)
    
    for step in steps:
        step_dir = RESULTS_DIR / f"step_{step}"
        if not step_dir.exists():
            continue
            
        print(f"Analyzing step {step}...")
        r_path = step_dir / "r.npy"
        x_hat_path = step_dir / "x_hat.npy"
        
        r_data = np.lib.format.open_memmap(r_path, mode='r')
        x_hat_data = np.lib.format.open_memmap(x_hat_path, mode='r')
        
        stats = analyze_residuals(step, r_data, x_hat_data, wrapper)
        results[step] = stats
        
        print(f"  Norm: {stats['norm_mean']:.2f} ± {stats['norm_std']:.2f}")
        print(f"  Eff Rank: {stats['eff_rank']:.1f} / 512")
        print(f"  Cos(Iso): {stats['cos_iso_mean']:.4f}")
        print(f"  Cos(Grad): {stats['cos_grad_mean']:.4f}")
        
    # Check if statistically distinguishable from isotropic
    # In 512 dimensions, mean abs cosine of two random vectors is ~1/sqrt(512) ≈ 0.044
    # If cos_grad > 0.1, it's highly anisotropic.
    is_anisotropic = any(s['cos_grad_mean'] > 0.06 for s in results.values())
    
    print("\n" + "="*70)
    print("  GATE 2 — VERIFICATION RESULTS")
    print("="*70)
    print(f"  Anisotropic residuals: {'Yes' if is_anisotropic else 'No'}")
    
    if is_anisotropic:
        print("  ✓ GATE 2 PASSED — Residuals are structured and track the margin.")
    else:
        print("  ✗ GATE 2 FAILED — Residuals are indistinguishable from isotropic noise.")
        print("    Track A (sensitivity regularization) may lack motivation.")
        
    with open(RESULTS_DIR / "gate2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return 0 if is_anisotropic else 1
    
if __name__ == "__main__":
    sys.exit(main())
