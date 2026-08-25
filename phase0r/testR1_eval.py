#!/usr/bin/env python3
"""R1 — Evaluate Training-Density Fix models.

Evaluates ODE/SDE degeneracy rate, log-PPL paired bootstrap for non-degenerate
samples, and curvature-overconfidence correlation for both standard and oversampled
toy models.
"""

import json
import sys
import numpy as np
import torch
import math
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.model import ELF
from utils.sampling_utils import _ode_step, _sde_step, get_sampling_steps
from phase0e.testE1_external_quality import load_detokenizer
from phase0j.testJ1_trajectory_curvature import compute_turning_angles
import scipy.stats

def proportion_confint(count, nobs, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2)
    p = count / nobs
    q = 1 - p
    denom = 1 + z**2 / nobs
    center = (p + z**2 / (2 * nobs)) / denom
    spread = z * math.sqrt(p * q / nobs + z**2 / (4 * nobs**2)) / denom
    return (center - spread, center + spread)

def bootstrap_ci_paired_diff(gap_A, gap_B, n_boot=1000, alpha=0.05):
    # gap_A and gap_B are parallel arrays of the same size. 
    # we want the CI of mean(gap_A - gap_B)
    rng = np.random.default_rng(42)
    n = len(gap_A)
    diffs = gap_A - gap_B
    boot_diffs = np.array([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)

def bootstrap_ci(diffs, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(diffs)
    boot_diffs = np.array([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def compute_ppl_distilgpt2(texts, model, tokenizer, device):
    ppls = []
    for text in texts:
        if not text.strip(): 
            ppls.append(50000.0)
            continue
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        if input_ids.shape[1] < 2:
            ppls.append(50000.0)
            continue
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        val = math.exp(loss.item())
        if math.isnan(val) or math.isinf(val):
            val = 50000.0
        ppls.append(val)
    return np.array(ppls)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class DummyConfig:
    t_eps = 0.05
    denoiser_noise_scale = 1.0
    self_cond_prob = 0.0
    num_self_cond_cfg_tokens = 0

def generate_and_track(model, n_samples=256, n_steps=32, use_sde=False):
    """Generate samples and return texts, trajectories, and scores."""
    batch_size = 64
    all_texts = []
    all_traj = []
    
    detokenizer_map = load_detokenizer()
    
    config = DummyConfig()
    
    t_steps = get_sampling_steps(n_steps, time_schedule="uniform", device=DEVICE).flip(0)
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            bsz = min(batch_size, n_samples - start_idx)
            z = torch.randn(bsz, 16, 128, device=DEVICE)
            
            cond_seq = torch.zeros_like(z)
            cond_seq_mask = torch.zeros(bsz, 16, device=DEVICE)
            
            traj = [z.detach().cpu().numpy()]
            
            for i in range(len(t_steps) - 1):
                t_curr = t_steps[i]
                t_next = t_steps[i+1]
                
                if use_sde and i < len(t_steps) - 2:
                    # SDE step
                    z, _ = _sde_step(
                        model, z, t_curr, t_next, None,
                        config, 1.0, 1.0, cond_seq, cond_seq_mask,
                        gamma=0.5, generator=None
                    )
                else:
                    z, _ = _ode_step(
                        model, z, t_curr, t_next, None,
                        config, 1.0, 1.0, cond_seq, cond_seq_mask
                    )
                traj.append(z.detach().cpu().numpy())
                
            all_traj.append(np.stack(traj, axis=0)) # (steps+1, bsz, seq_len, dim)
            
            # Decode
            t_final = torch.ones(z.shape[0], dtype=z.dtype, device=DEVICE)
            _, logits = model(z, t_final, deterministic=True,
                              decoder_step_active=torch.ones(z.shape[0], device=DEVICE))
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            for b in range(bsz):
                words = [detokenizer_map.get(str(p), detokenizer_map.get(int(p), "")) for p in preds[b]]
                all_texts.append(" ".join(words))
            
    final_traj = np.concatenate(all_traj, axis=1) # (steps+1, n_samples, seq_len, dim)
    return all_texts, final_traj

def detect_degeneracy(texts):
    """Simple exact repetition detector."""
    is_deg = []
    for text in texts:
        words = text.split()
        if len(words) < 4:
            is_deg.append(True)
            continue
        
        # Check for 4-gram repetitions
        n = 4
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        if len(ngrams) == 0:
            is_deg.append(False)
            continue
            
        unique_ngrams = set(ngrams)
        if len(unique_ngrams) < len(ngrams) * 0.5:
            is_deg.append(True)
        else:
            is_deg.append(False)
    return np.array(is_deg)

def load_toy_model(ckpt_name):
    ckpt = torch.load(RESULTS_DIR / ckpt_name, map_location=DEVICE, weights_only=False)
    c = ckpt["config"]
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

def evaluate_model(ckpt_name, label, n_samples=256):
    print(f"\n{'='*40}")
    print(f"Evaluating {label} ({ckpt_name})")
    print(f"{'='*40}")
    
    model = load_toy_model(ckpt_name)
    
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
    
    # Non-degenerate quality comparison (log-PPL paired bootstrap)
    # Filter to samples where neither is degenerate
    valid = ~(ode_deg | sde_deg)
    
    if valid.sum() > 10:
        valid_ode_ppls = ode_ppls[valid]
        valid_sde_ppls = sde_ppls[valid]
        
        # log-PPL
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
        
    # Curvature-overconfidence correlation
    ode_curv = compute_turning_angles(ode_traj)
    
    valid_curv = ode_curv[valid]
    
    if valid.sum() > 10:
        r, p = scipy.stats.pearsonr(valid_curv, diff)
        print(f"Curvature vs SDE-ODE log-PPL penalty correlation: r = {r:.3f} (p={p:.4f})")
    else:
        r, p = 0, 1.0
        
    return {
        "ode_deg_rate": float(ode_deg_rate),
        "sde_deg_rate": float(sde_deg_rate),
        "ode_log_ppl": float(log_ode.mean()) if valid.sum() > 10 else None,
        "sde_log_ppl": float(log_sde.mean()) if valid.sum() > 10 else None,
        "sde_penalty": float(diff.mean()) if valid.sum() > 10 else None,
        "sde_penalty_ci": [float(x) for x in diff_ci],
        "curvature_correlation": float(r),
        "avg_ode_curvature": float(ode_curv.mean()),
        "ode_log_ppls": log_ode if valid.sum() > 10 else None,
        "sde_log_ppls": log_sde if valid.sum() > 10 else None
    }


def main():
    res_std = evaluate_model("testR1_model_std.pt", "Standard Schedule", n_samples=768)
    res_over = evaluate_model("testR1_model_over.pt", "Oversampled Schedule", n_samples=768)
    
    # Save results
    out_path = RESULTS_DIR / "testR1_power_check.json"
    save_dict = {
        "standard": {k: v for k, v in res_std.items() if k not in ["ode_log_ppls", "sde_log_ppls"]},
        "oversampled": {k: v for k, v in res_over.items() if k not in ["ode_log_ppls", "sde_log_ppls"]}
    }
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
        
    print(f"\nSaved R1 evaluation to {out_path}")
    
    print("\n--- Go/No-Go Criterion Check ---")
    deg_reduced = res_over["ode_deg_rate"] < res_std["ode_deg_rate"]
    print("\n--- Go/No-Go Criterion Check ---")
    diffs_std = res_std["sde_log_ppls"] - res_std["ode_log_ppls"]
    diffs_over = res_over["sde_log_ppls"] - res_over["ode_log_ppls"]
    
    mean_diff_diff, lo, hi = bootstrap_ci_paired_diff(diffs_over, diffs_std)
    
    print(f"Control SDE-ODE Gap: {res_std['sde_penalty']:.4f}")
    print(f"Oversampled SDE-ODE Gap: {res_over['sde_penalty']:.4f}")
    print(f"Difference in Gaps (Oversampled - Control): {mean_diff_diff:.4f} CI: ({lo:.4f}, {hi:.4f})")
    
    excludes_zero = (lo > 0) or (hi < 0)
    if mean_diff_diff < 0 and excludes_zero:
        print("GO: Oversampled training improved the SDE vs ODE gap significantly!")
    elif mean_diff_diff < 0:
        print("PARTIAL: Oversampled training improved the SDE vs ODE gap, but CI includes zero.")
    else:
        print("NO-GO: Oversampled training did not improve the gap.")

if __name__ == "__main__":
    main()
