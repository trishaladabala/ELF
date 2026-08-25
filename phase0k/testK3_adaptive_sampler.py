#!/usr/bin/env python3
"""K3 — Curvature-Driven Adaptive Sampler Prototype.

Implements an adaptive SDE sampler that modulates noise injection g
based on real-time trajectory curvature. Compares against fixed ODE
and fixed corrected-SDE on the same toy checkpoint.
"""

import json
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF
from utils.sampling_utils import net_out_to_v_x, get_sampling_steps

RESULTS_DIR = Path(__file__).resolve().parent / "results"
E_RESULTS = Path(__file__).resolve().parent.parent / "phase0e" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

T_EPS = 0.05
SIGMA = 1.0
N_SAMPLES = 512
BATCH_SIZE = 64
N_BOOTSTRAP = 1000
N_STEPS = 8
G_FIXED = 0.5
G_MAX = 0.5  # adaptive max
CURVATURE_THRESHOLD = 0.1  # radians; above this we start reducing g


def load_detokenizer():
    with open(E_RESULTS / "detokenizer_map.json", "r") as f:
        d_map = json.load(f)
    return {int(k): v for k, v in d_map.items()}


def decode_tokens_to_text(token_ids, detok_map):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid in detok_map:
            word = detok_map[tid]
            if word.startswith(' '): words.append(' ' + word[1:])
            else: words.append(word)
    return "".join(words).replace("  ", " ").strip()


def setup_external_lm():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE).eval()
    return model, tokenizer


def score_text(text, model, tokenizer, device=DEVICE):
    if not text.strip(): return float('inf')
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2: return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def load_real_checkpoint():
    ckpt_path = Path(__file__).resolve().parent.parent / "phase0c" / "results" / "mini_elf_real_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ELF(
        text_encoder_dim=cfg["encoder_dim"], max_length=cfg["max_length"],
        hidden_size=cfg["hidden_size"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        bottleneck_dim=cfg["bottleneck_dim"], num_time_tokens=2,
        num_self_cond_cfg_tokens=0, num_model_mode_tokens=2,
        vocab_size=cfg["vocab_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, cfg


def get_tokens(model, z, device):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_f = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_f, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        return logits.argmax(dim=-1).cpu().numpy()


def is_degenerate(token_ids):
    return len(set(int(t) for t in token_ids)) <= 2


def ode_sample(model, n, noise):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = noise.clone().to(DEVICE)
        steps = get_sampling_steps(N_STEPS, time_schedule="logit_normal", device=DEVICE, dtype=dtype)
        for i in range(N_STEPS):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            t_b = torch.full((n,), tc, dtype=dtype, device=DEVICE)
            net_out, _ = model(z, t_b, deterministic=True, decoder_step_active=None)
            v, _ = net_out_to_v_x(net_out, z, t_b, T_EPS)
            z = z + h * v
    return z


def fixed_sde_sample(model, n, noise, g=G_FIXED):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = noise.clone().to(DEVICE)
        steps = get_sampling_steps(N_STEPS, time_schedule="logit_normal", device=DEVICE, dtype=dtype)
        for i in range(N_STEPS):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            t_b = torch.full((n,), tc, dtype=dtype, device=DEVICE)
            net_out, _ = model(z, t_b, deterministic=True, decoder_step_active=None)
            _, x_pred = net_out_to_v_x(net_out, z, t_b, T_EPS)
            denom_v = max(1 - tc, T_EPS)
            v = (x_pred - z) / denom_v
            denom_s = max(1 - tc, T_EPS)**2 * SIGMA**2
            score = (tc * x_pred - z) / denom_s
            drift = v + (g**2 / 2) * score
            w = torch.randn_like(z)
            diffusion = g * math.sqrt(abs(h)) * w
            z = z + h * drift + diffusion
    return z


def adaptive_sde_sample(model, n, noise, g_max=G_MAX, curvature_threshold=CURVATURE_THRESHOLD):
    """Adaptive sampler: modulates g based on local curvature.
    
    At each step, we measure the turning angle between current and previous
    velocity. High curvature -> reduce g (go ODE-like). Low curvature -> keep
    full g (SDE, anti-degeneracy).
    """
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = noise.clone().to(DEVICE)
        steps = get_sampling_steps(N_STEPS, time_schedule="logit_normal", device=DEVICE, dtype=dtype)
        prev_v = None
        
        for i in range(N_STEPS):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            t_b = torch.full((n,), tc, dtype=dtype, device=DEVICE)
            net_out, _ = model(z, t_b, deterministic=True, decoder_step_active=None)
            _, x_pred = net_out_to_v_x(net_out, z, t_b, T_EPS)
            denom_v = max(1 - tc, T_EPS)
            v = (x_pred - z) / denom_v
            
            # Compute per-sample curvature from turning angle
            if prev_v is not None:
                # Flatten spatial dims for dot product: (n, seq_len*dim)
                v_flat = v.reshape(n, -1)
                pv_flat = prev_v.reshape(n, -1)
                dot = (v_flat * pv_flat).sum(dim=-1)
                nv = v_flat.norm(dim=-1)
                npv = pv_flat.norm(dim=-1)
                cos_sim = dot / (nv * npv + 1e-12)
                cos_sim = cos_sim.clamp(-1.0, 1.0)
                curvature = torch.acos(cos_sim)  # (n,) in radians
                
                # Adaptive g: linear interpolation
                # curvature < threshold -> g = g_max (full SDE)
                # curvature > 2*threshold -> g = 0 (pure ODE)
                alpha = ((curvature - curvature_threshold) / curvature_threshold).clamp(0.0, 1.0)
                g_adaptive = g_max * (1.0 - alpha)  # (n,)
                g_adaptive = g_adaptive.unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)
            else:
                # First step: use full g (no curvature info yet)
                g_adaptive = torch.full((n, 1, 1), g_max, dtype=dtype, device=DEVICE)
                curvature = torch.zeros(n, device=DEVICE)
            
            prev_v = v.clone()
            
            # SDE step with per-sample adaptive g
            denom_s = max(1 - tc, T_EPS)**2 * SIGMA**2
            score = (tc * x_pred - z) / denom_s
            drift = v + (g_adaptive**2 / 2) * score
            w = torch.randn_like(z)
            diffusion = g_adaptive * math.sqrt(abs(h)) * w
            z = z + h * drift + diffusion
    return z


def bootstrap_paired_ci(log_a, log_b, n_boot=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(log_a)
    diffs = np.array(log_b) - np.array(log_a)
    boot = np.array([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)


def evaluate_sampler(model, name, sample_fn, noise, detok_map, lm_model, lm_tokenizer):
    """Run a sampler and evaluate degeneracy + PPL."""
    print(f"  Evaluating {name}...")
    ppls = []
    degs = []
    
    for start in range(0, N_SAMPLES, BATCH_SIZE):
        bs = min(BATCH_SIZE, N_SAMPLES - start)
        n_batch = noise[start:start+bs]
        z_final = sample_fn(model, bs, n_batch)
        tokens = get_tokens(model, z_final, DEVICE)
        
        for i in range(bs):
            if is_degenerate(tokens[i]):
                ppls.append(50000.0)
                degs.append(True)
            else:
                text = decode_tokens_to_text(tokens[i], detok_map)
                ppl = score_text(text, lm_model, lm_tokenizer, DEVICE)
                if math.isnan(ppl) or math.isinf(ppl): ppl = 50000.0
                ppls.append(ppl)
                degs.append(False)
    
    ppls = np.array(ppls)
    degs = np.array(degs)
    deg_rate = float(degs.mean())
    
    # Non-degenerate subset
    valid = ~degs
    if valid.sum() > 0:
        log_ppls_valid = np.log(ppls[valid])
        mean_log_ppl = float(np.mean(log_ppls_valid))
    else:
        mean_log_ppl = float('nan')
    
    print(f"    Degeneracy: {100*deg_rate:.1f}%, Mean log-PPL (non-degen): {mean_log_ppl:.3f}")
    return ppls, degs, deg_rate, mean_log_ppl


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("K3 — Curvature-Driven Adaptive Sampler Prototype")
    print("=" * 60)
    
    model, cfg = load_real_checkpoint()
    detok_map = load_detokenizer()
    lm_model, lm_tokenizer = setup_external_lm()
    
    # Shared noise for all samplers
    torch.manual_seed(42)
    enc_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]
    noise = torch.randn(N_SAMPLES, seq_len, enc_dim, device=DEVICE)
    
    # Run all three
    ode_ppls, ode_degs, ode_deg_rate, ode_log_ppl = evaluate_sampler(
        model, "ODE", ode_sample, noise, detok_map, lm_model, lm_tokenizer)
    
    sde_ppls, sde_degs, sde_deg_rate, sde_log_ppl = evaluate_sampler(
        model, "Fixed SDE (g=0.5)", fixed_sde_sample, noise, detok_map, lm_model, lm_tokenizer)
    
    ada_ppls, ada_degs, ada_deg_rate, ada_log_ppl = evaluate_sampler(
        model, "Adaptive SDE", adaptive_sde_sample, noise, detok_map, lm_model, lm_tokenizer)
    
    # Pairwise comparisons on non-degenerate subset (all three valid)
    all_valid = ~ode_degs & ~sde_degs & ~ada_degs
    n_valid = int(all_valid.sum())
    print(f"\n  Jointly non-degenerate samples: {n_valid}/{N_SAMPLES}")
    
    if n_valid > 10:
        log_ode = np.log(ode_ppls[all_valid])
        log_sde = np.log(sde_ppls[all_valid])
        log_ada = np.log(ada_ppls[all_valid])
        
        # SDE vs ODE
        d1_mean, d1_lo, d1_hi = bootstrap_paired_ci(log_ode, log_sde)
        # Adaptive vs ODE
        d2_mean, d2_lo, d2_hi = bootstrap_paired_ci(log_ode, log_ada)
        # Adaptive vs SDE
        d3_mean, d3_lo, d3_hi = bootstrap_paired_ci(log_sde, log_ada)
    else:
        d1_mean = d1_lo = d1_hi = float('nan')
        d2_mean = d2_lo = d2_hi = float('nan')
        d3_mean = d3_lo = d3_hi = float('nan')
    
    print(f"\n  --- Non-Degenerate Quality (log-PPL, lower=better) ---")
    print(f"  SDE vs ODE:      Δ={d1_mean:+.3f} [{d1_lo:+.3f}, {d1_hi:+.3f}]")
    print(f"  Adaptive vs ODE: Δ={d2_mean:+.3f} [{d2_lo:+.3f}, {d2_hi:+.3f}]")
    print(f"  Adaptive vs SDE: Δ={d3_mean:+.3f} [{d3_lo:+.3f}, {d3_hi:+.3f}]")
    
    # The specific claim: does adaptive match SDE's degeneracy elimination
    # while reducing the non-degenerate quality penalty?
    ada_matches_deg = ada_deg_rate <= sde_deg_rate * 1.1  # within 10%
    ada_reduces_penalty = d3_mean < 0 and d3_hi < 0  # adaptive significantly better than SDE on non-degen
    
    if ada_matches_deg and ada_reduces_penalty:
        verdict = "GO: Adaptive sampler matches degeneracy elimination and reduces quality penalty."
    elif ada_matches_deg and d3_mean < 0:
        verdict = "PARTIAL: Adaptive matches degeneracy but quality improvement is not significant."
    elif not ada_matches_deg:
        verdict = "NO-GO: Adaptive sampler does not match SDE's degeneracy elimination."
    else:
        verdict = "NO-GO: Adaptive sampler does not clearly outperform fixed SDE."
    
    print(f"\n  Verdict: {verdict}")
    
    results = {
        "ode": {"deg_rate": ode_deg_rate, "mean_log_ppl_nondegen": ode_log_ppl},
        "fixed_sde": {"deg_rate": sde_deg_rate, "mean_log_ppl_nondegen": sde_log_ppl},
        "adaptive_sde": {"deg_rate": ada_deg_rate, "mean_log_ppl_nondegen": ada_log_ppl},
        "comparisons": {
            "sde_vs_ode": {"mean": d1_mean, "ci_lo": d1_lo, "ci_hi": d1_hi},
            "adaptive_vs_ode": {"mean": d2_mean, "ci_lo": d2_lo, "ci_hi": d2_hi},
            "adaptive_vs_sde": {"mean": d3_mean, "ci_lo": d3_lo, "ci_hi": d3_hi},
        },
        "verdict": verdict,
    }
    
    with open(RESULTS_DIR / "testK3_adaptive_sampler.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Degeneracy rates
    ax = axes[0]
    names = ["ODE", "Fixed SDE\n(g=0.5)", "Adaptive\nSDE"]
    rates = [100*ode_deg_rate, 100*sde_deg_rate, 100*ada_deg_rate]
    colors = ["#DD8452", "#4C72B0", "#55A868"]
    bars = ax.bar(names, rates, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Degeneracy Rate (%)")
    ax.set_title("Degeneracy Elimination")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{rate:.1f}%", ha='center', fontsize=10)
    
    # Non-degenerate quality
    ax = axes[1]
    log_ppls = [ode_log_ppl, sde_log_ppl, ada_log_ppl]
    bars = ax.bar(names, log_ppls, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Mean Log-PPL (non-degenerate, lower=better)")
    ax.set_title("Quality on Non-Degenerate Samples")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, log_ppls):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f"{val:.3f}", ha='center', fontsize=10)
    
    plt.suptitle("K3: Adaptive Sampler Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testK3_adaptive_sampler.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testK3_adaptive_sampler.json'}")
    print(f"  Saved → {RESULTS_DIR / 'testK3_adaptive_sampler.png'}")


if __name__ == "__main__":
    main()
