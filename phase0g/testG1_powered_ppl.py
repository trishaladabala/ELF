#!/usr/bin/env python3
"""G1 — Properly Powered External-PPL Comparison.

Fixes E1's statistical weakness: uses 256 sequences per cell (up from 32),
paired bootstrap CIs, and degenerate-sequence filtering.
"""

import json
import sys
import time
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
N_SAMPLES = 256          # Up from 32
BATCH_SIZE = 64          # Generate in batches to avoid OOM
STEP_BUDGETS = [4, 8, 16, 32, 64]
TIME_SCHEDULES = ["uniform", "logit_normal"]
N_BOOTSTRAP = 1000


def load_detokenizer():
    map_path = E_RESULTS / "detokenizer_map.json"
    with open(map_path, "r") as f:
        d_map = json.load(f)
    return {int(k): v for k, v in d_map.items()}


def decode_tokens_to_text(token_ids, detok_map):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid in detok_map:
            word = detok_map[tid]
            if word.startswith(' '):
                words.append(' ' + word[1:])
            else:
                words.append(word)
    text = "".join(words).replace("  ", " ").strip()
    return text


def setup_external_lm():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"Loading distilgpt2 on {DEVICE}...")
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE).eval()
    return model, tokenizer


def score_text(text, model, tokenizer, device=DEVICE):
    if not text.strip():
        return float('inf')
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


def is_degenerate(token_ids):
    """Flag sequences that are all-repeated or near-empty."""
    unique = len(set(int(t) for t in token_ids))
    return unique <= 2  # 2 or fewer unique tokens = degenerate


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


# ── Samplers (reused from E1) ──

def ode_sample(model, n_samples, n_steps, encoder_dim, seq_len,
               time_schedule="uniform", device="cpu", initial_noise=None):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = initial_noise.clone().to(device)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            z = z + h * v_pred
    return z


def elf_sde_sample(model, n_samples, n_steps, encoder_dim, seq_len,
                   gamma=0.5, time_schedule="uniform", device="cpu",
                   initial_noise=None):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = initial_noise.clone().to(device)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            alpha = max(0.0, min(1.0, 1.0 - gamma * h))
            t_back = alpha * t_curr
            if gamma > 0:
                eps = torch.randn_like(z)
                z_back = alpha * z + (1.0 - alpha) * eps
            else:
                z_back = z
            t_batch = torch.full((n_samples,), t_back, dtype=dtype, device=device)
            net_out, _ = model(z_back, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z_back, t_batch, T_EPS)
            z = z_back + (t_next - t_back) * v_pred
    return z


def corrected_sde_sample(model, n_samples, n_steps, encoder_dim, seq_len,
                         g=0.5, time_schedule="uniform", device="cpu",
                         initial_noise=None):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = initial_noise.clone().to(device)
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            _, x_pred = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            denom_v = max(1 - t_curr, T_EPS)
            v = (x_pred - z) / denom_v
            denom_s = max(1 - t_curr, T_EPS)**2 * SIGMA**2
            score = (t_curr * x_pred - z) / denom_s
            drift = v + (g**2 / 2) * score
            noise = torch.randn_like(z) if g > 0 else 0.0
            diffusion = g * math.sqrt(abs(h)) * noise if g > 0 else 0.0
            z = z + h * drift + diffusion
    return z


def get_tokens(model, z, device):
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        tokens = logits.argmax(dim=-1).cpu().numpy()
    return tokens


def generate_and_score(sampler_fn, n_steps, sched, elf_model, lm_model,
                       lm_tokenizer, detok_map, encoder_dim, seq_len):
    """Generate N_SAMPLES sequences in batches, decode, filter degenerates, score."""
    all_ppls = []
    all_divs = []
    n_degenerate = 0

    for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
        bs = min(BATCH_SIZE, N_SAMPLES - batch_start)
        torch.manual_seed(42 + batch_start)
        noise = torch.randn(bs, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)
        z_final = sampler_fn(elf_model, bs, n_steps, encoder_dim, seq_len,
                             time_schedule=sched, device=DEVICE, initial_noise=noise)
        tokens = get_tokens(elf_model, z_final, DEVICE)

        for i in range(bs):
            tids = tokens[i]
            if is_degenerate(tids):
                n_degenerate += 1
                continue
            unique = len(np.unique(tids))
            all_divs.append(unique)
            text = decode_tokens_to_text(tids, detok_map)
            ppl = score_text(text, lm_model, lm_tokenizer, DEVICE)
            if not math.isinf(ppl) and not math.isnan(ppl):
                all_ppls.append(ppl)

    return np.array(all_ppls), np.array(all_divs), n_degenerate


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI for the mean."""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(data)), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI for the mean difference (a - b). Paired if same length."""
    rng = np.random.default_rng(42)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    diffs = a - b
    boot_diffs = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("G1 — Properly Powered External-PPL Comparison")
    print(f"     N={N_SAMPLES} sequences/cell, {N_BOOTSTRAP} bootstrap resamples")
    print("=" * 60)

    detok_map = load_detokenizer()
    lm_model, lm_tokenizer = setup_external_lm()
    elf_model, cfg = load_real_checkpoint()
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]

    # Only the 3 most important samplers for the comparison
    sampler_configs = {
        "ODE": lambda model, bs, ns, ed, sl, **kw: ode_sample(model, bs, ns, ed, sl, **kw),
        "ELF-SDE γ=0.5": lambda model, bs, ns, ed, sl, **kw: elf_sde_sample(model, bs, ns, ed, sl, gamma=0.5, **kw),
        "Corrected-SDE g=0.5": lambda model, bs, ns, ed, sl, **kw: corrected_sde_sample(model, bs, ns, ed, sl, g=0.5, **kw),
    }

    all_results = {}
    all_raw_ppls = {}  # For pairwise bootstrap

    for sched in TIME_SCHEDULES:
        print(f"\n  ═══ {sched} ═══")
        sched_results = {}
        sched_ppls = {}
        for name, fn in sampler_configs.items():
            print(f"    {name}:")
            step_results = {}
            step_ppls = {}
            for n_steps in STEP_BUDGETS:
                t0 = time.time()
                ppls, divs, n_degen = generate_and_score(
                    fn, n_steps, sched, elf_model, lm_model, lm_tokenizer,
                    detok_map, encoder_dim, seq_len
                )
                elapsed = time.time() - t0

                if len(ppls) > 0:
                    mean, lo, hi = bootstrap_ci(ppls)
                else:
                    mean, lo, hi = float('inf'), float('inf'), float('inf')

                step_results[str(n_steps)] = {
                    "ppl_mean": mean, "ppl_ci_lo": lo, "ppl_ci_hi": hi,
                    "diversity_mean": float(np.mean(divs)) if len(divs) > 0 else 0.0,
                    "n_valid": len(ppls),
                    "n_degenerate": n_degen,
                    "time_s": elapsed,
                }
                step_ppls[str(n_steps)] = ppls

                print(f"      steps={n_steps:3d}: "
                      f"PPL={mean:.1f} [{lo:.1f}, {hi:.1f}], "
                      f"div={np.mean(divs):.1f}, "
                      f"valid={len(ppls)}/{N_SAMPLES}, degen={n_degen}, "
                      f"time={elapsed:.0f}s")

            sched_results[name] = step_results
            sched_ppls[name] = step_ppls
        all_results[sched] = sched_results
        all_raw_ppls[sched] = sched_ppls

    # Pairwise bootstrap CIs for key comparisons
    print(f"\n{'=' * 60}")
    print("PAIRWISE BOOTSTRAP COMPARISONS (Δ = sampler - ODE)")
    print(f"{'=' * 60}")
    comparisons = {}
    for sched in TIME_SCHEDULES:
        print(f"\n  ── {sched} ──")
        comp_sched = {}
        for other in ["ELF-SDE γ=0.5", "Corrected-SDE g=0.5"]:
            comp_steps = {}
            for ns in STEP_BUDGETS:
                ode_ppls = all_raw_ppls[sched]["ODE"][str(ns)]
                other_ppls = all_raw_ppls[sched][other][str(ns)]
                if len(ode_ppls) > 10 and len(other_ppls) > 10:
                    diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(other_ppls, ode_ppls)
                    excludes_zero = (diff_lo > 0) or (diff_hi < 0)
                    comp_steps[str(ns)] = {
                        "diff_mean": diff_mean, "diff_ci_lo": diff_lo, "diff_ci_hi": diff_hi,
                        "excludes_zero": excludes_zero,
                    }
                    sig = "✅ SIG" if excludes_zero else "NS"
                    direction = "↑ worse" if diff_mean > 0 else "↓ better"
                    print(f"    {other} vs ODE at {ns} steps: "
                          f"Δ={diff_mean:+.1f} [{diff_lo:+.1f}, {diff_hi:+.1f}] {sig} ({direction})")
                else:
                    comp_steps[str(ns)] = {"diff_mean": None, "note": "insufficient valid samples"}
            comp_sched[other] = comp_steps
        comparisons[sched] = comp_sched

    # Verdict
    any_sig_advantage = False
    for sched in comparisons:
        for sampler in comparisons[sched]:
            for ns in comparisons[sched][sampler]:
                c = comparisons[sched][sampler][ns]
                if c.get("excludes_zero") and c.get("diff_mean", 0) < 0:
                    any_sig_advantage = True

    print(f"\n{'=' * 60}")
    if any_sig_advantage:
        print("🟢 G1: At least one sampler shows a statistically significant PPL advantage over ODE.")
    else:
        print("🔴 G1: No sampler shows a statistically significant PPL advantage over ODE.")
        print("   The SDE quality claim is not supported at toy scale with proper statistics.")
    print(f"{'=' * 60}")

    # Save
    with open(RESULTS_DIR / "testG1_powered_ppl_comparison.json", "w") as f:
        json.dump({
            "per_cell": all_results,
            "pairwise_comparisons": comparisons,
            "any_significant_advantage": any_sig_advantage,
            "n_samples_per_cell": N_SAMPLES,
            "n_bootstrap": N_BOOTSTRAP,
        }, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {"ODE": "#4C72B0", "ELF-SDE γ=0.5": "#55A868", "Corrected-SDE g=0.5": "#C44E52"}
    markers = {"ODE": "o", "ELF-SDE γ=0.5": "s", "Corrected-SDE g=0.5": "^"}

    for col, sched in enumerate(TIME_SCHEDULES):
        sd = all_results[sched]
        # PPL with CI
        ax = axes[0][col]
        for name, data in sd.items():
            steps = sorted([int(s) for s in data.keys()])
            means = [data[str(s)]["ppl_mean"] for s in steps]
            los = [data[str(s)]["ppl_ci_lo"] for s in steps]
            his = [data[str(s)]["ppl_ci_hi"] for s in steps]
            yerr_lo = [m - lo for m, lo in zip(means, los)]
            yerr_hi = [hi - m for m, hi in zip(means, his)]
            ax.errorbar(steps, means, yerr=[yerr_lo, yerr_hi],
                        fmt=f"{markers[name]}-", color=colors[name], lw=2,
                        markersize=7, label=name, capsize=4, alpha=0.85)
        ax.set_xlabel("Steps"); ax.set_ylabel("External PPL (↓ better)")
        ax.set_title(f"Gen PPL (distilgpt2, N={N_SAMPLES}) — {sched}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

        # Diversity
        ax = axes[1][col]
        for name, data in sd.items():
            steps = sorted([int(s) for s in data.keys()])
            divs = [data[str(s)]["diversity_mean"] for s in steps]
            ax.plot(steps, divs, f"{markers[name]}-", color=colors[name],
                    lw=2, markersize=7, label=name, alpha=0.85)
        ax.set_xlabel("Steps"); ax.set_ylabel("Unique Tokens (↑ better)")
        ax.set_title(f"Diversity — {sched}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.suptitle("G1: Properly Powered PPL Comparison (Bootstrap 95% CI)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testG1_powered_ppl_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testG1_powered_ppl_comparison.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
