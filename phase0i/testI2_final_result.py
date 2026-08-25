#!/usr/bin/env python3
"""I2 — Final Confirmatory Test for Idea 6.

Executes the pre-registered comparison (Corrected-SDE g=0.5 vs ODE under
logit-normal at 8 steps) with N=512 for higher power.

Also saves the full generation trajectories (z_t over time) for these
confirmatory samples to be used in Part J (Geometry Diagnostics).
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
N_SAMPLES = 512
BATCH_SIZE = 64
N_BOOTSTRAP = 2000

# We'll run a few descriptive contexts, but 8 steps is the confirmatory one.
DESCRIPTIVE_STEPS = [4, 16, 32]
CONFIRMATORY_STEP = 8


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
    return "".join(words).replace("  ", " ").strip()


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
    unique = len(set(int(t) for t in token_ids))
    return unique <= 2


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


def ode_sample(model, n_samples, n_steps, encoder_dim, seq_len,
               time_schedule="logit_normal", device="cpu", initial_noise=None,
               return_trajectory=False):
    dtype = next(model.parameters()).dtype
    trajectory = []
    with torch.no_grad():
        z = initial_noise.clone().to(device)
        if return_trajectory:
            trajectory.append(z.cpu().numpy())
        steps = get_sampling_steps(n_steps, time_schedule=time_schedule, device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            z = z + h * v_pred
            if return_trajectory:
                trajectory.append(z.cpu().numpy())
    if return_trajectory:
        return z, np.array(trajectory)
    return z


def corrected_sde_sample(model, n_samples, n_steps, encoder_dim, seq_len,
                         g=0.5, time_schedule="logit_normal", device="cpu",
                         initial_noise=None, return_trajectory=False):
    dtype = next(model.parameters()).dtype
    trajectory = []
    with torch.no_grad():
        z = initial_noise.clone().to(device)
        if return_trajectory:
            trajectory.append(z.cpu().numpy())
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
            if return_trajectory:
                trajectory.append(z.cpu().numpy())
    if return_trajectory:
        return z, np.array(trajectory)
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


def bootstrap_paired_ci(ode_scores, sde_scores, n_boot=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(ode_scores)
    diffs = np.array(ode_scores) - np.array(sde_scores)  # > 0 means SDE is better
    boot_diffs = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("I2 — Final Confirmatory Test for Idea 6")
    print(f"     N={N_SAMPLES}, schedule=logit_normal")
    print("=" * 60)

    detok_map = load_detokenizer()
    lm_model, lm_tokenizer = setup_external_lm()
    elf_model, cfg = load_real_checkpoint()
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]

    results = {"descriptive": {}, "confirmatory": {}}

    # 1. Run the Confirmatory Test (8 steps, logit_normal, returning trajectories)
    print(f"\n── Running CONFIRMATORY test (8 steps) ──")
    ode_ppls = []
    sde_ppls = []
    ode_traj_all = []
    sde_traj_all = []

    for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
        bs = min(BATCH_SIZE, N_SAMPLES - batch_start)
        torch.manual_seed(999 + batch_start)
        noise = torch.randn(bs, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)

        z_ode, t_ode = ode_sample(elf_model, bs, CONFIRMATORY_STEP, encoder_dim, seq_len,
                                  time_schedule="logit_normal", device=DEVICE, initial_noise=noise,
                                  return_trajectory=True)
        z_sde, t_sde = corrected_sde_sample(elf_model, bs, CONFIRMATORY_STEP, encoder_dim, seq_len,
                                            g=0.5, time_schedule="logit_normal", device=DEVICE, initial_noise=noise,
                                            return_trajectory=True)

        ode_traj_all.append(t_ode)
        sde_traj_all.append(t_sde)

        tokens_ode = get_tokens(elf_model, z_ode, DEVICE)
        tokens_sde = get_tokens(elf_model, z_sde, DEVICE)

        for i in range(bs):
            # We don't drop degenerates entirely, we cap them to a high penalty to maintain paired structure
            # (Though Corrected-SDE and ODE shouldn't produce many anyway).
            to = tokens_ode[i]
            ts = tokens_sde[i]

            if is_degenerate(to):
                po = 50000.0
            else:
                text_o = decode_tokens_to_text(to, detok_map)
                po = score_text(text_o, lm_model, lm_tokenizer, DEVICE)
                if math.isnan(po) or math.isinf(po): po = 50000.0

            if is_degenerate(ts):
                ps = 50000.0
            else:
                text_s = decode_tokens_to_text(ts, detok_map)
                ps = score_text(text_s, lm_model, lm_tokenizer, DEVICE)
                if math.isnan(ps) or math.isinf(ps): ps = 50000.0

            ode_ppls.append(po)
            sde_ppls.append(ps)

    # Save trajectories for Part J
    # Concatenate along batch dimension (axis 1 because shape is [steps, batch, seq, dim])
    ode_traj_full = np.concatenate(ode_traj_all, axis=1)
    sde_traj_full = np.concatenate(sde_traj_all, axis=1)
    np.save(RESULTS_DIR / "testI2_ode_trajectories.npy", ode_traj_full)
    np.save(RESULTS_DIR / "testI2_sde_trajectories.npy", sde_traj_full)
    print(f"  Saved trajectories to {RESULTS_DIR}")

    diff_mean, diff_lo, diff_hi = bootstrap_paired_ci(ode_ppls, sde_ppls)
    excludes_zero = (diff_lo > 0) or (diff_hi < 0)
    
    results["confirmatory"]["8_steps"] = {
        "ode_ppl_mean": float(np.mean(ode_ppls)),
        "sde_ppl_mean": float(np.mean(sde_ppls)),
        "diff_mean": diff_mean,
        "diff_ci_lo": diff_lo,
        "diff_ci_hi": diff_hi,
        "excludes_zero": excludes_zero
    }

    print(f"  ODE PPL: {np.mean(ode_ppls):.1f}")
    print(f"  SDE PPL: {np.mean(sde_ppls):.1f}")
    print(f"  Paired Δ: {diff_mean:+.1f} [{diff_lo:+.1f}, {diff_hi:+.1f}]")
    print(f"  Excludes 0: {excludes_zero}")

    # 2. Run Descriptive Contexts (4, 16, 32 steps)
    print(f"\n── Running DESCRIPTIVE contexts ──")
    for steps in DESCRIPTIVE_STEPS:
        print(f"  {steps} steps...")
        o_p = []
        s_p = []
        for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
            bs = min(BATCH_SIZE, N_SAMPLES - batch_start)
            torch.manual_seed(999 + batch_start)
            noise = torch.randn(bs, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)

            z_ode = ode_sample(elf_model, bs, steps, encoder_dim, seq_len,
                               time_schedule="logit_normal", device=DEVICE, initial_noise=noise)
            z_sde = corrected_sde_sample(elf_model, bs, steps, encoder_dim, seq_len,
                                         g=0.5, time_schedule="logit_normal", device=DEVICE, initial_noise=noise)

            tokens_ode = get_tokens(elf_model, z_ode, DEVICE)
            tokens_sde = get_tokens(elf_model, z_sde, DEVICE)

            for i in range(bs):
                to = tokens_ode[i]
                ts = tokens_sde[i]

                po = 50000.0 if is_degenerate(to) else score_text(decode_tokens_to_text(to, detok_map), lm_model, lm_tokenizer, DEVICE)
                ps = 50000.0 if is_degenerate(ts) else score_text(decode_tokens_to_text(ts, detok_map), lm_model, lm_tokenizer, DEVICE)

                if math.isnan(po) or math.isinf(po): po = 50000.0
                if math.isnan(ps) or math.isinf(ps): ps = 50000.0
                o_p.append(po)
                s_p.append(ps)

        d_mean, d_lo, d_hi = bootstrap_paired_ci(o_p, s_p)
        results["descriptive"][f"{steps}_steps"] = {
            "ode_ppl_mean": float(np.mean(o_p)),
            "sde_ppl_mean": float(np.mean(s_p)),
            "diff_mean": d_mean,
            "diff_ci_lo": d_lo,
            "diff_ci_hi": d_hi,
            "excludes_zero": (d_lo > 0) or (d_hi < 0)
        }
        print(f"    Δ: {d_mean:+.1f} [{d_lo:+.1f}, {d_hi:+.1f}]")

    print(f"\n{'=' * 60}")
    if results["confirmatory"]["8_steps"]["excludes_zero"] and results["confirmatory"]["8_steps"]["diff_mean"] > 0:
        print("✅ CONFIRMED: Corrected-SDE significantly outperforms ODE under realistic logit_normal settings.")
        verdict = "CONFIRMED_GO"
    else:
        print("❌ NULL/NEGATIVE: No significant advantage for Corrected-SDE under realistic logit_normal settings.")
        verdict = "NULL_FINDING"
    print(f"{'=' * 60}")

    results["verdict"] = verdict

    with open(RESULTS_DIR / "testI2_final_result.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_steps = sorted(DESCRIPTIVE_STEPS + [CONFIRMATORY_STEP])
    means = []
    err_los = []
    err_his = []
    colors = []
    
    for s in all_steps:
        if s == CONFIRMATORY_STEP:
            d = results["confirmatory"][f"{s}_steps"]
            colors.append('#C44E52')  # Highlight confirmatory
        else:
            d = results["descriptive"][f"{s}_steps"]
            colors.append('#4C72B0')
            
        means.append(d["diff_mean"])
        err_los.append(d["diff_mean"] - d["diff_ci_lo"])
        err_his.append(d["diff_ci_hi"] - d["diff_mean"])

    ax.errorbar(all_steps, means, yerr=[err_los, err_his], fmt='o', color='gray', ecolor='gray', capsize=5, zorder=1)
    ax.scatter(all_steps, means, color=colors, s=100, zorder=2)
    
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel("Sampling Steps")
    ax.set_ylabel("PPL Difference (ODE - SDE)\nPositive = SDE is better")
    ax.set_title("I2: Paired Bootstrap Mean Δ PPL (logit_normal)\nRed = Pre-registered Confirmatory Test", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_steps)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testI2_final_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot to {RESULTS_DIR / 'testI2_final_plot.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
