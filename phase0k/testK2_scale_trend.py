#!/usr/bin/env python3
"""K2 — Intermediate-Scale Curvature-Direction Check.

Trains two additional mini-ELF checkpoints at ~1M and ~5M params,
then runs the curvature-vs-entropy pipeline at all three scales to
check if the overconfidence correlation's direction is stable.
"""

import json
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF
from utils.sampling_utils import sample_timesteps, net_out_to_v_x, get_sampling_steps

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PHASE0_RESULTS = Path(__file__).resolve().parent.parent / "phase0" / "results"
C4_RESULTS = Path(__file__).resolve().parent.parent / "phase0c" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ENCODER_DIM = 128
TOY_MAX_LEN = 16
MAX_VOCAB_SIZE = 1024
T_EPS = 0.05
SIGMA = 1.0

SCALE_CONFIGS = [
    {"name": "275K", "hidden": 64,  "depth": 2, "heads": 4, "bottleneck": 32,  "mlp_ratio": 2.0, "steps": 15000},
    {"name": "1M",   "hidden": 128, "depth": 4, "heads": 8, "bottleneck": 64,  "mlp_ratio": 2.0, "steps": 20000},
    {"name": "5M",   "hidden": 256, "depth": 6, "heads": 8, "bottleneck": 128, "mlp_ratio": 2.0, "steps": 25000},
]

N_EVAL = 256
BATCH_SIZE = 32
N_STEPS_SAMPLE = 8


class ToyConfig:
    t_eps = T_EPS
    self_cond_prob = 0.0
    denoiser_p_mean = 0.8
    denoiser_p_std = 0.8
    denoiser_noise_scale = 1.0
    time_schedule = "logit_normal"
    decoder_prob = 0.5
    decoder_noise_scale = 1.0
    decoder_p_mean = 0.8
    decoder_p_std = 0.8
    num_self_cond_cfg_tokens = 0
    label_drop_prob = 0.0
    pad_token = "pad"
    latent_mean = 0.0
    latent_std = 1.0


def prepare_real_data():
    x_128 = np.load(PHASE0_RESULTS / "x_128.npy")
    seq_ids = np.load(PHASE0_RESULTS / "seq_ids.npy")
    pos_ids = np.load(PHASE0_RESULTS / "pos_ids.npy")

    unique_seqs = np.unique(seq_ids)
    sequences_emb = []

    for sid in unique_seqs:
        mask = seq_ids == sid
        embs = x_128[mask]
        positions = pos_ids[mask]
        order = np.argsort(positions)
        embs = embs[order]
        if len(embs) >= TOY_MAX_LEN:
            sequences_emb.append(embs[:TOY_MAX_LEN])
        elif len(embs) >= 4:
            padded = np.zeros((TOY_MAX_LEN, embs.shape[1]), dtype=embs.dtype)
            padded[:len(embs)] = embs
            sequences_emb.append(padded)

    sequences_emb = np.array(sequences_emb, dtype=np.float32)

    from sklearn.cluster import MiniBatchKMeans
    n_clusters = min(MAX_VOCAB_SIZE, 1024)
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(x_128.shape[0], min(50000, x_128.shape[0]), replace=False)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(x_128[sub_idx])

    all_token_ids = []
    for emb_seq in sequences_emb:
        cluster_ids = kmeans.predict(emb_seq)
        all_token_ids.append(cluster_ids)
    all_token_ids = np.array(all_token_ids, dtype=np.int64)

    return sequences_emb, all_token_ids, n_clusters


def create_model(cfg, vocab_size):
    return ELF(
        text_encoder_dim=ENCODER_DIM, max_length=TOY_MAX_LEN,
        hidden_size=cfg["hidden"], depth=cfg["depth"],
        num_heads=cfg["heads"], mlp_ratio=cfg["mlp_ratio"],
        bottleneck_dim=cfg["bottleneck"], num_time_tokens=2,
        num_self_cond_cfg_tokens=0, num_model_mode_tokens=2,
        vocab_size=vocab_size,
    )


def train_step(model, x0, input_ids, attention_mask, cond_seq_mask, config, device):
    dtype = next(model.parameters()).dtype
    batch_size, seq_length, encoder_dim = x0.shape
    t = sample_timesteps(batch_size, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                         time_schedule=config.time_schedule, device=device, dtype=dtype)
    noise = torch.randn_like(x0)
    loss_mask = attention_mask * (1 - cond_seq_mask)
    cond_mask_3d = cond_seq_mask.unsqueeze(-1)
    t_exp = t.reshape(-1, 1, 1)
    denoiser_z = t_exp * x0 + (1 - t_exp) * noise * config.denoiser_noise_scale
    denoiser_z = cond_mask_3d * x0 + (1 - cond_mask_3d) * denoiser_z
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_exp, min=config.t_eps)

    decoder_active = torch.bernoulli(torch.full((batch_size,), config.decoder_prob)).to(device=device, dtype=dtype)
    decoder_B11 = decoder_active.view(-1, 1, 1)
    decoder_B1 = decoder_active.view(-1, 1)

    decoder_z_vals = (torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
                      * config.decoder_p_std + config.decoder_p_mean)
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn_like(x0) * config.decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_mixed = decoder_active * 1.0 + (1.0 - decoder_active) * t
    z_mixed = decoder_B11 * decoder_z + (1.0 - decoder_B11) * denoiser_z

    net_out, decoder_logits = model(z_mixed, t_mixed, deterministic=False, decoder_step_active=decoder_active)
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
    l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_B1
    l2_mask = loss_mask_f * (1.0 - decoder_B1)
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)
    ce_loss = ((ce_per_token * ce_mask).sum() / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    return loss, ce_loss


def train_checkpoint(scale_cfg, embeddings, token_ids, vocab_size):
    name = scale_cfg["name"]
    ckpt_path = RESULTS_DIR / f"ckpt_{name}.pt"
    if ckpt_path.exists():
        print(f"  Checkpoint {name} already exists, loading...")
        return ckpt_path

    print(f"\n  Training {name} model...")
    model = create_model(scale_cfg, vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    num_steps = scale_cfg["steps"]
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    config = ToyConfig()
    rng = np.random.default_rng(42)
    n_sequences = len(embeddings)
    random_baseline = np.log(vocab_size)

    model.train()
    t0 = time.time()
    ce_history = []
    for step in range(num_steps):
        idx = rng.choice(n_sequences, BATCH_SIZE, replace=False)
        x0 = torch.tensor(embeddings[idx], dtype=torch.float32, device=DEVICE)
        ids = torch.tensor(token_ids[idx], dtype=torch.long, device=DEVICE)
        attn = torch.ones(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)
        cond = torch.zeros(BATCH_SIZE, TOY_MAX_LEN, device=DEVICE)

        optimizer.zero_grad()
        loss, ce_loss = train_step(model, x0, ids, attn, cond, config, DEVICE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ce_history.append(float(ce_loss.item()))

        if (step + 1) % 2000 == 0:
            avg_ce = np.mean(ce_history[-500:])
            print(f"    Step {step+1}/{num_steps}: CE={avg_ce:.4f} (random={random_baseline:.2f}), "
                  f"time={time.time()-t0:.0f}s")

    final_ce = float(np.mean(ce_history[-500:]))
    print(f"  {name} done: final CE={final_ce:.4f}, reduction={100*(1-final_ce/random_baseline):.1f}%")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "encoder_dim": ENCODER_DIM, "vocab_size": vocab_size,
            "hidden_size": scale_cfg["hidden"], "depth": scale_cfg["depth"],
            "num_heads": scale_cfg["heads"], "mlp_ratio": scale_cfg["mlp_ratio"],
            "bottleneck_dim": scale_cfg["bottleneck"], "max_length": TOY_MAX_LEN,
        },
        "training": {"final_ce": final_ce, "params": n_params},
    }, ckpt_path)
    return ckpt_path


def load_ckpt(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
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


def ode_sample(model, n, n_steps, enc_dim, seq_len, device, noise, return_traj=False):
    dtype = next(model.parameters()).dtype
    traj = []
    with torch.no_grad():
        z = noise.clone().to(device)
        if return_traj: traj.append(z.cpu().numpy())
        steps = get_sampling_steps(n_steps, time_schedule="logit_normal", device=device, dtype=dtype)
        for i in range(n_steps):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            t_b = torch.full((n,), tc, dtype=dtype, device=device)
            net_out, _ = model(z, t_b, deterministic=True, decoder_step_active=None)
            v, _ = net_out_to_v_x(net_out, z, t_b, T_EPS)
            z = z + h * v
            if return_traj: traj.append(z.cpu().numpy())
    if return_traj: return z, np.array(traj)
    return z


def sde_sample(model, n, n_steps, enc_dim, seq_len, g, device, noise, return_traj=False):
    dtype = next(model.parameters()).dtype
    traj = []
    with torch.no_grad():
        z = noise.clone().to(device)
        if return_traj: traj.append(z.cpu().numpy())
        steps = get_sampling_steps(n_steps, time_schedule="logit_normal", device=device, dtype=dtype)
        for i in range(n_steps):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            t_b = torch.full((n,), tc, dtype=dtype, device=device)
            net_out, _ = model(z, t_b, deterministic=True, decoder_step_active=None)
            _, x_pred = net_out_to_v_x(net_out, z, t_b, T_EPS)
            denom_v = max(1 - tc, T_EPS)
            v = (x_pred - z) / denom_v
            denom_s = max(1 - tc, T_EPS)**2 * SIGMA**2
            score = (tc * x_pred - z) / denom_s
            drift = v + (g**2 / 2) * score
            w = torch.randn_like(z) if g > 0 else 0.0
            diffusion = g * math.sqrt(abs(h)) * w if g > 0 else 0.0
            z = z + h * drift + diffusion
            if return_traj: traj.append(z.cpu().numpy())
    if return_traj: return z, np.array(traj)
    return z


def get_tokens(model, z, device):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_f = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_f, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        return logits.argmax(dim=-1).cpu().numpy()


def decode_entropy(model, z, device):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_f = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_f, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return entropy.cpu().numpy()


def is_degenerate(token_ids):
    return len(set(int(t) for t in token_ids)) <= 2


def compute_turning_angles_per_token(trajectory):
    v = np.diff(trajectory, axis=0)
    steps, N, seq_len, dim = v.shape
    if steps < 2: return np.zeros((N, seq_len))
    v_prev, v_curr = v[:-1], v[1:]
    dot = np.sum(v_prev * v_curr, axis=-1)
    n_prev = np.linalg.norm(v_prev, axis=-1)
    n_curr = np.linalg.norm(v_curr, axis=-1)
    cos_sim = np.clip(dot / (n_prev * n_curr + 1e-12), -1.0, 1.0)
    return np.mean(np.arccos(cos_sim), axis=0)


def run_pipeline_for_scale(model, cfg, scale_name):
    """Run ODE+SDE generation, extract curvature, compute correlations."""
    enc_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]

    print(f"\n  Generating N={N_EVAL} samples for {scale_name}...")
    torch.manual_seed(42)
    noise = torch.randn(N_EVAL, seq_len, enc_dim, device=DEVICE)

    # Generate in batches
    ode_traj_all, sde_traj_all = [], []
    BS = 64
    for start in range(0, N_EVAL, BS):
        bs = min(BS, N_EVAL - start)
        n_batch = noise[start:start+bs]
        _, ode_t = ode_sample(model, bs, N_STEPS_SAMPLE, enc_dim, seq_len, DEVICE, n_batch, return_traj=True)
        _, sde_t = sde_sample(model, bs, N_STEPS_SAMPLE, enc_dim, seq_len, 0.5, DEVICE, n_batch, return_traj=True)
        ode_traj_all.append(ode_t)
        sde_traj_all.append(sde_t)

    ode_traj = np.concatenate(ode_traj_all, axis=1)
    sde_traj = np.concatenate(sde_traj_all, axis=1)

    z_ode = torch.tensor(ode_traj[-1], dtype=torch.float32, device=DEVICE)
    z_sde = torch.tensor(sde_traj[-1], dtype=torch.float32, device=DEVICE)

    # Curvature
    ode_curv = compute_turning_angles_per_token(ode_traj)

    # Entropy
    ent_ode_all, ent_sde_all = [], []
    deg_ode, deg_sde = [], []
    for start in range(0, N_EVAL, BS):
        bs = min(BS, N_EVAL - start)
        eo = decode_entropy(model, z_ode[start:start+bs], DEVICE)
        es = decode_entropy(model, z_sde[start:start+bs], DEVICE)
        ent_ode_all.append(eo)
        ent_sde_all.append(es)
        to = get_tokens(model, z_ode[start:start+bs], DEVICE)
        ts = get_tokens(model, z_sde[start:start+bs], DEVICE)
        for i in range(bs):
            deg_ode.append(is_degenerate(to[i]))
            deg_sde.append(is_degenerate(ts[i]))

    ent_ode = np.concatenate(ent_ode_all, axis=0)
    ent_sde = np.concatenate(ent_sde_all, axis=0)
    deg_ode = np.array(deg_ode)
    deg_sde = np.array(deg_sde)

    ode_deg_rate = float(deg_ode.mean())
    sde_deg_rate = float(deg_sde.mean())
    print(f"    ODE degeneracy: {100*ode_deg_rate:.1f}%, SDE degeneracy: {100*sde_deg_rate:.1f}%")

    # Entropy advantage (per token, ODE entropy - SDE entropy; >0 means SDE more confident)
    ent_gap = ent_ode - ent_sde
    uncapped_seq = ~(deg_ode | deg_sde)
    uncapped_tok = np.repeat(uncapped_seq[:, None], seq_len, axis=1)

    # All tokens
    sp_all, p_all = spearmanr(ode_curv.flatten(), ent_gap.flatten())
    # Exclude degenerate sequences
    if uncapped_tok.sum() > 100:
        sp_excl, p_excl = spearmanr(ode_curv[uncapped_tok], ent_gap[uncapped_tok])
    else:
        sp_excl, p_excl = float('nan'), float('nan')

    print(f"    Entropy-advantage correlation (all): r={sp_all:.3f}, p={p_all:.1e}")
    print(f"    Entropy-advantage correlation (excl): r={sp_excl:.3f}, p={p_excl:.1e}")

    return {
        "scale": scale_name,
        "params": sum(p.numel() for p in model.parameters()),
        "ode_deg_rate": ode_deg_rate,
        "sde_deg_rate": sde_deg_rate,
        "entropy_corr_all": {"r": float(sp_all), "p": float(p_all)},
        "entropy_corr_excl": {"r": float(sp_excl), "p": float(p_excl)},
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("K2 — Intermediate-Scale Curvature-Direction Check")
    print("=" * 60)

    embeddings, token_ids, vocab_size = prepare_real_data()
    print(f"  Data: {len(embeddings)} sequences, vocab={vocab_size}")

    results = []

    for i, scale_cfg in enumerate(SCALE_CONFIGS):
        name = scale_cfg["name"]
        print(f"\n{'='*40} Scale: {name} {'='*40}")

        if name == "275K":
            # Use existing C4 checkpoint
            ckpt_path = C4_RESULTS / "mini_elf_real_checkpoint.pt"
            if not ckpt_path.exists():
                print(f"  ERROR: C4 checkpoint not found at {ckpt_path}")
                continue
        else:
            ckpt_path = train_checkpoint(scale_cfg, embeddings, token_ids, vocab_size)

        model, cfg = load_ckpt(ckpt_path)
        scale_result = run_pipeline_for_scale(model, cfg, name)
        results.append(scale_result)

        # Free memory
        del model
        torch.mps.empty_cache() if DEVICE == "mps" else None

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Curvature-Overconfidence Trend Across Scales")
    print("=" * 60)
    for r in results:
        print(f"  {r['scale']:6s} ({r['params']:>10,} params): "
              f"r_all={r['entropy_corr_all']['r']:+.3f}, "
              f"r_excl={r['entropy_corr_excl']['r']:+.3f}, "
              f"ODE_deg={100*r['ode_deg_rate']:.1f}%, "
              f"SDE_deg={100*r['sde_deg_rate']:.1f}%")

    # Direction check
    signs_all = [np.sign(r["entropy_corr_all"]["r"]) for r in results if not np.isnan(r["entropy_corr_all"]["r"])]
    signs_excl = [np.sign(r["entropy_corr_excl"]["r"]) for r in results if not np.isnan(r["entropy_corr_excl"]["r"])]
    all_same_sign = len(set(signs_all)) == 1 if signs_all else False
    excl_same_sign = len(set(signs_excl)) == 1 if signs_excl else False

    magnitudes_excl = [r["entropy_corr_excl"]["r"] for r in results if not np.isnan(r["entropy_corr_excl"]["r"])]
    if len(magnitudes_excl) >= 2:
        trend = "strengthening" if magnitudes_excl[-1] > magnitudes_excl[0] else (
            "weakening" if magnitudes_excl[-1] < magnitudes_excl[0] else "flat")
    else:
        trend = "insufficient data"

    print(f"\n  Direction consistent (all tokens): {all_same_sign}")
    print(f"  Direction consistent (excl degen): {excl_same_sign}")
    print(f"  Magnitude trend (excl degen): {trend}")

    with open(RESULTS_DIR / "testK2_scale_trend.json", "w") as f:
        json.dump({"scales": results, "direction_consistent_all": all_same_sign,
                   "direction_consistent_excl": excl_same_sign, "magnitude_trend": trend}, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    params = [r["params"] for r in results]
    r_all = [r["entropy_corr_all"]["r"] for r in results]
    r_excl = [r["entropy_corr_excl"]["r"] for r in results]

    ax = axes[0]
    ax.plot(params, r_all, "o-", color="#4C72B0", markersize=10, label="All tokens")
    ax.plot(params, r_excl, "s--", color="#C44E52", markersize=10, label="Excl. degenerate")
    ax.axhline(0, color="k", ls=":", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters")
    ax.set_ylabel("Spearman r (Curvature vs Entropy Advantage)")
    ax.set_title("Curvature-Overconfidence Correlation vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    deg_ode = [100 * r["ode_deg_rate"] for r in results]
    deg_sde = [100 * r["sde_deg_rate"] for r in results]
    ax.plot(params, deg_ode, "o-", color="#DD8452", markersize=10, label="ODE")
    ax.plot(params, deg_sde, "s--", color="#55A868", markersize=10, label="SDE")
    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters")
    ax.set_ylabel("Degeneracy Rate (%)")
    ax.set_title("Degeneracy Rate vs Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testK2_scale_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testK2_scale_trend.json'}")
    print(f"  Saved → {RESULTS_DIR / 'testK2_scale_trend.png'}")


if __name__ == "__main__":
    main()
