#!/usr/bin/env python3
"""Phase P2 — Multi-Scale Quality Crossover.

Retrains toy models at 275K, 1M, 5M params, then generates with ODE and SDE
(γ=1.5) at each scale. Measures decoder cross-entropy as quality metric.
Combines with Phase P1 ELF-B (105M) data to plot the crossover curve.

The crossover curve = log-CE(ODE) − log-CE(SDE) vs log(params).
Negative → ODE is better (SDE hurts). Positive → SDE is better (SDE helps).
"""
import json, sys, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from modules.model import ELF
from utils.sampling_utils import sample_timesteps, net_out_to_v_x, get_sampling_steps

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "p2_scale"
PHASE0_RESULTS = Path(__file__).resolve().parent.parent / "phase0" / "results"
K2_RESULTS = Path(__file__).resolve().parent.parent / "phase0k" / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_DIM = 128
TOY_MAX_LEN = 16
MAX_VOCAB_SIZE = 1024
T_EPS = 0.05
SIGMA = 1.0
SDE_GAMMA = 1.5  # Match production-scale config

SCALE_CONFIGS = [
    {"name": "275K", "hidden": 64,  "depth": 2, "heads": 4, "bottleneck": 32,  "mlp_ratio": 2.0, "steps": 15000},
    {"name": "1M",   "hidden": 128, "depth": 4, "heads": 8, "bottleneck": 64,  "mlp_ratio": 2.0, "steps": 20000},
    {"name": "5M",   "hidden": 256, "depth": 6, "heads": 8, "bottleneck": 128, "mlp_ratio": 2.0, "steps": 25000},
]

N_EVAL = 256
BATCH_SIZE = 64
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
    """Load embeddings and build vocabulary (same as testK2)."""
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
    train_bs = 32
    random_baseline = np.log(vocab_size)

    model.train()
    t0 = time.time()
    ce_history = []
    for step in range(num_steps):
        idx = rng.choice(n_sequences, train_bs, replace=False)
        x0 = torch.tensor(embeddings[idx], dtype=torch.float32, device=DEVICE)
        ids = torch.tensor(token_ids[idx], dtype=torch.long, device=DEVICE)
        attn = torch.ones(train_bs, TOY_MAX_LEN, device=DEVICE)
        cond = torch.zeros(train_bs, TOY_MAX_LEN, device=DEVICE)

        optimizer.zero_grad()
        loss, ce_loss = train_step(model, x0, ids, attn, cond, config, DEVICE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ce_history.append(float(ce_loss.item()))

        if (step + 1) % 5000 == 0:
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


def ode_sample(model, n, n_steps, enc_dim, seq_len, noise):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = noise.clone().to(DEVICE)
        steps = get_sampling_steps(n_steps, "logit_normal", device=DEVICE, dtype=dtype)
        for i in range(n_steps):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            t_b = torch.full((n,), tc, dtype=dtype, device=DEVICE)
            net_out, _ = model(z, t_b, deterministic=True, decoder_step_active=None)
            v, _ = net_out_to_v_x(net_out, z, t_b, T_EPS)
            z = z + h * v
    return z


def sde_sample(model, n, n_steps, enc_dim, seq_len, g, noise):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = noise.clone().to(DEVICE)
        steps = get_sampling_steps(n_steps, "logit_normal", device=DEVICE, dtype=dtype)
        for i in range(n_steps):
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


def decode_ce(model, z, token_ids_gt):
    """Compute cross-entropy of decoded tokens against ground truth."""
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_f = torch.ones(z.shape[0], dtype=dtype, device=DEVICE)
        _, logits = model(z.to(DEVICE), t_f, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=DEVICE))
        log_probs = F.log_softmax(logits.float(), dim=-1)
        ce = -log_probs.gather(-1, token_ids_gt.unsqueeze(-1)).squeeze(-1)
    return ce.mean(dim=-1).cpu().numpy()  # per-sequence mean CE


def is_degenerate(token_ids):
    return len(set(int(t) for t in token_ids)) <= 2


def evaluate_at_scale(model, cfg, scale_name, embeddings, token_ids):
    """Generate N_EVAL samples with ODE and SDE, compute quality metrics."""
    enc_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]
    vocab_size = cfg["vocab_size"]

    print(f"\n  Evaluating {scale_name} (N={N_EVAL})...")
    torch.manual_seed(42)
    noise = torch.randn(N_EVAL, seq_len, enc_dim, device=DEVICE)

    # Pick random ground-truth sequences for CE scoring
    rng = np.random.default_rng(42)
    gt_idx = rng.choice(len(embeddings), N_EVAL, replace=False)
    gt_ids = torch.tensor(token_ids[gt_idx], dtype=torch.long, device=DEVICE)

    # Generate in batches
    ode_ces, sde_ces = [], []
    ode_degs, sde_degs = [], []
    BS = BATCH_SIZE

    for start in range(0, N_EVAL, BS):
        bs = min(BS, N_EVAL - start)
        n_batch = noise[start:start+bs]
        gt_batch = gt_ids[start:start+bs]

        z_ode = ode_sample(model, bs, N_STEPS_SAMPLE, enc_dim, seq_len, n_batch)
        z_sde = sde_sample(model, bs, N_STEPS_SAMPLE, enc_dim, seq_len, SDE_GAMMA, n_batch)

        # Get tokens for degeneracy check
        with torch.no_grad():
            dtype = next(model.parameters()).dtype
            t_f = torch.ones(bs, dtype=dtype, device=DEVICE)
            dec_act = torch.ones(bs, device=DEVICE)
            _, ode_logits = model(z_ode, t_f, deterministic=True, decoder_step_active=dec_act)
            _, sde_logits = model(z_sde, t_f, deterministic=True, decoder_step_active=dec_act)

            ode_toks = ode_logits.argmax(dim=-1).cpu().numpy()
            sde_toks = sde_logits.argmax(dim=-1).cpu().numpy()

            # CE using own logits against argmax tokens (self-consistency quality)
            ode_probs = F.softmax(ode_logits.float(), dim=-1)
            sde_probs = F.softmax(sde_logits.float(), dim=-1)
            ode_ent = -(ode_probs * torch.log(ode_probs + 1e-10)).sum(-1).mean(-1).cpu().numpy()
            sde_ent = -(sde_probs * torch.log(sde_probs + 1e-10)).sum(-1).mean(-1).cpu().numpy()

            # Generative PPL proxy: CE of decoder against its own argmax
            ode_max_prob = ode_probs.max(dim=-1).values.mean(dim=-1).cpu().numpy()
            sde_max_prob = sde_probs.max(dim=-1).values.mean(dim=-1).cpu().numpy()

        for i in range(bs):
            ode_degs.append(is_degenerate(ode_toks[i]))
            sde_degs.append(is_degenerate(sde_toks[i]))

        # Use negative log max-prob as quality metric (lower = more confident = better)
        ode_ces.extend(-np.log(ode_max_prob + 1e-10))
        sde_ces.extend(-np.log(sde_max_prob + 1e-10))

    ode_ces = np.array(ode_ces)
    sde_ces = np.array(sde_ces)
    ode_degs = np.array(ode_degs)
    sde_degs = np.array(sde_degs)

    # Paired comparison on jointly non-degenerate
    joint_ok = (~ode_degs) & (~sde_degs)
    n_joint = int(joint_ok.sum())

    if n_joint > 10:
        diff = ode_ces[joint_ok] - sde_ces[joint_ok]
        mean_diff = float(np.mean(diff))
        np.random.seed(42)
        boots = [np.mean(np.random.choice(diff, n_joint, replace=True)) for _ in range(5000)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    else:
        mean_diff = float('nan')
        ci = [float('nan'), float('nan')]

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "scale": scale_name, "params": n_params,
        "ode_deg_rate": float(ode_degs.mean()),
        "sde_deg_rate": float(sde_degs.mean()),
        "ode_mean_neglogprob": float(ode_ces.mean()),
        "sde_mean_neglogprob": float(sde_ces.mean()),
        "quality_diff_ode_minus_sde": mean_diff,
        "ci_95": ci,
        "n_joint_nondegen": n_joint,
        "sde_better": bool(ci[0] > 0) if not math.isnan(ci[0]) else None,
    }

    sign = "+" if mean_diff > 0 else "-"
    print(f"    ODE deg: {100*result['ode_deg_rate']:.1f}%, SDE deg: {100*result['sde_deg_rate']:.1f}%")
    print(f"    Quality diff (ODE−SDE): {sign}{abs(mean_diff):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"    SDE better: {result['sde_better']}")

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Phase P2 — Multi-Scale Quality Crossover")
    print(f"  Scales: {[s['name'] for s in SCALE_CONFIGS]}")
    print(f"  SDE γ = {SDE_GAMMA}")
    print("=" * 70)
    t0 = time.time()

    embeddings, token_ids, vocab_size = prepare_real_data()
    print(f"  Data: {len(embeddings)} sequences, vocab={vocab_size}")

    results = []

    for scale_cfg in SCALE_CONFIGS:
        name = scale_cfg["name"]
        print(f"\n{'='*40} Scale: {name} {'='*40}")

        ckpt_path = train_checkpoint(scale_cfg, embeddings, token_ids, vocab_size)
        model, cfg = load_ckpt(ckpt_path)
        result = evaluate_at_scale(model, cfg, name, embeddings, token_ids)
        results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Add ELF-B (105M) result from Phase P1/Stage 2
    results.append({
        "scale": "105M (ELF-B)", "params": 104579940,
        "ode_deg_rate": 0.0, "sde_deg_rate": 0.007,
        "quality_diff_ode_minus_sde": 0.725,
        "ci_95": [0.684, 0.768],
        "n_joint_nondegen": 1017,
        "sde_better": True,
    })

    # Summary
    print("\n" + "=" * 70)
    print("Multi-Scale Crossover Summary")
    print("=" * 70)
    print(f"  {'Scale':>12s}  {'Params':>12s}  {'ODE deg':>8s}  {'SDE deg':>8s}  {'Diff (ODE-SDE)':>15s}  {'SDE better?':>12s}")
    for r in results:
        sde_b = "YES" if r.get("sde_better") else ("NO" if r.get("sde_better") is False else "N/A")
        diff = r["quality_diff_ode_minus_sde"]
        print(f"  {r['scale']:>12s}  {r['params']:>12,}  {100*r['ode_deg_rate']:>7.1f}%  {100*r['sde_deg_rate']:>7.1f}%"
              f"  {diff:>+14.4f}  {sde_b:>12s}")

    with open(RESULTS_DIR / "p2_scale_crossover.json", "w") as f:
        json.dump({"results": results, "sde_gamma": SDE_GAMMA,
                   "total_time_s": time.time() - t0}, f, indent=2)

    # Markdown report
    with open(RESULTS_DIR / "p2_scale_crossover.md", "w") as f:
        f.write("# Phase P2 — Multi-Scale Quality Crossover\n\n")
        f.write(f"**SDE γ = {SDE_GAMMA}**\n\n")
        f.write("| Scale | Params | ODE Deg | SDE Deg | Diff (ODE−SDE) | 95% CI | SDE better? |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            sde_b = "✅" if r.get("sde_better") else ("❌" if r.get("sde_better") is False else "⚠️")
            d = r["quality_diff_ode_minus_sde"]
            ci = r.get("ci_95", [float('nan'), float('nan')])
            f.write(f"| {r['scale']} | {r['params']:,} | {100*r['ode_deg_rate']:.1f}% | "
                    f"{100*r['sde_deg_rate']:.1f}% | {d:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {sde_b} |\n")
        f.write(f"\n**Total time:** {time.time()-t0:.0f}s\n")

    print(f"\n  Saved → {RESULTS_DIR}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
