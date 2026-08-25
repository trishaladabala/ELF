#!/usr/bin/env python3
"""H3 — Refinement Loop Convergence and Stability.

Tests whether iterative correction (detect → re-corrupt → re-decode) converges
or oscillates/diverges.
"""

import json
import sys
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
from utils.sampling_utils import net_out_to_v_x

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PHASE0C_RESULTS = Path(__file__).resolve().parent.parent / "phase0c" / "results"
PHASE0_RESULTS = Path(__file__).resolve().parent.parent / "phase0" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

N_SAMPLES = 128
N_PASSES = 5
ENTROPY_THRESHOLD_PERCENTILE = 75  # Flag top 25% entropy positions


def load_real_checkpoint():
    ckpt_path = PHASE0C_RESULTS / "mini_elf_real_checkpoint.pt"
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


def load_real_data():
    x_128 = np.load(PHASE0_RESULTS / "x_128.npy")
    seq_ids = np.load(PHASE0_RESULTS / "seq_ids.npy")
    pos_ids = np.load(PHASE0_RESULTS / "pos_ids.npy")

    from sklearn.cluster import MiniBatchKMeans
    n_clusters = 1024
    subsample_size = min(50000, x_128.shape[0])
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(x_128.shape[0], subsample_size, replace=False)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(x_128[sub_idx])

    SEQ_LEN = 16
    unique_seqs = np.unique(seq_ids)
    sequences_emb = []
    sequences_ids = []
    for sid in unique_seqs:
        mask = seq_ids == sid
        embs = x_128[mask]
        positions = pos_ids[mask]
        order = np.argsort(positions)
        embs = embs[order]
        if len(embs) >= SEQ_LEN:
            embs = embs[:SEQ_LEN]
            cids = kmeans.predict(embs)
            sequences_emb.append(embs)
            sequences_ids.append(cids)
        if len(sequences_emb) >= N_SAMPLES:
            break

    return (np.array(sequences_emb, dtype=np.float32),
            np.array(sequences_ids, dtype=np.int64),
            kmeans)


def decode_batch(model, z, device):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values
        tokens = logits.argmax(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return tokens.cpu().numpy(), max_prob.cpu().numpy(), entropy.cpu().numpy()


def inject_errors(embeddings, token_ids, kmeans, n_per_seq=3):
    """Inject substitution errors for the convergence test."""
    rng = np.random.default_rng(99)
    emb_out = embeddings.copy()
    ids_out = token_ids.copy()
    error_positions = set()

    for i in range(len(embeddings)):
        seq_len = embeddings.shape[1]
        error_pos = rng.choice(seq_len, size=min(n_per_seq, seq_len), replace=False)
        for pos in error_pos:
            orig_id = token_ids[i, pos]
            new_id = rng.integers(0, kmeans.n_clusters)
            while new_id == orig_id:
                new_id = rng.integers(0, kmeans.n_clusters)
            emb_out[i, pos] = kmeans.cluster_centers_[new_id]
            ids_out[i, pos] = new_id
            error_positions.add((i, int(pos)))

    return emb_out, ids_out, error_positions


def re_corrupt(z, positions_to_fix, p=0.3, noise_scale=1.0):
    """Re-corrupt specified positions using ELF-style re-corruption."""
    z_out = z.clone()
    for (i, j) in positions_to_fix:
        eps = torch.randn(z.shape[-1], device=z.device) * noise_scale
        z_out[i, j] = p * z[i, j] + (1 - p) * eps
    return z_out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("H3 — Refinement Loop Convergence")
    print("=" * 60)

    model, cfg = load_real_checkpoint()
    vocab_size = cfg["vocab_size"]

    print("Loading real sequences...")
    embeddings, token_ids, kmeans = load_real_data()
    seq_len = embeddings.shape[1]

    # Inject errors
    err_emb, err_ids, original_errors = inject_errors(embeddings, token_ids, kmeans)
    print(f"  Injected {len(original_errors)} errors across {N_SAMPLES} sequences")

    z = torch.tensor(err_emb, dtype=torch.float32, device=DEVICE)
    noise_scale = float(z.std())

    pass_metrics = []

    for pass_num in range(N_PASSES + 1):  # Pass 0 = initial state before any correction
        tokens, conf, entropy = decode_batch(model, z, DEVICE)

        # Count errors: positions where decoded token != ground truth
        errors_remaining = set()
        for i in range(N_SAMPLES):
            for j in range(seq_len):
                if tokens[i, j] != token_ids[i, j]:
                    errors_remaining.add((i, j))

        # Count how many of the ORIGINAL injected errors remain
        orig_remaining = len(original_errors & errors_remaining)
        # Count NEW errors (not in the original set)
        new_errors = errors_remaining - original_errors
        n_new = len(new_errors)

        # Count flagged positions (high entropy)
        thresh = np.percentile(entropy, ENTROPY_THRESHOLD_PERCENTILE)
        flagged = entropy > thresh
        n_flagged = int(flagged.sum())

        metrics = {
            "pass": pass_num,
            "total_errors": len(errors_remaining),
            "original_errors_remaining": orig_remaining,
            "new_errors_introduced": n_new,
            "n_flagged": n_flagged,
            "mean_entropy": float(entropy.mean()),
            "threshold": float(thresh),
        }
        pass_metrics.append(metrics)

        print(f"  Pass {pass_num}: "
              f"total_errors={len(errors_remaining)}, "
              f"orig_remaining={orig_remaining}/{len(original_errors)}, "
              f"new_errors={n_new}, "
              f"flagged={n_flagged}, "
              f"mean_ent={entropy.mean():.4f}")

        if pass_num >= N_PASSES:
            break

        # Apply refinement: re-corrupt flagged positions and re-decode
        positions_to_fix = []
        for i in range(N_SAMPLES):
            for j in range(seq_len):
                if flagged[i, j]:
                    positions_to_fix.append((i, j))

        if len(positions_to_fix) == 0:
            print(f"  No positions flagged — stopping early at pass {pass_num}")
            break

        z = re_corrupt(z, positions_to_fix, p=0.3, noise_scale=noise_scale)

    # Analysis
    total_errors = [m["total_errors"] for m in pass_metrics]
    new_errors = [m["new_errors_introduced"] for m in pass_metrics]
    orig_remaining = [m["original_errors_remaining"] for m in pass_metrics]

    monotonic_decrease = all(
        total_errors[i+1] <= total_errors[i] for i in range(len(total_errors)-1)
    )
    # Allow plateau: errors decrease or stay same
    plateau_or_decrease = all(
        total_errors[i+1] <= total_errors[i] + 5 for i in range(len(total_errors)-1)
    )
    new_error_rate_low = all(ne / max(N_SAMPLES * seq_len, 1) < 0.05 for ne in new_errors)
    new_error_growing = (len(new_errors) >= 3 and
                         new_errors[-1] > new_errors[1] * 1.5)

    print(f"\n{'=' * 60}")
    if monotonic_decrease and new_error_rate_low:
        print("✅ H3 PASS: Error count decreases monotonically, new-error rate is low.")
        verdict = "PASS"
    elif plateau_or_decrease and not new_error_growing:
        print("🟡 H3 PARTIAL: Errors plateau/decrease, new-error rate acceptable.")
        verdict = "PARTIAL_PASS"
    else:
        if new_error_growing:
            print("⚠️  H3: New-error rate grows with passes — needs a stopping rule.")
        else:
            print("⚠️  H3: Error count oscillates — needs dampening.")
        verdict = "NEEDS_FIX"
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    passes = [m["pass"] for m in pass_metrics]

    axes[0].plot(passes, total_errors, 'o-', color='#C44E52', lw=2, markersize=8)
    axes[0].set_xlabel("Pass"); axes[0].set_ylabel("Total Errors")
    axes[0].set_title("Total Error Count"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(passes, orig_remaining, 's-', color='#4C72B0', lw=2, markersize=8,
                 label='Original errors remaining')
    axes[1].plot(passes, new_errors, '^-', color='#DD8452', lw=2, markersize=8,
                 label='New errors introduced')
    axes[1].set_xlabel("Pass"); axes[1].set_ylabel("Count")
    axes[1].set_title("Error Breakdown"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    mean_ent = [m["mean_entropy"] for m in pass_metrics]
    axes[2].plot(passes, mean_ent, 'D-', color='#55A868', lw=2, markersize=8)
    axes[2].set_xlabel("Pass"); axes[2].set_ylabel("Mean Entropy")
    axes[2].set_title("Mean Entropy Over Passes"); axes[2].grid(True, alpha=0.3)

    plt.suptitle("H3: Refinement Loop Convergence", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testH3_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testH3_convergence.png'}")

    with open(RESULTS_DIR / "testH3.json", "w") as f:
        json.dump({
            "pass_metrics": pass_metrics,
            "monotonic_decrease": monotonic_decrease,
            "new_error_rate_low": new_error_rate_low,
            "verdict": verdict,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
