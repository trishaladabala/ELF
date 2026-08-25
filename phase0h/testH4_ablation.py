#!/usr/bin/env python3
"""H4 — Naive-Masking vs Confidence-Gated Re-Corruption: Head-to-Head.

Compares zero-vector masking vs ELF re-corruption as the revision mechanism,
using the same H3 convergence loop.
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

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PHASE0C_RESULTS = Path(__file__).resolve().parent.parent / "phase0c" / "results"
PHASE0_RESULTS = Path(__file__).resolve().parent.parent / "phase0" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

N_SAMPLES = 128
N_PASSES = 5
ENTROPY_THRESHOLD_PERCENTILE = 75


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
        tokens = logits.argmax(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return tokens.cpu().numpy(), entropy.cpu().numpy()


def inject_errors(embeddings, token_ids, kmeans, n_per_seq=3):
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


def run_refinement_loop(model, z_init, token_ids, original_errors, method, noise_scale,
                        n_passes=N_PASSES, seq_len=16):
    """Run the full refinement loop with a given revision method."""
    z = z_init.clone()
    pass_metrics = []

    for pass_num in range(n_passes + 1):
        tokens, entropy = decode_batch(model, z, DEVICE)

        errors_remaining = set()
        for i in range(z.shape[0]):
            for j in range(seq_len):
                if tokens[i, j] != token_ids[i, j]:
                    errors_remaining.add((i, j))

        orig_remaining = len(original_errors & errors_remaining)
        new_errors = len(errors_remaining - original_errors)

        thresh = np.percentile(entropy, ENTROPY_THRESHOLD_PERCENTILE)
        flagged = entropy > thresh

        pass_metrics.append({
            "pass": pass_num,
            "total_errors": len(errors_remaining),
            "original_remaining": orig_remaining,
            "new_errors": new_errors,
            "n_flagged": int(flagged.sum()),
        })

        if pass_num >= n_passes:
            break

        positions_to_fix = []
        for i in range(z.shape[0]):
            for j in range(seq_len):
                if flagged[i, j]:
                    positions_to_fix.append((i, j))

        if not positions_to_fix:
            break

        # Apply the revision method
        z_new = z.clone()
        for (i, j) in positions_to_fix:
            if method == "zero_mask":
                z_new[i, j] = 0.0
            elif method == "re_corruption":
                p = 0.3
                eps = torch.randn(z.shape[-1], device=z.device) * noise_scale
                z_new[i, j] = p * z[i, j] + (1 - p) * eps
        z = z_new

    return pass_metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("H4 — Naive Masking vs Re-Corruption Ablation")
    print("=" * 60)

    model, cfg = load_real_checkpoint()

    print("Loading real sequences...")
    embeddings, token_ids, kmeans = load_real_data()
    seq_len = embeddings.shape[1]

    err_emb, err_ids, original_errors = inject_errors(embeddings, token_ids, kmeans)
    print(f"  Injected {len(original_errors)} errors")

    z_init = torch.tensor(err_emb, dtype=torch.float32, device=DEVICE)
    noise_scale = float(z_init.std())

    methods = {"zero_mask": "Naive Zero-Vector Mask", "re_corruption": "ELF Re-Corruption"}
    all_results = {}

    for method, desc in methods.items():
        print(f"\n── {desc} ──")
        metrics = run_refinement_loop(
            model, z_init, token_ids, original_errors, method, noise_scale, seq_len=seq_len
        )
        for m in metrics:
            print(f"  Pass {m['pass']}: total={m['total_errors']}, "
                  f"orig_remaining={m['original_remaining']}/{len(original_errors)}, "
                  f"new={m['new_errors']}")
        all_results[method] = metrics

    # Compare final state
    print(f"\n{'=' * 60}")
    print("HEAD-TO-HEAD COMPARISON (after 5 passes)")
    print(f"{'=' * 60}")

    for method, desc in methods.items():
        final = all_results[method][-1]
        initial = all_results[method][0]
        error_reduction = initial["total_errors"] - final["total_errors"]
        print(f"  {desc}:")
        print(f"    Initial errors:   {initial['total_errors']}")
        print(f"    Final errors:     {final['total_errors']}")
        print(f"    Net change:       {error_reduction:+d}")
        print(f"    New errors:       {final['new_errors']}")
        print(f"    Orig remaining:   {final['original_remaining']}/{len(original_errors)}")

    zero_final = all_results["zero_mask"][-1]["total_errors"]
    recorr_final = all_results["re_corruption"][-1]["total_errors"]

    if recorr_final < zero_final - 10:
        winner = "re_corruption"
        print(f"\n  → Re-corruption clearly outperforms naive masking.")
    elif zero_final < recorr_final - 10:
        winner = "zero_mask"
        print(f"\n  → Naive masking outperforms re-corruption (simplifies the method).")
    else:
        winner = "tie"
        print(f"\n  → No clear winner — methods are comparable.")
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_m = {"zero_mask": "#C44E52", "re_corruption": "#4C72B0"}

    for method, desc in methods.items():
        metrics = all_results[method]
        passes = [m["pass"] for m in metrics]
        total_errs = [m["total_errors"] for m in metrics]
        new_errs = [m["new_errors"] for m in metrics]

        axes[0].plot(passes, total_errs, 'o-', color=colors_m[method], lw=2,
                     markersize=8, label=desc)
        axes[1].plot(passes, new_errs, 's-', color=colors_m[method], lw=2,
                     markersize=8, label=desc)

    axes[0].set_xlabel("Pass"); axes[0].set_ylabel("Total Errors")
    axes[0].set_title("Total Errors Over Passes"); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Pass"); axes[1].set_ylabel("New Errors Introduced")
    axes[1].set_title("New Errors Over Passes"); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("H4: Zero Mask vs Re-Corruption", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testH4_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testH4_ablation.png'}")

    with open(RESULTS_DIR / "testH4.json", "w") as f:
        json.dump({
            "methods": all_results,
            "winner": winner,
            "n_original_errors": len(original_errors),
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
