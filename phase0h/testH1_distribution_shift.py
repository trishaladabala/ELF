#!/usr/bin/env python3
"""H1 — Distribution-Shift Probe: Which Revision Input Does the Frozen Decoder Tolerate?

Tests whether the frozen decode-mode branch can produce sensible output when
given corrupted/revised embeddings (the core assumption of Idea 15).
"""

import json
import sys
import time
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
PHASE0C_RESULTS = Path(__file__).resolve().parent.parent / "phase0c" / "results"
PHASE0_RESULTS = Path(__file__).resolve().parent.parent / "phase0" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

T_EPS = 0.05
N_SAMPLES = 128


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
    """Load real embeddings and cluster token IDs from C4."""
    x_128 = np.load(PHASE0_RESULTS / "x_128.npy")
    seq_ids = np.load(PHASE0_RESULTS / "seq_ids.npy")
    pos_ids = np.load(PHASE0_RESULTS / "pos_ids.npy")

    # Reproduce C4's KMeans
    from sklearn.cluster import MiniBatchKMeans
    n_clusters = 1024
    subsample_size = min(50000, x_128.shape[0])
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(x_128.shape[0], subsample_size, replace=False)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(x_128[sub_idx])

    # Group into sequences of length 16
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
            np.array(sequences_ids, dtype=np.int64))


def decode_batch(model, z, device):
    """Run model in decode mode and return logits + predicted tokens."""
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values  # (B, S)
        tokens = logits.argmax(dim=-1)        # (B, S)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B, S)
    return tokens.cpu().numpy(), max_prob.cpu().numpy(), entropy.cpu().numpy()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("H1 — Distribution-Shift Probe")
    print("=" * 60)

    model, cfg = load_real_checkpoint()
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]
    vocab_size = cfg["vocab_size"]

    print("Loading real sequences...")
    embeddings, token_ids = load_real_data()
    print(f"  Loaded {len(embeddings)} sequences of shape {embeddings.shape}")

    x_clean = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
    gt_tokens = token_ids

    # 1. Baseline: decode clean embeddings
    print("\n── Baseline: Clean embeddings ──")
    clean_tokens, clean_conf, clean_entropy = decode_batch(model, x_clean, DEVICE)
    accuracy = (clean_tokens == gt_tokens).mean()
    print(f"  Accuracy vs ground truth: {accuracy:.4f}")
    print(f"  Mean confidence (max_p): {clean_conf.mean():.4f}")
    print(f"  Mean entropy: {clean_entropy.mean():.4f}")

    # Identify high-confidence correct tokens for the probe
    correct_mask = (clean_tokens == gt_tokens)  # (N, S)
    high_conf_mask = clean_conf > 0.7
    good_mask = correct_mask & high_conf_mask
    n_good = good_mask.sum()
    print(f"  High-confidence correct tokens: {n_good}/{correct_mask.size}")

    results = {"baseline": {
        "accuracy": float(accuracy),
        "mean_confidence": float(clean_conf.mean()),
        "mean_entropy": float(clean_entropy.mean()),
        "n_good_tokens": int(n_good),
    }}

    # Probe each candidate revision transformation
    revision_candidates = {
        "zero_vector": "Hard zero vector",
        "gaussian_noise": "Fresh Gaussian noise (training scale)",
        "re_corruption": "ELF re-corruption (z̃ = p·x + (1-p)·ε)",
    }

    # For each candidate, we perturb ONLY the "good" tokens and check behavior
    for cand_name, cand_desc in revision_candidates.items():
        print(f"\n── Candidate: {cand_desc} ──")

        x_perturbed = x_clean.clone()
        for i in range(x_perturbed.shape[0]):
            for j in range(x_perturbed.shape[1]):
                if good_mask[i, j]:
                    if cand_name == "zero_vector":
                        x_perturbed[i, j] = 0.0
                    elif cand_name == "gaussian_noise":
                        x_perturbed[i, j] = torch.randn(encoder_dim, device=DEVICE) * x_clean.std()
                    elif cand_name == "re_corruption":
                        # z̃ = p·x + (1-p)·ε, with p=0.3 (high noise, forcing revision)
                        p = 0.3
                        eps = torch.randn(encoder_dim, device=DEVICE) * x_clean.std()
                        x_perturbed[i, j] = p * x_clean[i, j] + (1 - p) * eps

        perturbed_tokens, perturbed_conf, perturbed_entropy = decode_batch(model, x_perturbed, DEVICE)

        # Metrics on the perturbed (good) positions
        stayed_same = 0
        changed_to_valid = 0  # Changed but still plausible
        collapsed_uniform = 0
        total_good = 0

        for i in range(x_perturbed.shape[0]):
            for j in range(x_perturbed.shape[1]):
                if good_mask[i, j]:
                    total_good += 1
                    if perturbed_tokens[i, j] == clean_tokens[i, j]:
                        stayed_same += 1
                    else:
                        changed_to_valid += 1

        # Check confidence on perturbed positions
        perturbed_good_conf = perturbed_conf[good_mask]
        perturbed_good_entropy = perturbed_entropy[good_mask]

        # Check unperturbed positions (should be unaffected by attention)
        unperturbed_conf = perturbed_conf[~good_mask]
        clean_unperturbed_conf = clean_conf[~good_mask]
        conf_drift = float(np.abs(unperturbed_conf - clean_unperturbed_conf).mean())

        uniform_entropy = np.log(vocab_size)
        near_uniform_frac = float((perturbed_good_entropy > 0.9 * uniform_entropy).mean())

        print(f"  On perturbed positions (n={total_good}):")
        print(f"    Stayed same:     {stayed_same}/{total_good} ({100*stayed_same/max(total_good,1):.1f}%)")
        print(f"    Changed:         {changed_to_valid}/{total_good} ({100*changed_to_valid/max(total_good,1):.1f}%)")
        print(f"    Mean confidence: {perturbed_good_conf.mean():.4f} (was {clean_conf[good_mask].mean():.4f})")
        print(f"    Mean entropy:    {perturbed_good_entropy.mean():.4f} (was {clean_entropy[good_mask].mean():.4f})")
        print(f"    Near-uniform:    {near_uniform_frac*100:.1f}% of perturbed tokens")
        print(f"  Unperturbed positions confidence drift: {conf_drift:.4f}")

        # Classify behavior
        if near_uniform_frac > 0.8:
            behavior = "degenerate_uniform"
            desc = "Output collapsed to near-uniform — NOT usable"
        elif stayed_same / max(total_good, 1) > 0.8:
            behavior = "near_identity"
            desc = "Near-identity: perturbation ignored — NOT useful for revision"
        elif perturbed_good_conf.mean() > 0.3 and near_uniform_frac < 0.3:
            behavior = "sensible"
            desc = "Sensible: confident, non-uniform, changed output — USABLE"
        else:
            behavior = "mixed"
            desc = "Mixed behavior — needs closer look"

        print(f"  → Behavior: {desc}")

        results[cand_name] = {
            "description": cand_desc,
            "total_probed": total_good,
            "stayed_same": stayed_same, "changed": changed_to_valid,
            "mean_confidence": float(perturbed_good_conf.mean()),
            "mean_entropy": float(perturbed_good_entropy.mean()),
            "near_uniform_fraction": near_uniform_frac,
            "unperturbed_drift": conf_drift,
            "behavior": behavior,
        }

    # Go/No-Go
    any_sensible = any(
        results[c]["behavior"] == "sensible"
        for c in revision_candidates
    )

    print(f"\n{'=' * 60}")
    if any_sensible:
        best = [c for c in revision_candidates if results[c]["behavior"] == "sensible"]
        print(f"✅ H1 PASS: {len(best)} candidate(s) produce sensible output.")
        for c in best:
            print(f"   → {revision_candidates[c]}")
        print("   Proceed to H2.")
        results["verdict"] = "PASS"
        results["best_candidates"] = best
    else:
        print("❌ H1 FAIL: No candidate produces sensible, non-degenerate revision output.")
        print("   The frozen decoder cannot support post-hoc revision.")
        print("   Idea 15 needs to be rescoped as a training-time change (closer to Idea 5).")
        results["verdict"] = "FAIL"
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cand_names = list(revision_candidates.keys())
    for idx, cname in enumerate(cand_names):
        ax = axes[idx]
        r = results[cname]
        categories = ['Stayed Same', 'Changed', 'Near-Uniform']
        values = [r["stayed_same"], r["changed"],
                  int(r["near_uniform_fraction"] * r["total_probed"])]
        colors_bar = ['#55A868', '#4C72B0', '#C44E52']
        ax.bar(categories, values, color=colors_bar, alpha=0.8)
        ax.set_title(revision_candidates[cname], fontsize=10)
        ax.set_ylabel("Count")
        ax.annotate(f'conf={r["mean_confidence"]:.3f}', xy=(0.95, 0.95),
                    xycoords='axes fraction', ha='right', va='top', fontsize=9)

    plt.suptitle("H1: Distribution-Shift Probe — Revision Candidate Behavior",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testH1_distribution_shift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testH1_distribution_shift.png'}")

    with open(RESULTS_DIR / "testH1.json", "w") as f:
        json.dump(results, f, indent=2)

    return 0 if any_sensible else 1


if __name__ == "__main__":
    sys.exit(main())
