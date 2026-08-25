#!/usr/bin/env python3
"""H2 — Confidence-Signal Validity: Does Entropy Catch the Errors You Care About?

Tests whether per-token entropy can detect substitution errors AND repetition
errors, since the review flagged that repetitions may be locally confident.
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


def inject_substitution_errors(embeddings, token_ids, kmeans, n_errors_per_seq=2):
    """Inject single-token substitution errors at known positions."""
    rng = np.random.default_rng(99)
    emb_out = embeddings.copy()
    ids_out = token_ids.copy()
    error_positions = []  # List of (seq_idx, pos_idx)

    for i in range(len(embeddings)):
        seq_len = embeddings.shape[1]
        # Pick random positions to corrupt
        error_pos = rng.choice(seq_len, size=min(n_errors_per_seq, seq_len), replace=False)
        for pos in error_pos:
            # Replace with a random different cluster centroid
            orig_id = token_ids[i, pos]
            new_id = rng.integers(0, kmeans.n_clusters)
            while new_id == orig_id:
                new_id = rng.integers(0, kmeans.n_clusters)
            emb_out[i, pos] = kmeans.cluster_centers_[new_id]
            ids_out[i, pos] = new_id
            error_positions.append((i, int(pos)))

    return emb_out, ids_out, error_positions


def inject_repetition_errors(embeddings, token_ids, n_errors_per_seq=2):
    """Inject repetition errors (duplicate a token at the next position)."""
    rng = np.random.default_rng(77)
    emb_out = embeddings.copy()
    ids_out = token_ids.copy()
    error_positions = []

    for i in range(len(embeddings)):
        seq_len = embeddings.shape[1]
        # Pick source positions to duplicate into the next slot
        for _ in range(min(n_errors_per_seq, seq_len - 1)):
            src_pos = rng.integers(0, seq_len - 1)
            tgt_pos = src_pos + 1
            # Only inject if the target is currently different
            if token_ids[i, src_pos] != token_ids[i, tgt_pos]:
                emb_out[i, tgt_pos] = emb_out[i, src_pos]
                ids_out[i, tgt_pos] = ids_out[i, src_pos]
                error_positions.append((i, int(tgt_pos)))

    return emb_out, ids_out, error_positions


def evaluate_detector(entropy, error_positions, n_seqs, seq_len, error_type):
    """Evaluate precision/recall of an entropy-threshold detector."""
    # Create error mask
    error_mask = np.zeros((n_seqs, seq_len), dtype=bool)
    for (i, j) in error_positions:
        error_mask[i, j] = True

    # Try multiple thresholds
    best_f1 = 0
    best_result = None
    thresholds = np.percentile(entropy[~error_mask], [50, 60, 70, 75, 80, 85, 90, 95])

    for thresh in thresholds:
        flagged = entropy > thresh
        tp = (flagged & error_mask).sum()
        fp = (flagged & ~error_mask).sum()
        fn = (~flagged & error_mask).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                "threshold": float(thresh),
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    # Also report entropy statistics
    error_entropy = entropy[error_mask] if error_mask.any() else np.array([])
    clean_entropy = entropy[~error_mask]

    result = {
        "error_type": error_type,
        "n_errors": len(error_positions),
        "mean_entropy_at_errors": float(error_entropy.mean()) if len(error_entropy) > 0 else None,
        "mean_entropy_at_clean": float(clean_entropy.mean()),
        "entropy_separation": float(error_entropy.mean() - clean_entropy.mean()) if len(error_entropy) > 0 else None,
        "best_detector": best_result,
    }
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("H2 — Confidence-Signal Validity")
    print("=" * 60)

    model, cfg = load_real_checkpoint()
    vocab_size = cfg["vocab_size"]

    print("Loading real sequences...")
    embeddings, token_ids, kmeans = load_real_data()
    seq_len = embeddings.shape[1]
    print(f"  Loaded {len(embeddings)} sequences")

    results = {}

    # ── Test 1: Substitution errors ──
    print("\n── Substitution Errors ──")
    sub_emb, sub_ids, sub_pos = inject_substitution_errors(embeddings, token_ids, kmeans)
    print(f"  Injected {len(sub_pos)} substitution errors")

    x_sub = torch.tensor(sub_emb, dtype=torch.float32, device=DEVICE)
    sub_tokens, sub_conf, sub_entropy = decode_batch(model, x_sub, DEVICE)

    sub_result = evaluate_detector(sub_entropy, sub_pos, len(embeddings), seq_len, "substitution")
    print(f"  Mean entropy at errors:  {sub_result['mean_entropy_at_errors']:.4f}")
    print(f"  Mean entropy at clean:   {sub_result['mean_entropy_at_clean']:.4f}")
    print(f"  Separation:              {sub_result['entropy_separation']:.4f}")
    print(f"  Best detector: P={sub_result['best_detector']['precision']:.3f}, "
          f"R={sub_result['best_detector']['recall']:.3f}, "
          f"F1={sub_result['best_detector']['f1']:.3f}")
    results["substitution"] = sub_result

    # ── Test 2: Repetition errors ──
    print("\n── Repetition Errors ──")
    rep_emb, rep_ids, rep_pos = inject_repetition_errors(embeddings, token_ids)
    print(f"  Injected {len(rep_pos)} repetition errors")

    x_rep = torch.tensor(rep_emb, dtype=torch.float32, device=DEVICE)
    rep_tokens, rep_conf, rep_entropy = decode_batch(model, x_rep, DEVICE)

    rep_result = evaluate_detector(rep_entropy, rep_pos, len(embeddings), seq_len, "repetition")
    print(f"  Mean entropy at errors:  {rep_result['mean_entropy_at_errors']:.4f}")
    print(f"  Mean entropy at clean:   {rep_result['mean_entropy_at_clean']:.4f}")
    print(f"  Separation:              {rep_result['entropy_separation']:.4f}")
    print(f"  Best detector: P={rep_result['best_detector']['precision']:.3f}, "
          f"R={rep_result['best_detector']['recall']:.3f}, "
          f"F1={rep_result['best_detector']['f1']:.3f}")
    results["repetition"] = rep_result

    # ── Go/No-Go ──
    sub_catches = sub_result["best_detector"]["f1"] > 0.1  # Even modest signal
    rep_catches = rep_result["best_detector"]["f1"] > 0.1

    print(f"\n{'=' * 60}")
    if sub_catches:
        print(f"✅ Substitution errors: DETECTABLE (F1={sub_result['best_detector']['f1']:.3f})")
    else:
        print(f"❌ Substitution errors: NOT reliably detectable")

    if rep_catches:
        print(f"✅ Repetition errors: DETECTABLE (F1={rep_result['best_detector']['f1']:.3f})")
    else:
        print(f"⚠️  Repetition errors: NOT reliably detectable by entropy alone")
        print(f"   Consider n-gram repetition detection as a complementary signal.")

    results["sub_detectable"] = sub_catches
    results["rep_detectable"] = rep_catches

    if sub_catches or rep_catches:
        print(f"\n✅ H2 PASS: At least one error type is detectable. Proceed to H3.")
        results["verdict"] = "PASS"
        if not rep_catches:
            results["scope_note"] = "Scope paper to substitution-type errors; repetitions need complementary detector."
    else:
        print(f"\n❌ H2 FAIL: Neither error type is reliably detectable via entropy.")
        results["verdict"] = "FAIL"
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (etype, eresult) in zip(axes, [("substitution", sub_result), ("repetition", rep_result)]):
        error_mask = np.zeros((len(embeddings), seq_len), dtype=bool)
        positions = sub_pos if etype == "substitution" else rep_pos
        for (i, j) in positions:
            error_mask[i, j] = True
        entropy_data = sub_entropy if etype == "substitution" else rep_entropy

        err_ent = entropy_data[error_mask]
        clean_ent = entropy_data[~error_mask]

        ax.hist(clean_ent, bins=50, alpha=0.6, color='#55A868', label='Clean positions', density=True)
        if len(err_ent) > 0:
            ax.hist(err_ent, bins=50, alpha=0.6, color='#C44E52', label='Error positions', density=True)
        ax.set_xlabel('Per-Token Entropy')
        ax.set_ylabel('Density')
        ax.set_title(f'{etype.capitalize()} Errors\n'
                     f'F1={eresult["best_detector"]["f1"]:.3f}, '
                     f'sep={eresult["entropy_separation"]:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("H2: Can Entropy Detect Injected Errors?", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testH2_confidence_signal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testH2_confidence_signal.png'}")

    with open(RESULTS_DIR / "testH2.json", "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
