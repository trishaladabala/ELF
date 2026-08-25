#!/usr/bin/env python3
"""Test 1 — Data Preparation and Embedding Extraction.

Samples ~5,000-10,000 sentences from wikitext, encodes with frozen T5-small
(512-d), applies PCA-initialized linear projection to 128-d (emulating ELF's
bottleneck), and saves x_512.npy and x_128.npy with metadata.

Sanity checks: embedding norms and pairwise distance distributions should look
non-degenerate (not collapsed to a point, not exploding).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──
NUM_SENTENCES = 6000        # Target number of sentences
MAX_SEQ_LEN = 64            # Tokens per sentence (T5 tokenizer)
BATCH_SIZE = 64             # Encoding batch size
BOTTLENECK_DIM = 128        # ELF bottleneck dimension (projection target)
ENCODER_DIM = 512           # T5-small hidden dim
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_sentences(num_sentences: int) -> list[str]:
    """Load clean sentences from wikitext-2 (Hugging Face datasets)."""
    from datasets import load_dataset

    print(f"Loading wikitext-2-raw-v1 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences = []
    for row in ds:
        text = row["text"].strip()
        # Skip headers (start with ' = '), empty lines, or very short lines
        if not text or text.startswith(" =") or len(text) < 30:
            continue
        sentences.append(text)
        if len(sentences) >= num_sentences:
            break
    print(f"  Collected {len(sentences)} sentences.")
    return sentences


def encode_sentences(
    sentences: list[str],
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode sentences with frozen T5-small and return 512-d embeddings.

    Returns:
        embeddings_512: (N_tokens, 512) — all non-padding token embeddings
        seq_ids:        (N_tokens,) — which sentence each token came from
        pos_ids:        (N_tokens,) — position within that sentence
    """
    from transformers import T5Tokenizer, T5EncoderModel

    print(f"Loading T5-small tokenizer + encoder on {device} …")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
    model = T5EncoderModel.from_pretrained("t5-small").to(device).eval()

    all_embs, all_seq_ids, all_pos_ids = [], [], []
    n_batches = (len(sentences) + batch_size - 1) // batch_size

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            toks = tokenizer(
                batch_texts,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = toks["input_ids"].to(device)
            attention_mask = toks["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state.cpu().numpy()  # (B, S, 512)
            mask_np = attention_mask.cpu().numpy()         # (B, S)

            for j in range(hidden.shape[0]):
                valid_len = int(mask_np[j].sum())
                embs = hidden[j, :valid_len]               # (valid_len, 512)
                all_embs.append(embs)
                all_seq_ids.append(np.full(valid_len, i + j, dtype=np.int32))
                all_pos_ids.append(np.arange(valid_len, dtype=np.int32))

            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == n_batches:
                elapsed = time.time() - t0
                print(f"  Encoded batch {batch_idx}/{n_batches}  "
                      f"({elapsed:.1f}s elapsed)")

    embeddings_512 = np.concatenate(all_embs, axis=0)
    seq_ids = np.concatenate(all_seq_ids, axis=0)
    pos_ids = np.concatenate(all_pos_ids, axis=0)
    print(f"  Total tokens extracted: {embeddings_512.shape[0]}")
    return embeddings_512, seq_ids, pos_ids


def pca_project(x_512: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA-initialized linear projection to target_dim.

    Returns projected embeddings and the projection matrix (for reproducibility).
    """
    print(f"PCA-projecting {x_512.shape} → ({x_512.shape[0]}, {target_dim}) …")
    mean = x_512.mean(axis=0, keepdims=True)
    x_centered = x_512 - mean
    # Compute top-k PCA components via SVD on a subsample for speed
    subsample = min(20000, x_centered.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(x_centered.shape[0], subsample, replace=False)
    U, S, Vt = np.linalg.svd(x_centered[idx], full_matrices=False)
    proj_matrix = Vt[:target_dim].T  # (512, target_dim)
    x_projected = x_centered @ proj_matrix
    print(f"  Variance explained by top {target_dim} PCs: "
          f"{(S[:target_dim]**2).sum() / (S**2).sum():.4f}")
    return x_projected.astype(np.float32), proj_matrix.astype(np.float32)


def sanity_checks(x_512: np.ndarray, x_128: np.ndarray) -> dict:
    """Run sanity checks on embeddings and return metrics dict."""
    rng = np.random.default_rng(0)

    # Norms
    norms_512 = np.linalg.norm(x_512, axis=1)
    norms_128 = np.linalg.norm(x_128, axis=1)

    # Pairwise distances on a random subsample
    n_sample = min(2000, x_128.shape[0])
    idx = rng.choice(x_128.shape[0], n_sample, replace=False)
    sub_128 = x_128[idx]
    # Compute pairwise distances efficiently
    from scipy.spatial.distance import pdist
    pw_dists = pdist(sub_128)

    metrics = {
        "total_tokens": int(x_512.shape[0]),
        "x_512_shape": list(x_512.shape),
        "x_128_shape": list(x_128.shape),
        "norm_512_mean": float(norms_512.mean()),
        "norm_512_std": float(norms_512.std()),
        "norm_512_min": float(norms_512.min()),
        "norm_512_max": float(norms_512.max()),
        "norm_128_mean": float(norms_128.mean()),
        "norm_128_std": float(norms_128.std()),
        "norm_128_min": float(norms_128.min()),
        "norm_128_max": float(norms_128.max()),
        "pairwise_dist_128_mean": float(pw_dists.mean()),
        "pairwise_dist_128_std": float(pw_dists.std()),
        "pairwise_dist_128_min": float(pw_dists.min()),
        "pairwise_dist_128_max": float(pw_dists.max()),
    }

    # Checks: not collapsed, not exploding
    collapsed = norms_128.std() < 1e-6
    exploding = norms_128.max() > 1e6
    metrics["collapsed"] = bool(collapsed)
    metrics["exploding"] = bool(exploding)
    metrics["non_degenerate"] = not collapsed and not exploding

    return metrics, pw_dists, norms_128


def make_plots(norms_128: np.ndarray, pw_dists: np.ndarray):
    """Save diagnostic plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(norms_128, bins=80, edgecolor="black", alpha=0.7, color="#4C72B0")
    axes[0].set_title("128-d Embedding Norms", fontsize=12)
    axes[0].set_xlabel("L2 norm")
    axes[0].set_ylabel("Count")

    axes[1].hist(pw_dists, bins=100, edgecolor="black", alpha=0.7, color="#DD8452")
    axes[1].set_title("Pairwise Distances (128-d, subsample)", fontsize=12)
    axes[1].set_xlabel("Euclidean distance")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_01_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved plot → {RESULTS_DIR / 'test_01_distributions.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST 1 — Data Preparation and Embedding Extraction")
    print("=" * 60)

    # Step 1: Load sentences
    sentences = load_sentences(NUM_SENTENCES)

    # Step 2: Encode with T5-small
    x_512, seq_ids, pos_ids = encode_sentences(
        sentences, MAX_SEQ_LEN, BATCH_SIZE, DEVICE
    )

    # Step 3: PCA project to 128-d
    x_128, proj_matrix = pca_project(x_512, BOTTLENECK_DIM)

    # Step 4: Save embeddings + metadata
    np.save(RESULTS_DIR / "x_512.npy", x_512)
    np.save(RESULTS_DIR / "x_128.npy", x_128)
    np.save(RESULTS_DIR / "seq_ids.npy", seq_ids)
    np.save(RESULTS_DIR / "pos_ids.npy", pos_ids)
    np.save(RESULTS_DIR / "proj_matrix.npy", proj_matrix)
    print(f"Saved embeddings to {RESULTS_DIR}/")

    # Step 5: Sanity checks
    metrics, pw_dists, norms_128 = sanity_checks(x_512, x_128)

    # Step 6: Plots
    make_plots(norms_128, pw_dists)

    # Step 7: Save results
    with open(RESULTS_DIR / "test_01.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Results ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if metrics["non_degenerate"]:
        print("\n✅ PASS: Embeddings are non-degenerate. Ready for Tests 2-4.")
    else:
        print("\n❌ FAIL: Embeddings look degenerate!")
        if metrics["collapsed"]:
            print("  → Norms have near-zero variance (collapsed).")
        if metrics["exploding"]:
            print("  → Norms are extremely large (exploding).")
    print("=" * 60)

    return 0 if metrics["non_degenerate"] else 1


if __name__ == "__main__":
    sys.exit(main())
