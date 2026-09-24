#!/usr/bin/env python3
"""Ordered Assembly: a synthetic sequence-generation task with known
per-token insertion structure.

The key idea: sequences are assembled in W discrete waves.  Wave 1 fills
a regular subset of positions, wave 2 fills the next subset conditioned on
wave 1, and so on.  Because the wave assignment is deterministic and known,
we can:
  1. Compute ground-truth insertion times for every token.
  2. Measure whether the model's generation respects the correct ordering.
  3. Vary W to control the complexity of the timing signal.

This lets us test the theory prediction: optimal compression K ≈ W.

Data format
-----------
Each sample is a dict:
    {
        "token_ids":      LongTensor (L,)   — the token sequence
        "wave_ids":       LongTensor (L,)   — wave index 0..W-1 for each position
        "insertion_times": FloatTensor (L,) — t_ins_i = wave_i / (W-1)
    }

The embedding representation fed to the flow model is a learned or random
lookup of the token_ids (same interface as EmbeddingDataset).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch.utils.data import Dataset


# ── helpers ──────────────────────────────────────────────────────────
def _interleave_positions(seq_len: int, W: int) -> torch.LongTensor:
    """Assign each position to a wave via round-robin.

    Position i belongs to wave (i % W).
    Returns: (seq_len,) LongTensor of wave indices in [0, W-1].
    """
    return torch.arange(seq_len) % W


def _block_positions(seq_len: int, W: int) -> torch.LongTensor:
    """Assign each position to a wave via contiguous blocks.

    First seq_len//W positions → wave 0, next → wave 1, etc.
    Returns: (seq_len,) LongTensor of wave indices in [0, W-1].
    """
    block_size = seq_len // W
    waves = torch.zeros(seq_len, dtype=torch.long)
    for w in range(W):
        start = w * block_size
        end = start + block_size if w < W - 1 else seq_len
        waves[start:end] = w
    return waves


WAVE_PATTERNS = {
    "interleave": _interleave_positions,
    "block": _block_positions,
}


# ── data generator ──────────────────────────────────────────────────
def generate_ordered_assembly_data(
    num_samples: int,
    seq_len: int = 64,
    vocab_size: int = 256,
    W: int = 4,
    wave_pattern: str = "interleave",
    seed: int = 42,
) -> dict:
    """Generate synthetic ordered-assembly sequences.

    Generation process for each sample:
      1. Assign every position to a wave (0..W-1) via `wave_pattern`.
      2. Wave 0 tokens are sampled i.i.d. from Uniform(0, vocab_size).
      3. Wave w>0 tokens are deterministic functions of their wave-(w-1)
         neighbors:  token[i] = (left_neighbor + right_neighbor) % vocab_size.
         This creates a genuine dependency on earlier waves.

    Args:
        num_samples: number of sequences to generate.
        seq_len: length of each sequence.
        vocab_size: size of the token vocabulary.
        W: number of assembly waves (the complexity parameter).
        wave_pattern: "interleave" or "block".
        seed: random seed.

    Returns:
        dict with keys:
            "token_ids":       (num_samples, seq_len) LongTensor
            "wave_ids":        (num_samples, seq_len) LongTensor
            "insertion_times": (num_samples, seq_len) FloatTensor  — in [0, 1]
    """
    assert W >= 2, "Need at least 2 waves"
    assert wave_pattern in WAVE_PATTERNS, f"Unknown pattern: {wave_pattern}"

    rng = torch.Generator().manual_seed(seed)
    wave_fn = WAVE_PATTERNS[wave_pattern]

    wave_ids = wave_fn(seq_len, W)  # (seq_len,) — same for all samples
    wave_ids_batch = wave_ids.unsqueeze(0).expand(num_samples, -1)  # (N, L)

    # Insertion times: t_ins_i = wave_i / (W - 1), so wave 0 → 0.0, wave W-1 → 1.0
    insertion_times = wave_ids_batch.float() / max(W - 1, 1)

    # Generate tokens wave by wave
    token_ids = torch.zeros(num_samples, seq_len, dtype=torch.long)

    for w in range(W):
        mask_w = (wave_ids == w)  # (seq_len,) bool
        positions_w = mask_w.nonzero(as_tuple=True)[0]

        if w == 0:
            # Wave 0: i.i.d. uniform
            token_ids[:, positions_w] = torch.randint(
                0, vocab_size, (num_samples, len(positions_w)), generator=rng
            )
        else:
            # Wave w>0: each token = f(left_neighbor, right_neighbor)
            # where neighbors are from earlier waves
            for pos in positions_w:
                # Find nearest filled (earlier-wave) neighbors
                left_val = torch.zeros(num_samples, dtype=torch.long)
                right_val = torch.zeros(num_samples, dtype=torch.long)

                # Scan left for nearest earlier-wave token
                for j in range(pos.item() - 1, -1, -1):
                    if wave_ids[j] < w:
                        left_val = token_ids[:, j]
                        break

                # Scan right for nearest earlier-wave token
                for j in range(pos.item() + 1, seq_len):
                    if wave_ids[j] < w:
                        right_val = token_ids[:, j]
                        break

                token_ids[:, pos] = (left_val + right_val + w) % vocab_size

    return {
        "token_ids": token_ids,
        "wave_ids": wave_ids_batch,
        "insertion_times": insertion_times,
    }


# ── Dataset class ────────────────────────────────────────────────────
class OrderedAssemblyDataset(Dataset):
    """PyTorch Dataset wrapping ordered-assembly synthetic data.

    Stores token_ids as embeddings (one-hot or learned lookup) so the
    interface matches EmbeddingDataset from train_efm.py.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        seq_len: int = 64,
        vocab_size: int = 256,
        embed_dim: int = 512,
        W: int = 4,
        wave_pattern: str = "interleave",
        seed: int = 42,
        cache_path: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.W = W

        if cache_path and torch.serialization.os.path.exists(cache_path):
            print(f"Loading cached synthetic data from {cache_path}")
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.token_ids = data["token_ids"]
            self.wave_ids = data["wave_ids"]
            self.insertion_times = data["insertion_times"]
            self.embeddings = data["embeddings"]
        else:
            print(f"Generating ordered assembly data: "
                  f"N={num_samples}, L={seq_len}, V={vocab_size}, W={W}")
            raw = generate_ordered_assembly_data(
                num_samples=num_samples,
                seq_len=seq_len,
                vocab_size=vocab_size,
                W=W,
                wave_pattern=wave_pattern,
                seed=seed,
            )
            self.token_ids = raw["token_ids"]
            self.wave_ids = raw["wave_ids"]
            self.insertion_times = raw["insertion_times"]

            # Create random embedding table and look up
            rng = torch.Generator().manual_seed(seed + 1)
            embed_table = torch.randn(vocab_size, embed_dim, generator=rng) * 0.02
            self.embeddings = embed_table[self.token_ids]  # (N, L, D)

            if cache_path:
                import os
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                torch.save({
                    "token_ids": self.token_ids,
                    "wave_ids": self.wave_ids,
                    "insertion_times": self.insertion_times,
                    "embeddings": self.embeddings,
                }, cache_path)
                print(f"Cached synthetic data to {cache_path}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],         # (L, D)
            "token_ids": self.token_ids[idx],            # (L,)
            "wave_ids": self.wave_ids[idx],              # (L,)
            "insertion_times": self.insertion_times[idx], # (L,)
        }


# ── quick sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    for W in [2, 4, 8]:
        data = generate_ordered_assembly_data(
            num_samples=4, seq_len=32, vocab_size=64, W=W, seed=0
        )
        tids = data["token_ids"]
        wids = data["wave_ids"]
        tins = data["insertion_times"]

        print(f"\n=== W={W} ===")
        print(f"  token_ids shape:       {tids.shape}")
        print(f"  wave_ids[0]:           {wids[0].tolist()}")
        print(f"  insertion_times[0]:    {tins[0].tolist()}")
        print(f"  token_ids[0]:          {tids[0].tolist()}")

        # Verify: wave-w tokens depend on wave-(w-1)
        for w in range(1, W):
            mask = (wids[0] == w)
            if mask.any():
                pos = mask.nonzero(as_tuple=True)[0][0].item()
                expected_t = w / (W - 1)
                actual_t = tins[0, pos].item()
                assert abs(actual_t - expected_t) < 1e-6, \
                    f"insertion time mismatch at wave {w}"
        print(f"  ✓ insertion times verified")

        # Verify: per-wave entropy
        for w in range(W):
            mask = (wids[0] == w)
            tokens_w = tids[0, mask]
            unique = tokens_w.unique().numel()
            print(f"  wave {w}: {mask.sum().item()} positions, "
                  f"{unique} unique tokens")

    print("\n✓ All sanity checks passed.")

