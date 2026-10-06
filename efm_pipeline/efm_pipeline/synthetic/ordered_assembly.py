#!/usr/bin/env python3
"""Wave-Routed Copy: a synthetic task that mathematically forces a U-shaped capacity curve.

The Sequence is divided into W blocks of size M.
- The first W-1 blocks are "Key Banks" (generated in Wave 0, perfectly known).
- The final block is "Values".
- Each Value token is randomly assigned to a wave w in {1, ..., W-1}.
- A Value token assigned to wave w MUST copy its token from Key Bank w.

If the model compresses the local time too much (K < W-1), it will confuse 
which wave it belongs to, and thus copy from the wrong Key Bank. This strictly 
forces C* ~ W.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch.utils.data import Dataset

def generate_ordered_assembly_data(
    num_samples: int, seq_len: int = 64, vocab_size: int = 256,
    W: int = 4, seed: int = 42, sigma: float = 0.0, **kwargs
) -> dict:
    assert seq_len % W == 0, f"seq_len {seq_len} must be divisible by W {W}"
    M = seq_len // W
    rng = torch.Generator().manual_seed(seed)
    
    token_ids = torch.zeros(num_samples, seq_len, dtype=torch.long)
    wave_ids = torch.zeros(num_samples, seq_len, dtype=torch.long)
    
    num_keys = (W - 1) * M
    
    # Wave 0: Random Keys
    token_ids[:, :num_keys] = torch.randint(0, vocab_size, (num_samples, num_keys), generator=rng)
    wave_ids[:, :num_keys] = 0
    
    # Waves 1 to W-1: Values
    val_waves = torch.randint(1, W, (num_samples, M), generator=rng)
    wave_ids[:, num_keys:] = val_waves
    
    # Apply copying rule
    for b in range(num_samples):
        for i in range(M):
            w = val_waves[b, i].item()
            source_pos = (w - 1) * M + i
            token_ids[b, num_keys + i] = token_ids[b, source_pos]
            
    if sigma > 0.0:
        noise = torch.randn(wave_ids.shape, generator=rng) * sigma
        insertion_times = (wave_ids.float() + noise) / max(W - 1, 1)
        insertion_times = insertion_times.clamp(0.0, 1.0)
    else:
        insertion_times = wave_ids.float() / max(W - 1, 1)
    
    return {
        "token_ids": token_ids,
        "wave_ids": wave_ids,
        "insertion_times": insertion_times,
    }

class OrderedAssemblyDataset(Dataset):
    def __init__(self, num_samples=5000, seq_len=64, vocab_size=256, embed_dim=512, W=4, seed=42, sigma=0.0, **kwargs):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.W = W
        raw = generate_ordered_assembly_data(num_samples, seq_len, vocab_size, W, seed, sigma)
        self.token_ids = raw["token_ids"]
        self.wave_ids = raw["wave_ids"]
        self.insertion_times = raw["insertion_times"]
        
        rng = torch.Generator().manual_seed(seed + 1)
        # Flow matching requires x0 to have variance ~1.0 so signal isn't drowned by eps
        embed_table = torch.randn(vocab_size, embed_dim, generator=rng)
        self.embeddings = embed_table[self.token_ids]

    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "token_ids": self.token_ids[idx],
            "wave_ids": self.wave_ids[idx],
            "insertion_times": self.insertion_times[idx],
        }
