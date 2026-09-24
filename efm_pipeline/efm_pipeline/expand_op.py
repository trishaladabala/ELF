"""Expand operator — inserts new tokens into the embedding sequence.

Given the current embedding sequence and insertion counts from the
InsertionHead, the expand operator creates an expanded sequence with
noise-filled positions for the newly inserted tokens.

Operations:
    1. Takes current sequence (B, S, D) + counts (B, S-1) from insertion head.
    2. Produces expanded sequence (B, S', D) where S' = S + Σcounts.
    3. New positions filled with: ε ~ N(0, I) (default) or N(μ_θ, I) (optional).
    4. Local-time assignment: existing τ kept, new tokens get τ_new = 0.

Constraint: The expand operator preserves existing tokens' embeddings
            exactly — it only adds, never modifies existing positions.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ExpandOperator(nn.Module):
    """Insert noise-filled tokens into the embedding sequence.

    Args:
        embed_dim: Embedding dimension (D, matches T5 encoder output).
        max_total_length: Maximum sequence length after expansion.
        noise_scale: Std dev of the Gaussian noise for new tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        max_total_length: int = 128,
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_total_length = max_total_length
        self.noise_scale = noise_scale

    def forward(
        self,
        embeddings: torch.Tensor,
        insert_counts: torch.Tensor,
        local_times: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand the embedding sequence by inserting noise tokens.

        Args:
            embeddings:    (B, S, D) — current embedding sequence.
            insert_counts: (B, S-1) — integer counts of tokens to insert per gap.
            local_times:   (B, S) — local times for existing tokens.
            generator:     Optional torch.Generator for reproducibility.

        Returns:
            expanded_embeddings: (B, S', D) — expanded sequence (padded to max_total_length).
            expanded_times:      (B, S') — local times (new tokens get τ=0).
            expansion_mask:      (B, S') — 1 for valid tokens, 0 for padding.
        """
        B, S, D = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        # Compute new sequence lengths per sample.
        total_inserts = insert_counts.sum(dim=-1)                    # (B,)
        new_lengths = S + total_inserts                               # (B,)
        S_prime = min(int(new_lengths.max().item()), self.max_total_length)

        # Allocate output tensors.
        expanded_emb = torch.zeros(B, S_prime, D, dtype=dtype, device=device)
        expanded_tau = torch.zeros(B, S_prime, dtype=dtype, device=device)
        expansion_mask = torch.zeros(B, S_prime, dtype=dtype, device=device)

        # Process each sample (variable-length expansion).
        for b in range(B):
            write_pos = 0
            for i in range(S):
                if write_pos >= S_prime:
                    break
                # Copy existing token.
                expanded_emb[b, write_pos] = embeddings[b, i]
                expanded_tau[b, write_pos] = local_times[b, i]
                expansion_mask[b, write_pos] = 1.0
                write_pos += 1

                # Insert new tokens after position i (before i+1).
                if i < S - 1:
                    n_insert = int(insert_counts[b, i].item())
                    for _ in range(n_insert):
                        if write_pos >= S_prime:
                            break
                        # New token: Gaussian noise.
                        noise = torch.randn(
                            D, dtype=dtype, device=device, generator=generator,
                        ) * self.noise_scale
                        expanded_emb[b, write_pos] = noise
                        expanded_tau[b, write_pos] = 0.0   # Fresh token, needs full denoising.
                        expansion_mask[b, write_pos] = 1.0
                        write_pos += 1

        return expanded_emb, expanded_tau, expansion_mask


class ExpandOperatorBatched(nn.Module):
    """Vectorised expand operator using scatter (no Python loops over B).

    More efficient for larger batch sizes. Produces the same result as
    ExpandOperator but avoids per-sample Python loops.

    Note: This requires all samples to expand to the same total length
    (padded to max). For variable-length output, use ExpandOperator.
    """

    def __init__(
        self,
        embed_dim: int,
        max_total_length: int = 128,
        noise_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_total_length = max_total_length
        self.noise_scale = noise_scale

    def forward(
        self,
        embeddings: torch.Tensor,
        insert_counts: torch.Tensor,
        local_times: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched expand using cumulative sums to compute target positions.

        Same interface as ExpandOperator.forward().
        """
        B, S, D = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        # Compute write positions for each existing token.
        # For token i, write_pos[i] = i + sum(insert_counts[:i]).
        # Pad insert_counts to S (append 0 for last token which has no following gap).
        counts_padded = torch.cat([
            insert_counts,
            torch.zeros(B, 1, dtype=insert_counts.dtype, device=device),
        ], dim=1)  # (B, S)

        # Cumulative inserts before each position.
        cum_inserts = torch.cumsum(counts_padded, dim=1) - counts_padded  # (B, S)
        write_positions = torch.arange(S, device=device).unsqueeze(0) + cum_inserts  # (B, S)

        S_prime = min(int((S + insert_counts.sum(dim=-1).max()).item()), self.max_total_length)

        # Scatter existing tokens into output.
        expanded_emb = torch.randn(
            B, S_prime, D, dtype=dtype, device=device, generator=generator,
        ) * self.noise_scale
        expanded_tau = torch.zeros(B, S_prime, dtype=dtype, device=device)
        expansion_mask = torch.zeros(B, S_prime, dtype=dtype, device=device)

        # Clamp write positions.
        wp = write_positions.long().clamp(0, S_prime - 1)  # (B, S)

        # Scatter existing embeddings.
        wp_emb = wp.unsqueeze(-1).expand(-1, -1, D)        # (B, S, D)
        expanded_emb.scatter_(1, wp_emb, embeddings)

        # Scatter existing local times.
        expanded_tau.scatter_(1, wp, local_times)

        # Mark all valid positions.
        # Valid = all positions from 0 to (S + total_inserts - 1) per sample.
        total_len = S + insert_counts.sum(dim=-1)            # (B,)
        pos_idx = torch.arange(S_prime, device=device).unsqueeze(0)  # (1, S')
        expansion_mask = (pos_idx < total_len.unsqueeze(1)).float()

        return expanded_emb, expanded_tau, expansion_mask
