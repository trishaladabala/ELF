"""Insertion head — predicts per-gap token insertion counts.

This module takes hidden states from the flow transformer and predicts
how many new tokens should be inserted between each pair of adjacent tokens.

Architecture:
    Input:  hidden states at gap positions (B, S-1, hidden_size)
    MLP:    hidden_size → hidden_size/4 → 1
    Output: non-negative real-valued counts via Softplus (Poisson-style)

During inference, outputs are rounded to integers and clamped to
[0, max_insert_per_gap].

Loss: Poisson NLL against ground-truth insertion counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InsertionHead(nn.Module):
    """Predicts per-gap insertion counts from flow transformer hidden states.

    Args:
        hidden_size:          Model hidden dimension (input size).
        hidden_dim_ratio:     MLP hidden dim = hidden_size × ratio.
        max_insert_per_gap:   Maximum tokens to insert per gap at inference.
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_dim_ratio: float = 0.25,
        max_insert_per_gap: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_insert_per_gap = max_insert_per_gap

        inner_dim = max(1, int(hidden_size * hidden_dim_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, 1),
        )
        # Small init so initial predictions are near zero.
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, -1.0)  # Softplus(-1) ≈ 0.31

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict per-gap insertion rates (continuous, non-negative).

        Args:
            hidden_states: (B, S, hidden_size) — hidden states from the transformer.
                           Gaps are between positions i and i+1, so we use
                           S-1 gap representations.

        Returns:
            rates: (B, S-1) — predicted insertion rates (Softplus output).
        """
        # Compute gap representations: average of adjacent hidden states.
        gap_repr = (hidden_states[:, :-1] + hidden_states[:, 1:]) / 2  # (B, S-1, H)
        logits = self.mlp(gap_repr).squeeze(-1)  # (B, S-1)
        rates = F.softplus(logits)               # Non-negative
        return rates

    def predict_counts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict integer insertion counts for inference.

        Args:
            hidden_states: (B, S, hidden_size)

        Returns:
            counts: (B, S-1) — integer insertion counts, clamped to [0, max_insert].
        """
        rates = self.forward(hidden_states)
        counts = torch.round(rates).long().clamp(0, self.max_insert_per_gap)
        return counts

    @staticmethod
    def poisson_nll_loss(
        predicted_rates: torch.Tensor,
        target_counts: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Poisson NLL loss for insertion count prediction.

        Args:
            predicted_rates: (B, S-1) — predicted λ from the insertion head.
            target_counts:   (B, S-1) — ground-truth integer counts.
            mask:            (B, S-1) — optional mask (1=valid, 0=padding).

        Returns:
            Scalar loss value.
        """
        # Poisson NLL: λ - k·log(λ) + log(k!)
        # F.poisson_nll_loss computes: input - target * log(input + eps)
        # when log_input=False.
        loss = F.poisson_nll_loss(
            predicted_rates, target_counts.float(),
            log_input=False, full=True, reduction="none",
        )
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)
        return loss.mean()
