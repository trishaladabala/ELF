"""Local-time conditioning module for EFM-style models.

Supports three modes for representing per-token local time τ_i ∈ [0,1]:

    continuous  — Full sinusoidal embedding (same as ELF's TimestepEmbedder,
                  but applied per-token instead of per-sample).
    quantized   — τ_i → floor(τ_i × K) → nn.Embedding(K, hidden_size).
    lowrank     — τ_i → φ_K(τ_i) → projection to hidden_size.
                  Basis types: "fourier" or "learned" MLP.

IMPORTANT: The low-rank mode compresses the local-time SIGNAL τ_i,
           NOT the token position i.

All modes produce a tensor of shape (B, S, hidden_size) that is added to
the hidden states after the bottleneck projection.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTimeConditioner(nn.Module):
    """Maps per-token local times τ ∈ [0,1] to conditioning vectors.

    Args:
        hidden_size: Model hidden dimension (output size).
        mode:        "continuous", "quantized", or "lowrank".
        K:           Number of bins (quantized) or basis dimensions (lowrank).
        lowrank_basis: "fourier" or "learned" (only for lowrank mode).
        frequency_embedding_size: Size of sinusoidal embedding for continuous mode.
    """

    def __init__(
        self,
        hidden_size: int,
        mode: str = "continuous",
        K: int = 16,
        lowrank_basis: str = "fourier",
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        self.K = K
        self.lowrank_basis = lowrank_basis
        self.frequency_embedding_size = frequency_embedding_size

        if mode == "continuous":
            # Same sinusoidal + MLP architecture as ELF's TimestepEmbedder,
            # but applied per-token (τ_i) instead of per-sample (t).
            self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size)
            self.mlp_1 = nn.Linear(hidden_size, hidden_size)
            nn.init.normal_(self.mlp_0.weight, std=0.02)
            nn.init.normal_(self.mlp_1.weight, std=0.02)
            nn.init.zeros_(self.mlp_0.bias)
            nn.init.zeros_(self.mlp_1.bias)

        elif mode == "quantized":
            # Learned embedding table for K discrete bins.
            self.bin_embedding = nn.Embedding(K, hidden_size)
            nn.init.normal_(self.bin_embedding.weight, std=0.02)
            # Optional adapter MLP for fine-tuning.
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            nn.init.normal_(self.adapter[0].weight, std=0.02)
            nn.init.zeros_(self.adapter[2].weight)
            nn.init.zeros_(self.adapter[2].bias)

        elif mode == "lowrank":
            if lowrank_basis == "fourier":
                # Fourier basis: φ_K(τ) = [sin(ω₁τ), cos(ω₁τ), ..., sin(ω_K τ), cos(ω_K τ)]
                # Results in 2K-dimensional representation.
                # Frequencies are learnable (initialised log-linearly).
                self.log_frequencies = nn.Parameter(
                    torch.linspace(0, math.log(100.0), K)
                )
                self.projection = nn.Linear(2 * K, hidden_size)
                nn.init.normal_(self.projection.weight, std=0.02)
                nn.init.zeros_(self.projection.bias)

            elif lowrank_basis == "learned":
                # Learned MLP: τ_i (scalar) → K-dim → hidden_size.
                self.basis_mlp = nn.Sequential(
                    nn.Linear(1, K * 4),
                    nn.SiLU(),
                    nn.Linear(K * 4, K),
                    nn.SiLU(),
                )
                self.projection = nn.Linear(K, hidden_size)
                nn.init.normal_(self.projection.weight, std=0.02)
                nn.init.zeros_(self.projection.bias)
            else:
                raise ValueError(f"Unknown lowrank_basis: {lowrank_basis}")
                
        elif mode == "none":
            pass # No parameters needed
        else:
            raise ValueError(f"Unknown local_time mode: {mode}")

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Map per-token local times to conditioning vectors.

        Args:
            tau: (B, S) tensor of local times in [0, 1].

        Returns:
            (B, S, hidden_size) conditioning vectors.
        """
        if self.mode == "none":
            return torch.zeros(*tau.shape, self.hidden_size, device=tau.device, dtype=torch.float32)
        elif self.mode == "continuous":
            return self._forward_continuous(tau)
        elif self.mode == "quantized":
            return self._forward_quantized(tau)
        elif self.mode == "lowrank":
            return self._forward_lowrank(tau)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _forward_continuous(self, tau: torch.Tensor) -> torch.Tensor:
        """Full sinusoidal embedding of τ_i (per-token)."""
        B, S = tau.shape
        # Flatten to (B*S,) for sinusoidal embedding.
        tau_flat = tau.reshape(-1)
        emb = self._sinusoidal_embedding(tau_flat, self.frequency_embedding_size)
        # MLP: (B*S, freq_dim) → (B*S, hidden)
        h = self.mlp_1(F.silu(self.mlp_0(emb)))
        return h.reshape(B, S, self.hidden_size)

    def _forward_quantized(self, tau: torch.Tensor) -> torch.Tensor:
        """Quantize τ_i to discrete bins, look up embeddings."""
        # Clamp and quantize: τ_i → floor(τ_i × K), clamped to [0, K-1].
        bin_ids = (tau * self.K).long().clamp(0, self.K - 1)
        emb = self.bin_embedding(bin_ids)         # (B, S, hidden)
        emb = emb + self.adapter(emb)             # Residual adapter
        return emb

    def _forward_lowrank(self, tau: torch.Tensor) -> torch.Tensor:
        """Low-rank functional approximation of τ_i."""
        B, S = tau.shape
        if self.lowrank_basis == "fourier":
            # φ_K(τ) = [sin(ω₁τ), cos(ω₁τ), ..., sin(ω_K τ), cos(ω_K τ)]
            freqs = torch.exp(self.log_frequencies)          # (K,)
            # (B, S, 1) * (K,) → (B, S, K)
            angles = tau.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
            basis = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, S, 2K)
            return self.projection(basis)                    # (B, S, hidden)

        elif self.lowrank_basis == "learned":
            # τ_i (scalar) → MLP → K-dim → projection → hidden.
            tau_input = tau.unsqueeze(-1)                    # (B, S, 1)
            basis = self.basis_mlp(tau_input)                # (B, S, K)
            return self.projection(basis)                    # (B, S, hidden)

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Sinusoidal timestep embedding: (N,) → (N, dim).

        Identical to ELF's TimestepEmbedder.timestep_embedding.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def num_conditioning_dims(self) -> int:
        """Return the effective number of conditioning dimensions used.

        Useful for analysis: how many 'dimensions of information' does each
        mode use to represent τ_i?
        """
        if self.mode == "continuous":
            return self.frequency_embedding_size
        elif self.mode == "quantized":
            return self.K
        elif self.mode == "lowrank":
            if self.lowrank_basis == "fourier":
                return 2 * self.K
            else:
                return self.K
        return 0
