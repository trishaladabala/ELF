#!/usr/bin/env python3
"""Local sensitivity (L_i) estimators for the decode readout.

Two estimators behind a common interface:
- FDEstimator: finite-difference approximation (cheap, 2 decode passes per probe)
- JVPEstimator: forward-mode Jacobian-vector product (exact, more expensive)

Probe directions:
- 'isotropic': random unit Gaussian, normalized per position
- 'decoder_grad': gradient of margin w.r.t. x̂ (one backward through decode)
- 'pca': top PCA directions of x̂ across positions
- 'trajectory': empirical residual direction (Phase 2, placeholder now)
"""

import abc
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F


class ProbeDirectionGenerator:
    """Generate probe directions for sensitivity estimation."""

    @staticmethod
    def isotropic(x_hat: torch.Tensor, n_probes: int = 4) -> List[torch.Tensor]:
        """Random isotropic unit directions, normalized per position.

        Args:
            x_hat: (B, L, D) clean-state prediction
            n_probes: number of probe directions

        Returns:
            list of (B, L, D) unit-norm probe tensors
        """
        probes = []
        for _ in range(n_probes):
            v = torch.randn_like(x_hat)
            v = F.normalize(v, dim=-1)  # normalize per position
            probes.append(v)
        return probes

    @staticmethod
    def decoder_grad(
        x_hat: torch.Tensor,
        decode_fn,
        n_probes: int = 1,
    ) -> List[torch.Tensor]:
        """Gradient of top1-top2 margin w.r.t. x̂.

        Args:
            x_hat: (B, L, D), must be detached
            decode_fn: callable that maps (B, L, D) -> (B, L, V) logits
            n_probes: ignored (always 1 direction from gradient)

        Returns:
            list of (B, L, D) unit-norm gradient directions
        """
        x = x_hat.detach().clone().requires_grad_(True)
        logits = decode_fn(x)  # (B, L, V)
        top2 = logits.topk(2, dim=-1)
        margin = top2.values[:, :, 0] - top2.values[:, :, 1]  # (B, L)
        # Backprop the sum of margins
        margin.sum().backward()
        grad = x.grad  # (B, L, D)
        # Normalize per position
        v = F.normalize(grad, dim=-1)
        return [v.detach()]

    @staticmethod
    def pca(x_hat: torch.Tensor, n_probes: int = 4) -> List[torch.Tensor]:
        """Top PCA directions of x̂ across positions.

        Computes PCA over the (B*L, D) matrix and returns the top
        n_probes principal components, broadcast to (B, L, D).

        Args:
            x_hat: (B, L, D)
            n_probes: number of top PC directions

        Returns:
            list of (B, L, D) tensors (each position gets the same direction)
        """
        B, L, D = x_hat.shape
        flat = x_hat.reshape(-1, D).float()  # (B*L, D)
        flat = flat - flat.mean(dim=0, keepdim=True)
        # SVD on centered data
        _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
        # Vh: (D, D), rows are principal components
        probes = []
        for i in range(min(n_probes, D)):
            v = Vh[i].unsqueeze(0).unsqueeze(0).expand(B, L, D)  # (B, L, D)
            v = v.to(x_hat.dtype).to(x_hat.device)
            probes.append(v)
        return probes


class SensitivityEstimator(abc.ABC):
    """Abstract base for local sensitivity estimators."""

    @abc.abstractmethod
    def estimate(
        self,
        x_hat: torch.Tensor,
        decode_fn,
        top1_ids: torch.Tensor,
        top2_ids: torch.Tensor,
        probe_directions: List[torch.Tensor],
    ) -> torch.Tensor:
        """Estimate L_i at each position.

        Args:
            x_hat: (B, L, D) clean-state prediction
            decode_fn: callable (B, L, D) -> (B, L, V) logits
            top1_ids: (B, L) argmax token IDs
            top2_ids: (B, L) second-best token IDs
            probe_directions: list of (B, L, D) unit-norm probe tensors

        Returns:
            L_i: (B, L) estimated local Lipschitz constant of the margin
        """
        ...


class FDEstimator(SensitivityEstimator):
    """Finite-difference sensitivity estimator.

    For each probe direction v:
        g_plus = decode(x̂ + h·v)
        g_minus = decode(x̂ - h·v)
        For each position, compute:
            L_i = max_j |(m_ij(x̂+hv) - m_ij(x̂-hv))| / (2h)
        where m_ij = g_i - g_j for competing tokens j in top-k.

    Average over probe directions.
    """

    def __init__(self, h: float = 1e-2):
        """
        Args:
            h: finite-difference step size
        """
        self.h = h

    @torch.no_grad()
    def estimate(
        self,
        x_hat: torch.Tensor,
        decode_fn,
        top1_ids: torch.Tensor,
        top2_ids: torch.Tensor,
        probe_directions: List[torch.Tensor],
    ) -> torch.Tensor:
        B, L, D = x_hat.shape
        h = self.h
        L_accum = torch.zeros(B, L, device=x_hat.device, dtype=torch.float32)
        n_probes = len(probe_directions)

        if n_probes == 0:
            return L_accum

        for v in probe_directions:
            # Forward and backward perturbations
            logits_plus = decode_fn(x_hat + h * v)   # (B, L, V)
            logits_minus = decode_fn(x_hat - h * v)  # (B, L, V)

            # Extract margin for top1 vs top2
            # m_plus = g_top1(x+hv) - g_top2(x+hv)
            g1_plus = logits_plus.gather(-1, top1_ids.unsqueeze(-1)).squeeze(-1)
            g2_plus = logits_plus.gather(-1, top2_ids.unsqueeze(-1)).squeeze(-1)
            m_plus = g1_plus - g2_plus

            g1_minus = logits_minus.gather(-1, top1_ids.unsqueeze(-1)).squeeze(-1)
            g2_minus = logits_minus.gather(-1, top2_ids.unsqueeze(-1)).squeeze(-1)
            m_minus = g1_minus - g2_minus

            # L_i = |m(x+hv) - m(x-hv)| / (2h)
            L_i = (m_plus - m_minus).abs().float() / (2 * h)
            L_accum += L_i

        return L_accum / n_probes


class JVPEstimator(SensitivityEstimator):
    """Forward-mode JVP sensitivity estimator.

    Uses torch.func.jvp to compute the exact directional derivative
    of the margin function along each probe direction.
    """

    def estimate(
        self,
        x_hat: torch.Tensor,
        decode_fn,
        top1_ids: torch.Tensor,
        top2_ids: torch.Tensor,
        probe_directions: List[torch.Tensor],
    ) -> torch.Tensor:
        B, L, D = x_hat.shape
        L_accum = torch.zeros(B, L, device=x_hat.device, dtype=torch.float32)
        n_probes = len(probe_directions)

        if n_probes == 0:
            return L_accum

        def margin_fn(x):
            logits = decode_fn(x)
            g1 = logits.gather(-1, top1_ids.unsqueeze(-1)).squeeze(-1)
            g2 = logits.gather(-1, top2_ids.unsqueeze(-1)).squeeze(-1)
            return g1 - g2  # (B, L)

        for v in probe_directions:
            try:
                _, jvp_out = torch.func.jvp(margin_fn, (x_hat,), (v,))
                L_i = jvp_out.abs().float()
            except Exception:
                # Fallback to FD if JVP fails (e.g., due to in-place ops)
                fd = FDEstimator(h=1e-2)
                L_i_fd = fd.estimate(x_hat, decode_fn, top1_ids, top2_ids, [v])
                L_i = L_i_fd
            L_accum += L_i

        return L_accum / n_probes


def get_estimator(method: str = "fd", **kwargs) -> SensitivityEstimator:
    """Factory for sensitivity estimators."""
    if method == "fd":
        return FDEstimator(**kwargs)
    elif method == "jvp":
        return JVPEstimator(**kwargs)
    else:
        raise ValueError(f"Unknown estimator: {method}. Use 'fd' or 'jvp'.")
