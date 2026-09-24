#!/usr/bin/env python3
"""Instrumented ODE/SDE sampler for decoder-basin analysis.

Performs identical updates to the existing pipeline (verified via equivalence
test) while recording per-step per-position diagnostics:
  - margin (top1 - top2), top-k logits and IDs (k=8)
  - decoder logit entropy
  - self-conditioning delta ‖x̂_s − x̂_{s−1}‖
  - L_i estimate (local sensitivity) and ρ_i = m_i / L_i
  - sequence-level: 10th-pctile margin, effective rank of x̂_s

All diagnostics are stored in CPU RAM as lists of tensors, NOT on GPU.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
from pathlib import Path

_ELF_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_ELF_SRC) not in sys.path:
    sys.path.insert(0, str(_ELF_SRC))

from utils.sampling_utils import (
    restore_cond, _ode_step, _sde_step,
    net_out_to_v_x, _forward_sample,
)

from elfbasin.sampling.sensitivity import (
    FDEstimator, ProbeDirectionGenerator, get_estimator,
)


@dataclass
class StepDiagnostics:
    """Per-step diagnostics for a batch of samples."""
    step_idx: int
    t: float
    # Per-position (B, L)
    margin: torch.Tensor            # top1 - top2
    top1_ids: torch.Tensor          # (B, L) int
    top2_ids: torch.Tensor          # (B, L) int
    top_k_logits: torch.Tensor      # (B, L, k) float16
    top_k_ids: torch.Tensor         # (B, L, k) int
    entropy: torch.Tensor           # (B, L) decoder logit entropy
    sc_delta: Optional[torch.Tensor] = None  # (B, L) ‖x̂_s − x̂_{s−1}‖
    L_i: Optional[torch.Tensor] = None       # (B, L) local sensitivity
    rho_i: Optional[torch.Tensor] = None     # (B, L) normalized margin m/L
    # Sequence-level (B,)
    margin_p10: Optional[torch.Tensor] = None  # 10th percentile margin
    effective_rank: Optional[torch.Tensor] = None  # entropy of SVD spectrum
    x_hat_val: Optional[torch.Tensor] = None
    z_val: Optional[torch.Tensor] = None


@dataclass
class TrajectoryDiagnostics:
    """Complete diagnostics for a full trajectory."""
    steps: List[StepDiagnostics] = field(default_factory=list)
    final_token_ids: Optional[torch.Tensor] = None  # (B, L) final decoded tokens
    # Post-hoc: per-step agreement with final decode
    step_agreement: Optional[List[torch.Tensor]] = None  # list of (B, L) bool


def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-position entropy from logits. (B, L, V) -> (B, L)"""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _compute_effective_rank(x_hat: torch.Tensor) -> torch.Tensor:
    """Compute effective rank of x̂ per sample via SVD spectrum entropy.

    Args:
        x_hat: (B, L, D)
    Returns:
        effective_rank: (B,)
    """
    B, L, D = x_hat.shape
    ranks = []
    for b in range(B):
        try:
            S = torch.linalg.svdvals(x_hat[b].float())  # (min(L,D),)
            # Normalize to probability distribution
            S_norm = S / (S.sum() + 1e-12)
            # Entropy of the distribution
            log_S = torch.log(S_norm + 1e-12)
            entropy = -(S_norm * log_S).sum()
            # Effective rank = exp(entropy)
            ranks.append(entropy.exp())
        except Exception:
            ranks.append(torch.tensor(float('nan')))
    return torch.stack(ranks)


@torch.no_grad()
def instrumented_generate(
    model: nn.Module,
    wrapper,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config,
    sampling_config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    # Instrumentation options
    compute_diagnostics: bool = True,
    compute_L_i: bool = False,
    store_states: bool = False,
    sensitivity_method: str = "fd",
    probe_type: str = "isotropic",
    n_probes: int = 4,
    top_k: int = 8,
    diagnostic_steps: Optional[List[int]] = None,
) -> tuple:
    """Generate samples with full per-step diagnostics.

    This function performs the IDENTICAL sampling loop as
    _generate_samples_single_batch, with additional diagnostic recording.

    Args:
        model: ELF model
        wrapper: ELFWrapper instance (for decode_fn)
        generator: torch random generator
        z: (B, L, D) initial noise
        t_steps: (num_steps+1,) time steps
        cond_seq: conditioning sequence or None
        cond_seq_mask: conditioning mask or None
        config: model config
        sampling_config: sampling config
        cfg_scale: input-conditioning CFG scale
        self_cond_cfg_scale: self-conditioning CFG scale
        compute_diagnostics: whether to record diagnostics
        compute_L_i: whether to estimate L_i (adds forward passes)
        sensitivity_method: 'fd' or 'jvp'
        probe_type: 'isotropic', 'decoder_grad', 'pca'
        n_probes: number of probe directions for L_i
        top_k: number of top logits to store
        diagnostic_steps: if set, only record diagnostics at these step indices.
                         None = record at every step.

    Returns:
        z_final: (B, L, D) final latent states
        trajectory: TrajectoryDiagnostics
    """
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape

    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model),
                               dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length),
                                    dtype=z.dtype, device=z.device)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
    x_pred_prev = None  # for SC delta

    n = t_steps.shape[0]
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)
    trajectory = TrajectoryDiagnostics()

    # Sensitivity estimator
    estimator = get_estimator(sensitivity_method) if compute_L_i else None

    # Decode function for diagnostics (uses the full nonlinear readout)
    def decode_fn(x):
        return wrapper.decode(x, sc_cfg=self_cond_cfg_scale)

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t_val = t_steps[i].item()
            t_next = t_steps[i + 1].item()

            # ── Standard step (identical to existing pipeline) ──
            if method == "sde":
                z, x_pred_new = _sde_step(
                    z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                )
            elif method == "ode":
                z, x_pred_new = _ode_step(
                    z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred,
                    **step_kwargs,
                )
            else:
                raise ValueError(f"Invalid sampling method: {method}")

            # ── Diagnostics ──
            should_record = compute_diagnostics and (
                diagnostic_steps is None or i in diagnostic_steps
            )

            if should_record:
                diag = _record_step_diagnostics(
                    step_idx=i,
                    t=t_next,
                    x_hat=x_pred_new,
                    x_hat_prev=x_pred_prev,
                    z_val=z if store_states else None,
                    decode_fn=decode_fn,
                    top_k=top_k,
                    compute_L_i=compute_L_i,
                    estimator=estimator,
                    probe_type=probe_type,
                    n_probes=n_probes,
                    store_states=store_states,
                )
                trajectory.steps.append(diag)

            x_pred_prev = x_pred.clone() if x_pred is not None else None
            x_pred = x_pred_new

        # ── Last step: always ODE ──
        t_val = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, x_pred_new = _ode_step(
            z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred,
            **step_kwargs,
        )

        # Record final step diagnostics
        if compute_diagnostics:
            diag = _record_step_diagnostics(
                step_idx=n - 2,
                t=t_next,
                x_hat=x_pred_new,
                x_hat_prev=x_pred,
                z_val=z if store_states else None,
                decode_fn=decode_fn,
                top_k=top_k,
                compute_L_i=compute_L_i,
                estimator=estimator,
                probe_type=probe_type,
                n_probes=n_probes,
                store_states=store_states,
            )
            trajectory.steps.append(diag)

    # ── Final decode ──
    final_logits = decode_fn(z)
    trajectory.final_token_ids = final_logits.argmax(dim=-1).cpu()

    # ── Post-hoc: compute agreement of each step's top1 with final decode ──
    if compute_diagnostics and trajectory.steps:
        trajectory.step_agreement = []
        for step_diag in trajectory.steps:
            agreement = (step_diag.top1_ids == trajectory.final_token_ids)
            trajectory.step_agreement.append(agreement)

    return z, trajectory


def _record_step_diagnostics(
    step_idx: int,
    t: float,
    x_hat: torch.Tensor,
    x_hat_prev: Optional[torch.Tensor],
    z_val: Optional[torch.Tensor],
    decode_fn: Callable,
    top_k: int = 8,
    compute_L_i: bool = False,
    estimator=None,
    probe_type: str = "isotropic",
    n_probes: int = 4,
    store_states: bool = False,
) -> StepDiagnostics:
    """Record diagnostics for a single step. All outputs are moved to CPU."""

    # Decode to get logits
    logits = decode_fn(x_hat)  # (B, L, V)

    # Top-k
    topk = logits.topk(top_k, dim=-1)
    top_k_logits = topk.values.half().cpu()   # store as fp16
    top_k_ids = topk.indices.cpu()
    top1_ids = top_k_ids[:, :, 0]
    top2_ids = top_k_ids[:, :, 1]

    # Margin
    margin = (topk.values[:, :, 0] - topk.values[:, :, 1]).float().cpu()

    # Entropy
    entropy = _compute_entropy(logits).float().cpu()

    # Self-conditioning delta
    sc_delta = None
    if x_hat_prev is not None:
        sc_delta = (x_hat - x_hat_prev).norm(dim=-1).float().cpu()

    # Sequence-level: 10th percentile margin
    margin_p10 = torch.quantile(margin.float(), 0.1, dim=-1).cpu()

    # Effective rank
    effective_rank = _compute_effective_rank(x_hat).cpu()

    # L_i estimation
    L_i = None
    rho_i = None
    if compute_L_i and estimator is not None:
        # Generate probe directions
        if probe_type == "isotropic":
            probes = ProbeDirectionGenerator.isotropic(x_hat, n_probes)
        elif probe_type == "decoder_grad":
            probes = ProbeDirectionGenerator.decoder_grad(x_hat, decode_fn, n_probes)
        elif probe_type == "pca":
            probes = ProbeDirectionGenerator.pca(x_hat, n_probes)
        else:
            probes = ProbeDirectionGenerator.isotropic(x_hat, n_probes)

        L_i = estimator.estimate(
            x_hat, decode_fn,
            top1_ids.to(x_hat.device),
            top2_ids.to(x_hat.device),
            probes,
        ).cpu()

        # ρ_i = m_i / L_i (clamp L_i away from 0)
        rho_i = margin / (L_i.clamp(min=1e-6))

    return StepDiagnostics(
        step_idx=step_idx,
        t=t,
        margin=margin,
        top1_ids=top1_ids,
        top2_ids=top2_ids,
        top_k_logits=top_k_logits,
        top_k_ids=top_k_ids,
        entropy=entropy,
        sc_delta=sc_delta,
        L_i=L_i,
        rho_i=rho_i,
        margin_p10=margin_p10,
        effective_rank=effective_rank,
        x_hat_val=x_hat.cpu().half() if store_states else None,
        z_val=z_val.cpu().half() if store_states and z_val is not None else None,
    )
