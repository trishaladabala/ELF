#!/usr/bin/env python3
"""Ground-truth evaluation for ordered assembly sequences.

Because we know the exact generative process, we can measure things that
Gen-PPL cannot:
  1. Wave-order accuracy: does the model assign lower τ to earlier-wave tokens?
  2. Dependency accuracy: are wave-w tokens correct functions of their neighbors?
  3. Token-level NLL against the known conditional distribution.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def wave_order_accuracy(
    predicted_tau: torch.Tensor,
    true_wave_ids: torch.Tensor,
) -> float:
    """Fraction of token pairs where τ ordering matches wave ordering.

    For every pair (i, j) where wave_i < wave_j, check that τ_i < τ_j.
    This measures whether the model has learned the correct insertion order.

    Args:
        predicted_tau: (B, L) predicted local times from the model.
        true_wave_ids: (B, L) ground-truth wave indices.

    Returns:
        Fraction of correctly ordered pairs (higher = better, 1.0 = perfect).
    """
    B, L = predicted_tau.shape
    correct = 0
    total = 0

    for b in range(B):
        for i in range(L):
            for j in range(i + 1, L):
                wi = true_wave_ids[b, i].item()
                wj = true_wave_ids[b, j].item()
                if wi == wj:
                    continue
                total += 1
                ti = predicted_tau[b, i].item()
                tj = predicted_tau[b, j].item()
                if (wi < wj and ti < tj) or (wi > wj and ti > tj):
                    correct += 1

    return correct / max(total, 1)


def dependency_accuracy(
    generated_token_ids: torch.Tensor,
    wave_ids: torch.Tensor,
    vocab_size: int,
) -> dict:
    """Check whether generated wave-w tokens are correct functions of neighbors.

    In the ordered assembly task, wave-w tokens are defined as:
        token[i] = (left_neighbor + right_neighbor + w) % vocab_size
    where neighbors come from earlier waves.

    Args:
        generated_token_ids: (B, L) generated token sequences.
        wave_ids: (B, L) wave assignments (same for all samples if
                  the pattern is deterministic).
        vocab_size: vocabulary size.

    Returns:
        dict with per-wave accuracy and overall accuracy.
    """
    B, L = generated_token_ids.shape
    W = wave_ids.max().item() + 1
    wave_template = wave_ids[0]  # Assume same pattern for all samples

    results = {}
    total_correct = 0
    total_count = 0

    for w in range(1, W):
        mask_w = (wave_template == w)
        positions_w = mask_w.nonzero(as_tuple=True)[0]
        correct = 0
        count = 0

        for pos in positions_w:
            # Find nearest earlier-wave neighbors
            left_val = torch.zeros(B, dtype=torch.long)
            right_val = torch.zeros(B, dtype=torch.long)

            for j in range(pos.item() - 1, -1, -1):
                if wave_template[j] < w:
                    left_val = generated_token_ids[:, j]
                    break

            for j in range(pos.item() + 1, L):
                if wave_template[j] < w:
                    right_val = generated_token_ids[:, j]
                    break

            expected = (left_val + right_val + w) % vocab_size
            actual = generated_token_ids[:, pos]
            correct += (expected == actual).sum().item()
            count += B

        acc = correct / max(count, 1)
        results[f"wave_{w}_acc"] = acc
        total_correct += correct
        total_count += count

    results["overall_dep_acc"] = total_correct / max(total_count, 1)
    return results


def synthetic_nll(
    model_logits: torch.Tensor,
    target_ids: torch.Tensor,
    wave_ids: torch.Tensor,
) -> dict:
    """Compute per-wave NLL from model logits.

    Args:
        model_logits: (B, L, V) logits from the decoder.
        target_ids: (B, L) ground-truth token ids.
        wave_ids: (B, L) wave assignments.

    Returns:
        dict with per-wave NLL and overall NLL.
    """
    B, L, V = model_logits.shape
    W = wave_ids.max().item() + 1

    # Overall NLL
    log_probs = F.log_softmax(model_logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    results = {}
    for w in range(W):
        mask = (wave_ids == w)
        if mask.any():
            wave_nll = -target_log_probs[mask].mean().item()
            results[f"wave_{w}_nll"] = wave_nll

    results["overall_nll"] = -target_log_probs.mean().item()
    return results


def full_evaluation(
    generated_token_ids: torch.Tensor,
    true_token_ids: torch.Tensor,
    wave_ids: torch.Tensor,
    vocab_size: int,
    predicted_tau: Optional[torch.Tensor] = None,
    model_logits: Optional[torch.Tensor] = None,
) -> dict:
    """Run all ground-truth evaluations.

    Args:
        generated_token_ids: (B, L) model outputs.
        true_token_ids: (B, L) ground-truth sequences.
        wave_ids: (B, L) wave assignments.
        vocab_size: vocabulary size.
        predicted_tau: (B, L) optional predicted local times.
        model_logits: (B, L, V) optional decoder logits.

    Returns:
        Combined metrics dict.
    """
    metrics = {}

    # 1. Dependency accuracy
    dep = dependency_accuracy(generated_token_ids, wave_ids, vocab_size)
    metrics.update(dep)

    # 2. Token-level exact match (per wave)
    W = wave_ids.max().item() + 1
    for w in range(W):
        mask = (wave_ids == w)
        if mask.any():
            acc = (generated_token_ids[mask] == true_token_ids[mask]).float().mean().item()
            metrics[f"wave_{w}_exact_match"] = acc
    overall_match = (generated_token_ids == true_token_ids).float().mean().item()
    metrics["overall_exact_match"] = overall_match

    # 3. Wave-order accuracy (if τ predictions available)
    if predicted_tau is not None:
        woa = wave_order_accuracy(predicted_tau, wave_ids)
        metrics["wave_order_accuracy"] = woa

    # 4. NLL (if logits available)
    if model_logits is not None:
        nll = synthetic_nll(model_logits, true_token_ids, wave_ids)
        metrics.update(nll)

    return metrics


if __name__ == "__main__":
    from efm_pipeline.synthetic.ordered_assembly import generate_ordered_assembly_data

    # Quick self-test
    data = generate_ordered_assembly_data(
        num_samples=8, seq_len=32, vocab_size=64, W=4, seed=0
    )
    tids = data["token_ids"]
    wids = data["wave_ids"]

    # Perfect prediction = the ground truth itself
    metrics = full_evaluation(
        generated_token_ids=tids,
        true_token_ids=tids,
        wave_ids=wids,
        vocab_size=64,
    )

    print("Ground-truth self-evaluation (should be perfect):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    assert metrics["overall_exact_match"] == 1.0
    assert metrics["overall_dep_acc"] == 1.0
    print("\n✓ Self-test passed.")

