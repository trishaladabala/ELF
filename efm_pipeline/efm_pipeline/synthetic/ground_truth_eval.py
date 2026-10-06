#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

def wave_order_accuracy(predicted_tau: torch.Tensor, true_wave_ids: torch.Tensor) -> float:
    B, L = predicted_tau.shape
    correct, total = 0, 0
    for b in range(B):
        for i in range(L):
            for j in range(i + 1, L):
                wi = true_wave_ids[b, i].item()
                wj = true_wave_ids[b, j].item()
                if wi == wj: continue
                total += 1
                ti, tj = predicted_tau[b, i].item(), predicted_tau[b, j].item()
                if (wi < wj and ti < tj) or (wi > wj and ti > tj): correct += 1
    return correct / max(total, 1)

def dependency_accuracy(generated_token_ids: torch.Tensor, true_token_ids: torch.Tensor, wave_ids: torch.Tensor, vocab_size: int) -> dict:
    """External dependency accuracy: do generated values match GROUND-TRUTH keys?

    This metric is meaningful for teacher-forced evaluation where the model
    sees the true key positions. For unconditional generation (where the model
    invents its own keys), use internal_dependency_accuracy instead.
    """
    B, L = generated_token_ids.shape
    W = wave_ids.max().item() + 1
    M = L // W
    num_keys = (W - 1) * M
    
    results = {}
    total_correct = 0
    total_count = 0
    
    for w in range(1, W):
        val_waves = wave_ids[:, num_keys:]
        mask_w = (val_waves == w)
        count = mask_w.sum().item()
        
        if count > 0:
            # Check if generated values match the TRUE (ground-truth) keys
            true_source_keys = true_token_ids[:, (w-1)*M : w*M]
            generated_vals = generated_token_ids[:, num_keys:]
            matches = (generated_vals == true_source_keys) & mask_w
            correct = matches.sum().item()
            results[f"wave_{w}_acc"] = correct / count
            total_correct += correct
            total_count += count
        else:
            results[f"wave_{w}_acc"] = 0.0
            
    results["overall_dep_acc"] = total_correct / max(total_count, 1)
    return results


def internal_dependency_accuracy(generated_token_ids: torch.Tensor, wave_ids: torch.Tensor, vocab_size: int) -> dict:
    """Internal consistency: do generated values copy from the model's OWN generated keys?

    For unconditional generation, the model starts from random noise and generates
    its own arbitrary key tokens. The correct test of wave-routing is whether
    the generated value tokens are internally consistent with the generated keys,
    NOT whether they match ground-truth keys (which will always be ~1/vocab_size).

    This is the primary metric for evaluating unconditional synthetic generation.
    """
    B, L = generated_token_ids.shape
    W = wave_ids.max().item() + 1
    M = L // W
    num_keys = (W - 1) * M

    results = {}
    total_correct = 0
    total_count = 0

    for w in range(1, W):
        val_waves = wave_ids[:, num_keys:]
        mask_w = (val_waves == w)
        count = mask_w.sum().item()

        if count > 0:
            # Check if generated values match the MODEL'S OWN generated keys
            gen_source_keys = generated_token_ids[:, (w-1)*M : w*M]
            generated_vals = generated_token_ids[:, num_keys:]
            matches = (generated_vals == gen_source_keys) & mask_w
            correct = matches.sum().item()
            results[f"wave_{w}_internal_acc"] = correct / count
            total_correct += correct
            total_count += count
        else:
            results[f"wave_{w}_internal_acc"] = 0.0

    results["overall_internal_dep_acc"] = total_correct / max(total_count, 1)
    return results

def synthetic_nll(model_logits, target_ids, wave_ids):
    log_probs = F.log_softmax(model_logits, dim=-1)
    target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    results = {"overall_nll": -target_log_probs.mean().item()}
    return results

def full_evaluation(
    generated_token_ids: torch.Tensor, true_token_ids: torch.Tensor,
    wave_ids: torch.Tensor, vocab_size: int,
    predicted_tau: Optional[torch.Tensor] = None, model_logits: Optional[torch.Tensor] = None,
) -> dict:
    # External: generated values vs ground-truth keys (valid for teacher-forced)
    metrics = dependency_accuracy(generated_token_ids, true_token_ids, wave_ids, vocab_size)

    # Internal: generated values vs model's own generated keys (valid for unconditional)
    internal_metrics = internal_dependency_accuracy(generated_token_ids, wave_ids, vocab_size)
    metrics.update(internal_metrics)

    overall_match = (generated_token_ids == true_token_ids).float().mean().item()
    metrics["overall_exact_match"] = overall_match
    return metrics
