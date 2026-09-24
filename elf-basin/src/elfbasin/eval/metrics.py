#!/usr/bin/env python3
"""Full metric suite for decoder-basin research.

Every experiment reports the full suite, never PPL alone.
(A recent paper arXiv 2607.00588 shows ELF's low PPL can reflect
a repetition attractor, so a PPL-only claim is not defensible.)

Metrics:
  - Gen. PPL (geometric and arithmetic mean) under GPT-2-Large
  - Unigram entropy, distinct-1/2, repeated-4-gram fraction
  - JS divergence to OWT reference token distribution
  - BLEU (sacreBLEU) for conditional (De-En)
  - ROUGE-1/2/L for summarization (XSum)
  - Token agreement to a reference decode
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ELF_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_ELF_SRC) not in sys.path:
    sys.path.insert(0, str(_ELF_SRC))


def compute_ppl(texts: List[str], model_name: str = "gpt2-large",
                batch_size: int = 16, max_length: int = 1024,
                device: str = "cuda") -> Dict:
    """Compute generative perplexity under a causal LM.

    Returns both geometric and arithmetic mean PPL.
    """
    from utils.metrics_utils import Metrics as PPLMetrics

    ppl_metrics = PPLMetrics(
        gen_ppl_eval_model_name_or_path=model_name,
        eval_ppl_batch_size=batch_size,
        eval_context_size=max_length,
    )

    result = ppl_metrics.record_generative_perplexity(
        texts, max_length=max_length, retokenize=True,
    )

    # Arithmetic mean PPL
    per_sample = result.get("per_sample_ppl", [])
    valid_ppls = [p for p in per_sample if p is not None and not np.isnan(p)]
    arith_ppl = float(np.mean(valid_ppls)) if valid_ppls else float('nan')

    return {
        "gen_ppl_geometric": result["ppl"],
        "gen_ppl_arithmetic": arith_ppl,
        "mean_entropy": result["mean_entropy"],
        "per_sample_ppl": per_sample,
    }


def compute_diversity(texts: List[str]) -> Dict:
    """Compute diversity metrics: unigram entropy, distinct-1/2, rep-4gram."""
    all_words = []
    all_bigrams = []
    total_4gram_count = 0
    repeated_4gram_count = 0

    for text in texts:
        words = text.split()
        all_words.extend(words)
        for i in range(len(words) - 1):
            all_bigrams.append((words[i], words[i + 1]))
        fourgrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        counts = Counter(fourgrams)
        total_4gram_count += len(fourgrams)
        repeated_4gram_count += sum(c - 1 for c in counts.values() if c > 1)

    # Unigram entropy (base 2)
    word_counts = Counter(all_words)
    total = sum(word_counts.values())
    if total > 0:
        probs = np.array([c / total for c in word_counts.values()])
        unigram_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        unigram_entropy = 0.0

    distinct_1 = len(set(all_words)) / max(len(all_words), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    rep_4gram_frac = repeated_4gram_count / max(total_4gram_count, 1)

    return {
        "unigram_entropy": unigram_entropy,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repeated_4gram_fraction": rep_4gram_frac,
        "total_words": total,
        "unique_words": len(set(all_words)),
    }


def compute_js_divergence(
    texts: List[str],
    reference_distribution: Optional[Dict[str, float]] = None,
    tokenizer_name: str = "t5-small",
) -> float:
    """Compute Jensen-Shannon divergence between generated and reference
    token distributions.

    If no reference distribution is provided, returns NaN.
    """
    if reference_distribution is None:
        return float('nan')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Build generated distribution
    gen_counts = Counter()
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        gen_counts.update(ids)

    total = sum(gen_counts.values())
    if total == 0:
        return float('nan')

    # Get all tokens
    all_tokens = set(gen_counts.keys()) | set(reference_distribution.keys())

    p = np.array([gen_counts.get(t, 0) / total for t in all_tokens])
    q = np.array([reference_distribution.get(t, 0) for t in all_tokens])
    q = q / (q.sum() + 1e-12)  # normalize

    # JS divergence
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / (m + 1e-12) + 1e-12))
    js += 0.5 * np.sum(q * np.log(q / (m + 1e-12) + 1e-12))

    return float(js)


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Compute sacreBLEU score."""
    import sacrebleu
    return sacrebleu.corpus_bleu(
        hypotheses, [references],
        lowercase=True, use_effective_order=True,
    ).score


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict:
    """Compute ROUGE-1/2/L scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1, r2, rL = [], [], []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure * 100)
        r2.append(s["rouge2"].fmeasure * 100)
        rL.append(s["rougeL"].fmeasure * 100)

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rL)),
    }


def compute_token_agreement(
    pred_ids_a: np.ndarray,
    pred_ids_b: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict:
    """Compute token-level agreement between two sets of predictions.

    Args:
        pred_ids_a: (N, L) token IDs from decode A
        pred_ids_b: (N, L) token IDs from decode B
        mask: (N, L) binary mask, 1 = valid position

    Returns:
        dict with agreement rate and per-sample agreement
    """
    matches = (pred_ids_a == pred_ids_b)
    if mask is not None:
        matches = matches & mask.astype(bool)
        valid = mask.sum()
    else:
        valid = matches.size

    agreement = float(matches.sum() / max(valid, 1))
    per_sample = matches.mean(axis=-1) if matches.ndim > 1 else matches

    return {
        "token_agreement": agreement,
        "per_sample_agreement": per_sample.tolist() if hasattr(per_sample, 'tolist') else [agreement],
    }


def evaluate_all(
    texts: List[str],
    task: str = "uncond",
    references: Optional[List[str]] = None,
    reference_distribution: Optional[Dict[str, float]] = None,
    pred_ids: Optional[np.ndarray] = None,
    ref_pred_ids: Optional[np.ndarray] = None,
    ppl_model: str = "gpt2-large",
    ppl_batch_size: int = 16,
    max_length: int = 1024,
    skip_ppl: bool = False,
    skip_mauve: bool = True,
) -> Dict:
    """Run the full metric suite.

    Args:
        texts: generated text samples
        task: 'uncond' (OWT), 'de-en', or 'xsum'
        references: reference texts (for BLEU/ROUGE)
        reference_distribution: token distribution for JS divergence
        pred_ids: predicted token IDs for token agreement
        ref_pred_ids: reference token IDs for token agreement
        ppl_model: model for PPL scoring
        ppl_batch_size: batch size for PPL scoring
        max_length: max sequence length
        skip_ppl: skip PPL computation (expensive)
        skip_mauve: skip MAUVE computation

    Returns:
        dict with all metrics
    """
    results = {}

    # Diversity metrics (always computed, cheap)
    results.update(compute_diversity(texts))

    # PPL (expensive)
    if not skip_ppl:
        ppl_results = compute_ppl(
            texts, model_name=ppl_model,
            batch_size=ppl_batch_size, max_length=max_length,
        )
        results["gen_ppl_geometric"] = ppl_results["gen_ppl_geometric"]
        results["gen_ppl_arithmetic"] = ppl_results["gen_ppl_arithmetic"]
        results["mean_entropy"] = ppl_results["mean_entropy"]

    # JS divergence
    if reference_distribution is not None:
        results["js_divergence"] = compute_js_divergence(
            texts, reference_distribution)

    # Task-specific metrics
    if task == "de-en" and references is not None:
        results["bleu"] = compute_bleu(texts, references)
    if task == "xsum" and references is not None:
        results.update(compute_rouge(texts, references))

    # Token agreement
    if pred_ids is not None and ref_pred_ids is not None:
        results.update(compute_token_agreement(pred_ids, ref_pred_ids))

    results["num_samples"] = len(texts)
    results["task"] = task
    return results
