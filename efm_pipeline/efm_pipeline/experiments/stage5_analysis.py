#!/usr/bin/env python3
"""Stage 5 — Analysis and ablations.

Aggregates all results from Stages 0-4 and produces:
  - K vs Gen-PPL curves (frozen quant / learned quant / low-rank)
  - Comparison table: ELF-B vs continuous EFM vs best compressed
  - Quality-vs-complexity tradeoff data
  - Summary CSV and JSON

Usage:
    python -m efm_pipeline.experiments.stage5_analysis --results_dir results
"""

import argparse
import csv
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from efm_pipeline.logger import load_all_results
from efm_pipeline.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Analysis")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    all_results = load_all_results(args.results_dir)
    if not all_results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"\n{'='*80}")
    print(f"  STAGE 5: COMPREHENSIVE ANALYSIS")
    print(f"  Found {len(all_results)} experiment results")
    print(f"{'='*80}\n")

    # --- 1. Baseline comparison ---
    print("=== Baseline Comparison ===")
    print(f"{'Experiment':<40} | {'Gen-PPL':>10} | {'Entropy':>10} | {'3-gram Rep':>10}")
    print(f"{'-'*40}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    baseline_keys = ["stage0_elf_baseline", "stage2_continuous_eval"]
    for key in baseline_keys:
        if key in all_results:
            r = all_results[key]
            print(f"{key:<40} | {r.get('gen_ppl', 0):>10.2f} | "
                  f"{r.get('unigram_entropy_bits', r.get('mean_entropy', 0)):>10.4f} | "
                  f"{r.get('trigram_repetition', 0):>10.4f}")
    print()

    # --- 2. Quantization sweep (frozen vs learned) ---
    print("=== Quantization: Frozen vs Learned ===")
    print(f"{'K':>4} | {'Frozen PPL':>12} | {'Learned PPL':>12} | {'Δ PPL':>10}")
    print(f"{'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    frozen_results = {k: v for k, v in all_results.items() if "quant_frozen" in k}
    learned_results = {k: v for k, v in all_results.items() if "quant_learned" in k}

    K_values_seen = set()
    for k, v in {**frozen_results, **learned_results}.items():
        if "K" in v:
            K_values_seen.add(v["K"])

    for K in sorted(K_values_seen):
        frozen_ppl = None
        learned_ppl = None
        for v in frozen_results.values():
            if v.get("K") == K:
                frozen_ppl = v.get("gen_ppl")
        for v in learned_results.values():
            if v.get("K") == K:
                learned_ppl = v.get("gen_ppl")

        f_str = f"{frozen_ppl:>12.2f}" if frozen_ppl else f"{'N/A':>12}"
        l_str = f"{learned_ppl:>12.2f}" if learned_ppl else f"{'N/A':>12}"
        delta = ""
        if frozen_ppl and learned_ppl:
            delta = f"{learned_ppl - frozen_ppl:>+10.2f}"
        print(f"{K:>4} | {f_str} | {l_str} | {delta:>10}")
    print()

    # --- 3. Low-rank sweep ---
    print("=== Low-Rank Compression ===")
    lowrank_results = {k: v for k, v in all_results.items() if "lowrank" in k}
    if lowrank_results:
        print(f"{'K':>4} | {'Basis':>8} | {'Gen-PPL':>10} | {'Entropy':>10}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
        for k in sorted(lowrank_results.keys()):
            v = lowrank_results[k]
            print(f"{v.get('K', '?'):>4} | {v.get('basis', '?'):>8} | "
                  f"{v.get('gen_ppl', 0):>10.2f} | "
                  f"{v.get('unigram_entropy_bits', 0):>10.4f}")
    else:
        print("  No low-rank results found.")
    print()

    # --- 4. Summary table (all experiments) ---
    print("=== Full Summary ===")
    summary_rows = []
    for name, r in sorted(all_results.items()):
        row = {
            "experiment": name,
            "gen_ppl": r.get("gen_ppl", ""),
            "entropy": r.get("unigram_entropy_bits", r.get("mean_entropy", "")),
            "trigram_rep": r.get("trigram_repetition", ""),
            "K": r.get("K", ""),
            "mode": r.get("mode", ""),
            "num_samples": r.get("num_samples", ""),
        }
        summary_rows.append(row)
        print(f"  {name:<50} Gen-PPL={r.get('gen_ppl', 'N/A')}")

    # Save CSV.
    csv_path = os.path.join(ensure_dir(args.output_dir), "stage5_full_summary.csv")
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nCSV summary saved → {csv_path}")

    # Save JSON.
    json_path = os.path.join(args.output_dir, "stage5_full_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON summary saved → {json_path}")

    print(f"\n{'='*80}")
    print("  Analysis complete.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
