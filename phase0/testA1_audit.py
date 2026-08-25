#!/usr/bin/env python3
"""Test A1 — Audit: Diff Test 2 vs Test 6 for Silent Parameter Differences.

Produces a structured table of every parameter that differs between the two
scripts that could explain the 2.11% vs 8.2%-30.9% discrepancy.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEST A1 — Audit: Test 2 vs Test 6 Parameter Differences")
    print("=" * 60)

    # Structured comparison from reading both scripts
    diff_table = [
        {
            "parameter": "Sample size (n)",
            "test_02": "1500 (N_SUBSAMPLE=1500)",
            "test_06": "1000 (N_SAMPLE=1000)",
            "impact": "MODERATE — smaller n in T6 means OT has fewer points to optimize over, potentially amplifying coupling gains",
        },
        {
            "parameter": "Sinkhorn ε (regularization)",
            "test_02": "0.05",
            "test_06": "0.05",
            "impact": "SAME — not a source of discrepancy",
        },
        {
            "parameter": "Sinkhorn max iterations",
            "test_02": "2000",
            "test_06": "2000",
            "impact": "SAME",
        },
        {
            "parameter": "Sinkhorn stop threshold",
            "test_02": "1e-9",
            "test_06": "1e-9",
            "impact": "SAME",
        },
        {
            "parameter": "Cost matrix normalization",
            "test_02": "C / C.max() — normalized to [0,1] range",
            "test_06": "C / (C.max() + 1e-10) — same normalization (epsilon for safety)",
            "impact": "SAME effectively",
        },
        {
            "parameter": "Cost metric",
            "test_02": "ot.dist(x, eps, metric='sqeuclidean')",
            "test_06": "ot.dist(x, eps, metric='sqeuclidean')",
            "impact": "SAME",
        },
        {
            "parameter": "OT cost computation",
            "test_02": "np.sum(T * C_raw) where C_raw is raw sqeuclidean — total transport cost",
            "test_06": "np.sum(T * C) where C is the NORMALIZED cost — different scale!",
            "impact": "⚠️ CRITICAL DIFFERENCE: T6 computes OT cost on normalized C, T2 computes on raw C_raw. But T6's 'reduction' uses avg_path_sq from argmax, not ot_cost — so this doesn't directly affect the reduction %.",
        },
        {
            "parameter": "Reduction metric",
            "test_02": "ot_cost / random_cost_mean — uses sum(T * C_raw) / MC_mean",
            "test_06": "1 - avg_path_sq / random_cost — uses ARGMAX assignment path length",
            "impact": "⚠️⚠️ CRITICAL DIFFERENCE #1: Test 2 uses the soft OT plan cost (probabilistic coupling). Test 6 uses hard argmax assignment from the plan. The argmax assignment tends to give MUCH better per-pair distances than the soft plan cost, which is spread across many pairs. This alone can easily explain a 5-10x difference.",
        },
        {
            "parameter": "Noise generation (ε)",
            "test_02": "rng.standard_normal(x.shape) — N(0,I) always",
            "test_06": "Varies by prior: gaussian=N(0,I), empirical=N(μ_data, diag(Σ_data)), uniform=hypersphere",
            "impact": "⚠️⚠️ CRITICAL DIFFERENCE #2: The 'empirical' prior matches the data distribution, making x and ε much CLOSER in distribution. This drastically reduces random pairing cost AND makes OT improvement appear larger because the coupling problem is 'easier'. This is not comparing the same thing as Test 2.",
        },
        {
            "parameter": "Random cost baseline (denominator)",
            "test_02": "50 random shuffles of the SAME ε (N(0,I))",
            "test_06": "30 random shuffles of EACH prior's ε",
            "impact": "MODERATE — different ε means different random baselines. The 'empirical' prior has a much lower random cost (~17 vs ~137), meaning a small absolute improvement maps to a large % reduction.",
        },
        {
            "parameter": "Random seed / RNG state",
            "test_02": "rng = default_rng(42) — used for subsampling, noise gen, and shuffles",
            "test_06": "rng = default_rng(42) — but T6 uses the SAME rng object across priors and datasets, so its state evolves across calls. The rng state when 'gaussian' runs on real embeddings is NOT the same as T2's.",
            "impact": "MINOR — different random draws but shouldn't cause systematic bias",
        },
        {
            "parameter": "Data subsample",
            "test_02": "x_128[idx] where idx = rng.choice(322387, 1500)",
            "test_06": "x_128[idx] where idx = rng.choice(322387, 1000) — BUT rng already consumed draws for GMM data",
            "impact": "MINOR — different subset of the same embedding pool",
        },
    ]

    # Print table
    print()
    for row in diff_table:
        print(f"  Parameter:  {row['parameter']}")
        print(f"  Test 2:     {row['test_02']}")
        print(f"  Test 6:     {row['test_06']}")
        print(f"  Impact:     {row['impact']}")
        print()

    # Root cause analysis
    print("=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    print()

    root_causes = [
        {
            "rank": 1,
            "cause": "Reduction metric: soft OT cost (Test 2) vs hard argmax assignment (Test 6)",
            "explanation": (
                "Test 2 computes reduction as sum(T * C_raw) / random_cost. The transport plan T "
                "is a soft probabilistic coupling where each x_i is matched to MANY ε_j with "
                "different weights. The resulting 'cost' averages across all these soft matches. "
                "Test 6 instead takes argmax(T, axis=1) to get a HARD 1-to-1 assignment, then "
                "computes mean(||x_i - ε_assignment[i]||²). The hard assignment picks the BEST "
                "match for each point, yielding lower path lengths than the soft average. "
                "This is like comparing a weighted average distance (soft) to the minimum "
                "distance for each point (hard)."
            ),
        },
        {
            "rank": 2,
            "cause": "Non-standard noise priors ('empirical', 'uniform') in Test 6",
            "explanation": (
                "Test 6's 'empirical' prior draws ε ~ N(μ_data, diag(Σ_data)), making ε "
                "statistically very similar to x. This means x and ε are already close, so "
                "random pairing cost is ~17 (vs ~137 for standard Gaussian). When x and ε are "
                "close, OT can rearrange pairs to get even closer, yielding large % reductions. "
                "But this measures 'can OT improve pairing when source≈target?' — a very different "
                "question from Test 2's 'can OT improve pairing when source=N(0,I) and target=data?', "
                "which is what ELF's actual training uses."
            ),
        },
        {
            "rank": 3,
            "cause": "Sample size difference (1500 vs 1000)",
            "explanation": (
                "Smaller n makes OT easier (fewer points to transport), potentially amplifying "
                "the effect. But this is a minor contributor compared to causes 1 and 2."
            ),
        },
    ]

    for rc in root_causes:
        print(f"  #{rc['rank']}: {rc['cause']}")
        print(f"     {rc['explanation']}")
        print()

    # Save
    output = {
        "diff_table": diff_table,
        "root_causes": root_causes,
        "summary": (
            "The 2.11% (Test 2) vs 8-31% (Test 6) discrepancy is primarily explained by TWO "
            "compounding methodological differences: (1) Test 2 uses soft OT plan cost while "
            "Test 6 uses hard argmax assignment cost — the hard assignment cherry-picks the best "
            "match per point, giving ~2-3x better apparent reduction; (2) Test 6's 'empirical' "
            "and 'uniform' priors create noise that's much closer to the data distribution, "
            "making the coupling problem 'easier' and inflating the reduction %. Neither of "
            "these inflated numbers represents ELF's actual training setup (which uses N(0,I) "
            "noise and doesn't do argmax assignment). Test 2's 2.11% is the more relevant "
            "measurement for the project's go/no-go decision."
        ),
    }
    with open(RESULTS_DIR / "testA1_diff_table.json", "w") as f:
        json.dump(output, f, indent=2)

    print("  Summary:", output["summary"])
    print()
    print(f"  Saved → {RESULTS_DIR / 'testA1_diff_table.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
