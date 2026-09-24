#!/usr/bin/env python3
"""Generate analysis plots and paper assets for distillation results.

Produces:
  1. Speed-quality Pareto frontier (BLEU vs wall-clock time)
  2. BLEU vs number of steps comparison
  3. Diversity metrics comparison
  4. Qualitative example table
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = _SCRIPT_DIR.parent / "runs" / "eval_results"
OUT_DIR = _SCRIPT_DIR.parent / "runs" / "analysis"


def load_results():
    """Load evaluation results."""
    results_path = EVAL_DIR / "eval_results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run eval_distilled.py first.")
        return None
    with open(results_path) as f:
        return json.load(f)


def generate_latex_table(results):
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Speed-Quality Comparison of Distillation Methods on WMT14 De-En}")
    lines.append(r"\label{tab:distillation}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Steps & BLEU$\uparrow$ & Dist-2$\uparrow$ & Rep-4$\downarrow$ & ms/sample$\downarrow$ \\")
    lines.append(r"\midrule")

    for model_name, model_results in results.items():
        display_name = model_name.replace("_", " ").title()
        for n_steps in sorted(model_results.keys(), key=int):
            r = model_results[n_steps]
            lines.append(
                f"{display_name} & {n_steps} & {r['bleu']:.1f} & "
                f"{r['dist2']:.3f} & {r['rep4']:.3f} & "
                f"{r['time_per_sample_s']*1000:.0f} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_markdown_table(results):
    """Generate markdown summary table."""
    lines = []
    lines.append("# Distillation Results Summary")
    lines.append("")
    lines.append("| Model | Steps | BLEU ↑ | Dist-2 ↑ | Rep-4 ↓ | ms/sample ↓ | Speedup |")
    lines.append("|-------|-------|--------|----------|---------|-------------|---------|")

    # Find teacher 64-step time for speedup calculation
    teacher_time = None
    if "teacher" in results and "64" in results["teacher"]:
        teacher_time = results["teacher"]["64"]["time_per_sample_s"]

    for model_name, model_results in results.items():
        for n_steps in sorted(model_results.keys(), key=lambda x: int(x)):
            r = model_results[n_steps]
            speedup = ""
            if teacher_time and r["time_per_sample_s"] > 0:
                speedup = f"{teacher_time / r['time_per_sample_s']:.1f}×"
            lines.append(
                f"| {model_name} | {n_steps} | {r['bleu']:.1f} | "
                f"{r['dist2']:.3f} | {r['rep4']:.3f} | "
                f"{r['time_per_sample_s']*1000:.0f} | {speedup} |"
            )

    return "\n".join(lines)


def try_generate_plots(results):
    """Try to generate matplotlib plots. Falls back to text if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # ── Plot 1: BLEU vs Steps ──────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        colors = {
            "teacher": "#2196F3",
            "consistency_ema": "#E91E63",
            "consistency_student": "#FF5722",
        }
        # Add colors for progressive models
        prog_colors = ["#4CAF50", "#8BC34A", "#CDDC39", "#FFC107"]
        for i, n in enumerate([32, 16, 8, 4]):
            colors[f"progressive_{n}step"] = prog_colors[i]

        for model_name, model_results in results.items():
            steps = sorted(model_results.keys(), key=lambda x: int(x))
            bleus = [model_results[s]["bleu"] for s in steps]
            steps_int = [int(s) for s in steps]
            color = colors.get(model_name, "#607D8B")
            label = model_name.replace("_", " ").title()
            ax.plot(steps_int, bleus, 'o-', label=label, color=color, linewidth=2, markersize=8)

        ax.set_xlabel("Number of ODE Steps", fontsize=14)
        ax.set_ylabel("BLEU Score", fontsize=14)
        ax.set_title("BLEU vs. Number of Steps — Distillation Comparison", fontsize=16)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "bleu_vs_steps.png", dpi=150)
        plt.close()
        print(f"  Saved: {OUT_DIR / 'bleu_vs_steps.png'}")

        # ── Plot 2: Speed-Quality Pareto ───────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for model_name, model_results in results.items():
            steps = sorted(model_results.keys(), key=lambda x: int(x))
            bleus = [model_results[s]["bleu"] for s in steps]
            times = [model_results[s]["time_per_sample_s"] * 1000 for s in steps]
            color = colors.get(model_name, "#607D8B")
            label = model_name.replace("_", " ").title()
            ax.plot(times, bleus, 'o-', label=label, color=color, linewidth=2, markersize=8)
            # Annotate with step count
            for t, b, s in zip(times, bleus, steps):
                ax.annotate(f"{s}", (t, b), textcoords="offset points",
                           xytext=(5, 5), fontsize=8, alpha=0.7)

        ax.set_xlabel("Time per Sample (ms)", fontsize=14)
        ax.set_ylabel("BLEU Score", fontsize=14)
        ax.set_title("Speed-Quality Pareto Frontier", fontsize=16)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "pareto_frontier.png", dpi=150)
        plt.close()
        print(f"  Saved: {OUT_DIR / 'pareto_frontier.png'}")

        # ── Plot 3: Diversity Metrics ──────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for model_name, model_results in results.items():
            steps = sorted(model_results.keys(), key=lambda x: int(x))
            dist2s = [model_results[s]["dist2"] for s in steps]
            rep4s = [model_results[s]["rep4"] for s in steps]
            steps_int = [int(s) for s in steps]
            color = colors.get(model_name, "#607D8B")
            label = model_name.replace("_", " ").title()
            ax1.plot(steps_int, dist2s, 'o-', label=label, color=color, linewidth=2)
            ax2.plot(steps_int, rep4s, 'o-', label=label, color=color, linewidth=2)

        ax1.set_xlabel("Number of Steps")
        ax1.set_ylabel("Distinct-2 ↑")
        ax1.set_title("Lexical Diversity")
        ax1.set_xscale('log', base=2)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Number of Steps")
        ax2.set_ylabel("Rep-4 ↓")
        ax2.set_title("4-gram Repetition")
        ax2.set_xscale('log', base=2)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT_DIR / "diversity_metrics.png", dpi=150)
        plt.close()
        print(f"  Saved: {OUT_DIR / 'diversity_metrics.png'}")

        return True

    except ImportError:
        print("  matplotlib not available, skipping plots")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Generating Analysis & Paper Assets")
    print("=" * 70)

    results = load_results()
    if results is None:
        return 1

    # ── Generate tables ────────────────────────────────────────────────
    print("\nGenerating tables...")

    latex = generate_latex_table(results)
    with open(OUT_DIR / "results_table.tex", "w") as f:
        f.write(latex)
    print(f"  Saved: {OUT_DIR / 'results_table.tex'}")

    markdown = generate_markdown_table(results)
    with open(OUT_DIR / "results_summary.md", "w") as f:
        f.write(markdown)
    print(f"  Saved: {OUT_DIR / 'results_summary.md'}")
    print(f"\n{markdown}")

    # ── Generate plots ─────────────────────────────────────────────────
    print("\nGenerating plots...")
    try_generate_plots(results)

    # ── Key findings ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    if "teacher" in results:
        teacher_64 = results["teacher"].get("64", {})
        if teacher_64:
            print(f"  Teacher (64-step): {teacher_64.get('bleu', 'N/A'):.1f} BLEU")

    for model_name in ["consistency_ema", "consistency_student"]:
        if model_name in results:
            for steps in ["4", "2", "1"]:
                if steps in results[model_name]:
                    r = results[model_name][steps]
                    print(f"  {model_name} ({steps}-step): {r['bleu']:.1f} BLEU")

    for n in [4, 8]:
        key = f"progressive_{n}step"
        if key in results and str(n) in results[key]:
            r = results[key][str(n)]
            print(f"  Progressive ({n}-step): {r['bleu']:.1f} BLEU")

    print(f"\nAll assets saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
