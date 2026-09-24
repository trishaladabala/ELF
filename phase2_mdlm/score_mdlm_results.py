#!/usr/bin/env python3
"""Phase 2 — Score MDLM generated samples with GPT-2 Large.

Reads generated text from MDLM output directories and computes:
1. GPT-2 Large perplexity
2. Degeneracy rate
3. Token-level entropy from MDLM's output logits (if available)
4. Paired statistical comparisons

Run this AFTER run_mdlm_experiments.sh has completed.
"""
import json, sys, math, time, glob, os
from pathlib import Path
from collections import Counter
import numpy as np

PHASE2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE2_DIR / "results" / "mdlm_samples"
OUT = PHASE2_DIR / "results" / "mdlm_scored"


def find_generated_texts(exp_dir):
    """Find and parse generated text samples from MDLM output.
    
    MDLM saves samples in various formats depending on version.
    We try multiple strategies.
    """
    exp_path = Path(exp_dir)
    texts = []

    # Strategy 1: Look for saved .txt or .json files with samples
    for ext in ["*.txt", "*.json", "*.jsonl"]:
        for f in exp_path.rglob(ext):
            if "sample" in f.name.lower() or "gen" in f.name.lower():
                if f.suffix == ".json":
                    try:
                        with open(f) as fp:
                            data = json.load(fp)
                        if isinstance(data, list):
                            texts.extend([str(s) for s in data])
                        elif isinstance(data, dict) and "samples" in data:
                            texts.extend([str(s) for s in data["samples"]])
                    except:
                        pass
                elif f.suffix == ".jsonl":
                    with open(f) as fp:
                        for line in fp:
                            try:
                                texts.append(json.loads(line.strip()))
                            except:
                                pass
                else:
                    with open(f) as fp:
                        content = fp.read()
                    # Split by separator
                    for sep in ["---", "===", "\n\n\n"]:
                        if sep in content:
                            texts.extend([t.strip() for t in content.split(sep) if t.strip()])
                            break
                    else:
                        texts.append(content.strip())

    # Strategy 2: Parse from log file
    if not texts:
        log_files = list(exp_path.parent.glob("*.log")) + list(exp_path.rglob("*.log"))
        for log_file in log_files:
            with open(log_file) as f:
                content = f.read()
            # MDLM often prints samples between markers
            if "Generated sample" in content or "Sample:" in content:
                lines = content.split('\n')
                current = []
                in_sample = False
                for line in lines:
                    if "Generated sample" in line or "Sample:" in line:
                        if current:
                            texts.append('\n'.join(current))
                        current = []
                        in_sample = True
                    elif in_sample and line.strip() == "":
                        if current:
                            texts.append('\n'.join(current))
                            current = []
                        in_sample = False
                    elif in_sample:
                        current.append(line)
                if current:
                    texts.append('\n'.join(current))

    # Strategy 3: Look for PyTorch tensors with token IDs
    if not texts:
        import torch
        for pt_file in exp_path.rglob("*.pt"):
            try:
                data = torch.load(pt_file, map_location="cpu", weights_only=False)
                if isinstance(data, torch.Tensor) and data.dim() == 2:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained("gpt2")
                    for i in range(data.shape[0]):
                        texts.append(tok.decode(data[i].tolist(), skip_special_tokens=True))
            except:
                pass

    return texts


def is_degenerate(text):
    words = text.split()
    if len(words) < 10: return True
    if len(set(words)) <= 5: return True
    if len(words) >= 8:
        ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        if ng and Counter(ng).most_common(1)[0][1] / len(ng) > 0.3: return True
    return False


def score_texts_ppl(texts, device="cuda"):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).half().eval()
    from tqdm import tqdm
    ppls = []
    for t in tqdm(texts, desc="Scoring texts"):
        if not t.strip(): ppls.append(float('inf')); continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=1024)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 2: ppls.append(float('inf')); continue
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            loss = gpt(ids, labels=ids).loss
        ppls.append(math.exp(loss.float().item()))
    del gpt; torch.cuda.empty_cache()
    return ppls


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 2 — Scoring MDLM Samples")
    print("=" * 70)
    t0 = time.time()

    # Find all experiment directories
    exp_dirs = sorted(RESULTS_DIR.glob("T*"))
    if not exp_dirs:
        print(f"  ERROR: No experiment directories found in {RESULTS_DIR}")
        print(f"  Run run_mdlm_experiments.sh first.")
        sys.exit(1)

    conditions = []
    for exp_dir in exp_dirs:
        label = exp_dir.name
        print(f"\n  ── Loading {label} ──")

        texts = find_generated_texts(exp_dir)
        if not texts:
            print(f"    WARNING: No texts found in {exp_dir}")
            continue

        print(f"    Found {len(texts)} texts")
        degs = [is_degenerate(t) for t in texts]
        deg_rate = sum(degs) / len(degs)
        print(f"    Degeneracy: {100*deg_rate:.1f}%")

        conditions.append({
            "label": label,
            "texts": texts,
            "degs": degs,
            "deg_rate": deg_rate,
        })

    if not conditions:
        print("  ERROR: No conditions with text found. Check MDLM output format.")
        sys.exit(1)

    # Score
    print(f"\n  ── PPL Scoring ──")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for c in conditions:
        print(f"    Scoring {c['label']}...")
        c["ppls"] = score_texts_ppl(c["texts"][:512], device)  # Cap at 512
        valid = [p for p, d in zip(c["ppls"], c["degs"]) if not d and not math.isinf(p)]
        c["mean_ppl"] = float(np.mean(valid)) if valid else float('inf')
        c["mean_logppl"] = float(np.mean(np.log(valid))) if valid else float('inf')
        c["n_valid"] = len(valid)
        print(f"      PPL: {c['mean_ppl']:.1f} (N_valid={c['n_valid']})")

    # Paired comparisons (each condition vs T0)
    if any(c["label"] == "T0" for c in conditions):
        base = next(c for c in conditions if c["label"] == "T0")
        for c in conditions:
            if c["label"] == "T0":
                continue
            n = min(len(base["ppls"]), len(c["ppls"]))
            base_lp = np.log(np.array(base["ppls"][:n]))
            c_lp = np.log(np.array(c["ppls"][:n]))
            valid = np.isfinite(base_lp) & np.isfinite(c_lp)
            if valid.sum() > 10:
                diff = float(np.mean(base_lp[valid] - c_lp[valid]))
                np.random.seed(42)
                boots = [np.mean(np.random.choice(base_lp[valid] - c_lp[valid], int(valid.sum()), replace=True))
                         for _ in range(5000)]
                ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            else:
                diff = float('nan'); ci = [float('nan'), float('nan')]
            c["vs_T0_diff"] = diff; c["vs_T0_ci"] = ci
            sign = "+" if diff > 0 else "-"
            print(f"    {c['label']} vs T0: {sign}{abs(diff):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Condition':>12s}  {'PPL':>8s}  {'Deg':>6s}  {'N_valid':>8s}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c['label']:>12s}  {c['mean_ppl']:8.1f}  {100*c['deg_rate']:5.1f}%  {c['n_valid']:>8d}")

    # Save
    summary = {
        "model": "MDLM (kuleshov-group/mdlm-owt)",
        "architecture": "Masked Discrete Diffusion (DiT)",
        "conditions": [{
            "label": c["label"],
            "deg_rate": c["deg_rate"],
            "mean_ppl": c["mean_ppl"],
            "mean_logppl": c["mean_logppl"],
            "n_valid": c["n_valid"],
            "vs_T0_diff": c.get("vs_T0_diff"),
            "vs_T0_ci": c.get("vs_T0_ci"),
        } for c in conditions],
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "mdlm_scored.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "mdlm_scored.md", "w") as f:
        f.write("# Phase 2 — MDLM: Stochastic vs Deterministic (Scored)\n\n")
        f.write("| Condition | PPL ↓ | Degeneracy | vs T=0 | 95% CI |\n")
        f.write("|---|---|---|---|---|\n")
        for c in conditions:
            d = c.get("vs_T0_diff", "-")
            ci = c.get("vs_T0_ci", ["-", "-"])
            if isinstance(d, float) and not math.isnan(d):
                f.write(f"| {c['label']} | {c['mean_ppl']:.1f} | "
                        f"{100*c['deg_rate']:.1f}% | {d:+.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |\n")
            else:
                f.write(f"| {c['label']} | {c['mean_ppl']:.1f} | {100*c['deg_rate']:.1f}% | — | — |\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
