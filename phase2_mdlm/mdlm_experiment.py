#!/usr/bin/env python3
"""Phase 2 — MDLM Experiment: Stochastic vs Deterministic Sampling.

Replicates our key ELF findings on MDLM (Masked Diffusion Language Model).

Uses MDLM's official main.py via subprocess to avoid import issues with
optional dependencies (mamba, causal-conv1d). The HF DiT model handles
everything internally.

Strategy:
1. Call MDLM main.py in sample_eval mode at different temperatures
2. Collect generated samples from output files
3. Score everything with GPT-2 Large PPL
"""
import json, sys, math, time, os, subprocess, re, glob
from pathlib import Path
from collections import Counter
import numpy as np

PHASE2_DIR = Path(__file__).resolve().parent
MDLM_DIR = PHASE2_DIR / "mdlm"
OUT = PHASE2_DIR / "results" / "mdlm_stochastic"

N = 512
BS = 4
STEPS = 1000
MAX_LEN = 1024
PYTHON = sys.executable  # Use same Python that's running this script


def is_degenerate(text):
    """Check if generated text is degenerate."""
    words = text.split()
    if len(words) < 10: return True
    if len(set(words)) <= 5: return True
    if len(words) >= 8:
        ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        if ng and Counter(ng).most_common(1)[0][1] / len(ng) > 0.3: return True
    return False


def generate_with_mdlm_main(temperature, output_dir):
    """Call MDLM main.py to generate samples at given temperature.
    
    MDLM's sample_eval mode generates samples and prints them to stdout.
    We capture everything and parse the generated text.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_batches = N // BS
    
    cmd = [
        PYTHON, str(MDLM_DIR / "main.py"),
        "mode=sample_eval",
        "eval.checkpoint_path=kuleshov-group/mdlm-owt",
        "data=openwebtext-split",
        f"model.length={MAX_LEN}",
        "sampling.predictor=ddpm_cache",
        f"sampling.steps={STEPS}",
        f"loader.eval_batch_size={BS}",
        f"sampling.num_sample_batches={n_batches}",
        "backbone=hf_dit",
        "+wandb.offline=true",
        f"hydra.run.dir={output_dir}",
    ]
    
    # Note: MDLM's default is temperature=0 (deterministic).
    # We add temperature override for stochastic sampling.
    if temperature > 0:
        cmd.append(f"+sampling.temperature={temperature}")
    
    print(f"    Running MDLM main.py (T={temperature})...")
    print(f"    CMD: {' '.join(cmd[-5:])}")
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True, 
        cwd=str(MDLM_DIR),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        bufsize=1
    )
    
    stdout_lines = []
    try:
        with open(output_dir / "stdout.txt", "w") as f:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                stdout_lines.append(line)
                
        process.wait()
    finally:
        # Ensure the child process is terminated if we exit this block early (e.g. from Ctrl+C)
        if process.poll() is None:
            process.terminate()
            process.wait()
    full_output = "".join(stdout_lines)
    
    with open(output_dir / "stderr.txt", "w") as f:
        f.write("Stderr was merged with stdout for streaming.")
    
    if process.returncode != 0:
        print(f"    WARNING: main.py exited with code {process.returncode}")
        print(f"    Last 500 chars of output: {full_output[-500:]}")
    
    return full_output, full_output


def parse_mdlm_outputs(output_dir):
    """Parse generated texts from MDLM output directory.
    
    MDLM may save samples as:
    - .txt files in the output dir
    - printed to stdout (captured separately)
    - .pt tensor files with token IDs
    """
    output_dir = Path(output_dir)
    texts = []
    
    # Strategy 1: Look for any text/json files with samples
    for pattern in ["**/*.txt", "**/*.json", "**/*.jsonl"]:
        for f in output_dir.glob(pattern):
            if f.name in ("stdout.txt", "stderr.txt"):
                continue
            try:
                content = f.read_text()
                if f.suffix == ".json":
                    data = json.loads(content)
                    if isinstance(data, list):
                        texts.extend([str(s) for s in data if len(str(s)) > 20])
                elif f.suffix == ".jsonl":
                    for line in content.split('\n'):
                        if line.strip():
                            try:
                                texts.append(str(json.loads(line)))
                            except: pass
                else:
                    # Try splitting by common separators
                    if "---" in content:
                        parts = [p.strip() for p in content.split("---") if len(p.strip()) > 20]
                        texts.extend(parts)
                    elif len(content.strip()) > 20:
                        texts.append(content.strip())
            except: pass
    
    # Strategy 2: Look for .pt files with token tensors
    if not texts:
        import torch
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        for pt_file in output_dir.rglob("*.pt"):
            try:
                data = torch.load(pt_file, map_location="cpu", weights_only=False)
                if isinstance(data, torch.Tensor):
                    if data.dim() == 2:
                        for i in range(data.shape[0]):
                            t = tokenizer.decode(data[i].tolist(), skip_special_tokens=True)
                            if len(t) > 20:
                                texts.append(t)
                    elif data.dim() == 1:
                        t = tokenizer.decode(data.tolist(), skip_special_tokens=True)
                        if len(t) > 20:
                            texts.append(t)
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 1:
                            if v.dim() == 2:
                                for i in range(v.shape[0]):
                                    t = tokenizer.decode(v[i].tolist(), skip_special_tokens=True)
                                    if len(t) > 20:
                                        texts.append(t)
            except Exception as e:
                print(f"      Could not load {pt_file}: {e}")
    
    # Strategy 3: Parse from stdout
    if not texts:
        stdout_file = output_dir / "stdout.txt"
        if stdout_file.exists():
            content = stdout_file.read_text()
            # MDLM often prints samples between markers or after "Generated:"
            # Try to find continuous blocks of text that look like samples
            lines = content.split('\n')
            current_block = []
            for line in lines:
                # Skip log/progress lines
                if any(x in line for x in ['Epoch', 'Step', 'Loss', 'Perplexity', 
                                            'Loading', 'Downloading', '━', '%|',
                                            'WARNING', 'INFO', 'DEBUG']):
                    if current_block and len(' '.join(current_block)) > 50:
                        texts.append(' '.join(current_block))
                    current_block = []
                elif line.strip():
                    current_block.append(line.strip())
            if current_block and len(' '.join(current_block)) > 50:
                texts.append(' '.join(current_block))
    
    return texts


def score_texts_ppl(texts, device="cuda"):
    """Score texts with GPT-2 Large for perplexity."""
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
    print("Phase 2 — MDLM: Stochastic vs Deterministic Sampling")
    print(f"  N={N}, steps={STEPS}, max_len={MAX_LEN}")
    print(f"  Using MDLM main.py via subprocess (avoids import issues)")
    print("=" * 70)
    t0 = time.time()
    
    # Conditions to test
    # T=0 → deterministic (MDLM default) — analog of ELF ODE
    # T>0 → stochastic — analog of ELF SDE
    temperatures = [0.0, 0.5, 1.0]
    
    conditions = []
    for temp in temperatures:
        label = f"T={temp}"
        exp_dir = OUT / f"run_T{temp}"
        
        print(f"\n  ── {label} ──")
        gt0 = time.time()
        
        # Generate
        stdout, stderr = generate_with_mdlm_main(temp, exp_dir)
        gen_time = time.time() - gt0
        print(f"    Generation took {gen_time:.0f}s")
        
        # Parse outputs  
        texts = parse_mdlm_outputs(exp_dir)
        print(f"    Parsed {len(texts)} texts")
        
        if not texts:
            print(f"    WARNING: No texts found! Check {exp_dir}/stdout.txt")
            print(f"    Stderr tail: {stderr[-300:]}")
            conditions.append({
                "label": label, "temperature": temp,
                "texts": [], "degs": [], "deg_rate": 1.0,
                "gen_time": gen_time, "n_texts": 0,
            })
            continue
        
        degs = [is_degenerate(t) for t in texts]
        deg_rate = sum(degs) / len(degs)
        print(f"    Degeneracy: {100*deg_rate:.1f}%")
        
        # Show first sample
        print(f"    Sample[0]: {texts[0][:150]}...")
        
        conditions.append({
            "label": label, "temperature": temp,
            "texts": texts, "degs": degs, "deg_rate": deg_rate,
            "gen_time": gen_time, "n_texts": len(texts),
        })
    
    # Score with GPT-2 Large
    print(f"\n  ── PPL Scoring ──")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for c in conditions:
        if not c["texts"]:
            c["mean_ppl"] = float('inf')
            c["n_valid"] = 0
            continue
        print(f"    Scoring {c['label']} ({len(c['texts'])} texts)...")
        c["ppls"] = score_texts_ppl(c["texts"][:512], device)
        valid = [p for p, d in zip(c["ppls"], c["degs"]) if not d and not math.isinf(p)]
        c["mean_ppl"] = float(np.mean(valid)) if valid else float('inf')
        c["mean_logppl"] = float(np.mean(np.log(valid))) if valid else float('inf')
        c["n_valid"] = len(valid)
        print(f"      PPL: {c['mean_ppl']:.1f} (N_valid={c['n_valid']})")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Condition':>12s}  {'PPL':>8s}  {'Deg':>6s}  {'N_texts':>8s}  {'N_valid':>8s}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c['label']:>12s}  {c['mean_ppl']:8.1f}  {100*c['deg_rate']:5.1f}%  {c['n_texts']:>8d}  {c['n_valid']:>8d}")
    
    # Save
    summary = {
        "model": "MDLM (kuleshov-group/mdlm-owt)",
        "architecture": "Masked Discrete Diffusion (DiT backbone)",
        "n_target_samples": N, "n_steps": STEPS,
        "conditions": [{
            "label": c["label"],
            "temperature": c["temperature"],
            "n_texts": c["n_texts"],
            "deg_rate": c["deg_rate"],
            "mean_ppl": c.get("mean_ppl", float('inf')),
            "mean_logppl": c.get("mean_logppl", float('inf')),
            "n_valid": c.get("n_valid", 0),
            "gen_time": c["gen_time"],
        } for c in conditions],
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "mdlm_stochastic.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(OUT / "mdlm_stochastic.md", "w") as f:
        f.write("# Phase 2 — MDLM: Stochastic vs Deterministic\n\n")
        f.write(f"**Model:** MDLM (kuleshov-group/mdlm-owt)\n\n")
        f.write("| Temperature | PPL ↓ | Degeneracy | N texts | N valid |\n")
        f.write("|---|---|---|---|---|\n")
        for c in conditions:
            f.write(f"| {c['temperature']} | {c.get('mean_ppl', 'inf'):.1f} | "
                    f"{100*c['deg_rate']:.1f}% | {c['n_texts']} | {c.get('n_valid', 0)} |\n")
    
    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
