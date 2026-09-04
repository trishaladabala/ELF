#!/usr/bin/env python3
"""Phase P4 — Diversity Analysis.

Generates a small batch of samples (N=128) for selected gamma values
to compute diversity metrics:
- Distinct-n (1, 2, 3)
- Zipf coefficient
"""
import json, sys, math, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from modules.model import ELF_models
from utils.sampling_utils import _sde_step, _ode_step, get_sampling_steps
from configs.config import Config

OUT = Path(__file__).resolve().parent / "results" / "p4_diversity"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 128; BS = 4; STEPS = 32; L = 1024; ENC = 512
GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SC_CFG = 3.0; NOISE_SCALE = 2.0

cfg = Config()
cfg.denoiser_p_mean = -1.5; cfg.denoiser_p_std = 0.8
cfg.denoiser_noise_scale = NOISE_SCALE; cfg.t_eps = 0.05
cfg.num_self_cond_cfg_tokens = 4; cfg.self_cond_prob = 0.5

COMMON = dict(text_encoder_dim=ENC, max_length=L, bottleneck_dim=128,
              num_time_tokens=4, num_self_cond_cfg_tokens=4,
              num_model_mode_tokens=4, vocab_size=32100)

def load_model():
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "embedded-language-flows/ELF-B-owt-torch"
    files = list_repo_files(repo)
    ckpts = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    path = hf_hub_download(repo, ckpts[0])
    model = ELF_models["ELF-B"](**COMMON)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_params1") or ckpt.get("params") or ckpt
    model.load_state_dict(sd, strict=False)
    return model.to(DEV, torch.float32).eval()

def compute_distinct_n(texts, n=2):
    from nltk import ngrams
    all_ngrams = set()
    total_ngrams = 0
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) < n: continue
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.update(text_ngrams)
        total_ngrams += len(text_ngrams)
    if total_ngrams == 0: return 0.0
    return len(all_ngrams) / total_ngrams

def compute_zipf(texts):
    from scipy.stats import linregress
    all_tokens = []
    for t in texts:
        all_tokens.extend(t.lower().split())
    if not all_tokens: return 0.0
    counts = sorted(Counter(all_tokens).values(), reverse=True)
    if len(counts) < 10: return 0.0
    ranks = np.arange(1, len(counts) + 1)
    # Fit log(count) = c - a * log(rank)
    slope, _, _, _, _ = linregress(np.log(ranks), np.log(counts))
    return -slope  # Natural language usually has Zipf ~ 1.0

def generate_with_gamma(model, gamma, all_noise, all_steps):
    n_batches = (N + BS - 1) // BS
    all_tokens = []

    for bi in range(n_batches):
        bs = min(BS, N - bi * BS)
        noise = all_noise[bi * BS : bi * BS + bs].clone()
        steps = all_steps[bi]

        z = noise.clone()
        cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
        mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
        x_prev = None

        if gamma == 0.0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
                for i in range(STEPS):
                    z, x_prev = _ode_step(model, z, steps[i].item(), steps[i+1].item(),
                                          x_prev, cfg, 1.0, SC_CFG, cond, mask)
        else:
            gen = torch.Generator(device=DEV)
            gen.manual_seed(7777 + bi)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
                for i in range(STEPS):
                    z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(),
                                          x_prev, cfg, 1.0, SC_CFG, cond, mask, gamma, gen)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
            t_f = torch.ones(bs, dtype=torch.float32, device=DEV)
            dec = torch.ones(bs, device=DEV)
            sc = torch.full((bs,), SC_CFG, dtype=torch.float32, device=DEV)
            z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
            _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
            tokens = logits.argmax(dim=-1).cpu()

        all_tokens.append(tokens)

    return torch.cat(all_tokens)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Phase P4 — Diversity Analysis at ELF-B (N=%d, L=%d)" % (N, L))
    print("  Gammas: %s" % GAMMAS)
    print("="*70)
    t0 = time.time()

    model = load_model()
    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    torch.manual_seed(42)
    all_noise = torch.randn(N, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE

    n_batches = (N + BS - 1) // BS
    all_steps = []
    for bi in range(n_batches):
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)
        all_steps.append(steps)

    results = []
    for gamma in GAMMAS:
        print(f"\n  ── γ = {gamma} ──")
        tokens = generate_with_gamma(model, gamma, all_noise, all_steps)
        texts = []
        for i in range(N):
            tids = tokens[i].tolist()
            texts.append(t5tok.decode(tids, skip_special_tokens=True))
        
        dist1 = compute_distinct_n(texts, 1)
        dist2 = compute_distinct_n(texts, 2)
        dist3 = compute_distinct_n(texts, 3)
        zipf = compute_zipf(texts)
        
        print(f"    Distinct-1: {dist1:.4f}")
        print(f"    Distinct-2: {dist2:.4f}")
        print(f"    Distinct-3: {dist3:.4f}")
        print(f"    Zipf coeff: {zipf:.4f}")
        
        results.append({
            "gamma": gamma,
            "distinct_1": dist1,
            "distinct_2": dist2,
            "distinct_3": dist3,
            "zipf": zipf
        })

    with open(OUT / "p4_diversity.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "p4_diversity.md", "w") as f:
        f.write("# Phase P4 — Diversity Analysis\n\n")
        f.write("| γ | Distinct-1 | Distinct-2 | Distinct-3 | Zipf |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['gamma']} | {r['distinct_1']:.4f} | {r['distinct_2']:.4f} | {r['distinct_3']:.4f} | {r['zipf']:.4f} |\n")

    print(f"\n  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
