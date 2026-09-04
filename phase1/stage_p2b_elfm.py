#!/usr/bin/env python3
"""Phase P2b — ELF-M (350M) Quality Comparison.

Runs ODE vs SDE (γ=1.5) on ELF-M to add a data point between 5M and 105M
on the scale crossover curve. Uses the same pipeline as Stage 2.
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

OUT = Path(__file__).resolve().parent / "results" / "p2b_elfm"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ELF-M uses 64 sampling steps (per the paper)
N = 256; BS = 2; STEPS = 64; L = 1024; ENC = 512
GAMMA = 1.5; SC_CFG = 3.0; NOISE_SCALE = 2.0

cfg = Config()
cfg.denoiser_p_mean = -1.5; cfg.denoiser_p_std = 0.8
cfg.denoiser_noise_scale = NOISE_SCALE; cfg.t_eps = 0.05
cfg.num_self_cond_cfg_tokens = 4; cfg.self_cond_prob = 0.5

COMMON = dict(text_encoder_dim=ENC, max_length=L, bottleneck_dim=128,
              num_time_tokens=4, num_self_cond_cfg_tokens=4,
              num_model_mode_tokens=4, vocab_size=32100)

def load_model():
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "embedded-language-flows/ELF-M-owt-torch"
    files = list_repo_files(repo)
    ckpt_files = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    if not ckpt_files:
        print(f"  Available files in {repo}: {files}")
        raise RuntimeError(f"No checkpoint found in {repo}")
    path = hf_hub_download(repo, ckpt_files[0])
    model = ELF_models["ELF-M"](**COMMON)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_params1") or ckpt.get("params") or ckpt
    model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded ELF-M: {n_params:,} params")
    return model.to(DEV, torch.float32).eval()

def is_degenerate(tids):
    ids = [int(t) for t in tids if int(t) != 0]
    if len(ids) < 10: return True
    if len(set(ids)) <= 5: return True
    if len(ids) >= 5:
        ng = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
        if ng and Counter(ng).most_common(1)[0][1] / len(ng) > 0.3: return True
    return False

def ode_generate(model, noise, steps):
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            z, x_prev = _ode_step(model, z, steps[i].item(), steps[i+1].item(),
                                  x_prev, cfg, 1.0, SC_CFG, cond, mask)
    return z

def sde_generate(model, noise, steps, gen):
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(),
                                  x_prev, cfg, 1.0, SC_CFG, cond, mask, GAMMA, gen)
    return z

def decode(model, z):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        t_f = torch.ones(z.shape[0], dtype=torch.float32, device=DEV)
        dec = torch.ones(z.shape[0], device=DEV)
        sc = torch.full((z.shape[0],), SC_CFG, dtype=torch.float32, device=DEV)
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
        tokens = logits.argmax(dim=-1).cpu()
    return tokens

def score_texts(texts, device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).half().eval()
    ppls = []
    for t in texts:
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
    print("Phase P2b — ELF-M (350M) Quality Crossover Point")
    print(f"  N={N}, L={L}, {STEPS} steps, γ={GAMMA}")
    print("=" * 70)
    t0 = time.time()

    model = load_model()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    ode_texts, sde_texts = [], []
    ode_degs, sde_degs = [], []
    n_batches = (N + BS - 1) // BS
    gen_t0 = time.time()

    for bi in range(n_batches):
        bs = min(BS, N - bi * BS)
        if (bi + 1) % 16 == 0 or bi == 0:
            print(f"  Batch {bi+1}/{n_batches}...")

        torch.manual_seed(42 + bi)
        noise = torch.randn(bs, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)

        z_ode = ode_generate(model, noise, steps)

        sde_gen = torch.Generator(device=DEV)
        sde_gen.manual_seed(7777 + bi)
        z_sde = sde_generate(model, noise, steps, sde_gen)

        ode_toks = decode(model, z_ode)
        sde_toks = decode(model, z_sde)

        for i in range(bs):
            ot = ode_toks[i].tolist(); st = sde_toks[i].tolist()
            ode_texts.append(t5tok.decode(ot, skip_special_tokens=True))
            sde_texts.append(t5tok.decode(st, skip_special_tokens=True))
            ode_degs.append(is_degenerate(ot))
            sde_degs.append(is_degenerate(st))

    gen_dt = time.time() - gen_t0
    ode_degs = np.array(ode_degs)
    sde_degs = np.array(sde_degs)
    print(f"\n  Generation done: {gen_dt:.0f}s")
    print(f"  ODE degeneracy: {100*ode_degs.mean():.1f}%")
    print(f"  SDE degeneracy: {100*sde_degs.mean():.1f}%")

    # Score with GPT-2 Large
    print(f"\n  Scoring with GPT-2 Large...")
    del model; torch.cuda.empty_cache()
    ode_ppls = score_texts(ode_texts, DEV)
    sde_ppls = score_texts(sde_texts, DEV)

    # Paired comparison
    joint_ok = (~ode_degs) & (~sde_degs)
    n_joint = int(joint_ok.sum())
    print(f"  Jointly non-degenerate: {n_joint}/{N}")

    if n_joint > 10:
        ode_lp = np.log(np.array(ode_ppls))[joint_ok]
        sde_lp = np.log(np.array(sde_ppls))[joint_ok]
        valid = np.isfinite(ode_lp) & np.isfinite(sde_lp)
        diff = float(np.mean(ode_lp[valid] - sde_lp[valid]))
        np.random.seed(42)
        boots = [np.mean(np.random.choice(ode_lp[valid] - sde_lp[valid], int(valid.sum()), replace=True))
                 for _ in range(5000)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    else:
        diff = float('nan'); ci = [float('nan'), float('nan')]

    ode_valid = [p for p, d in zip(ode_ppls, ode_degs) if not d and not math.isinf(p)]
    sde_valid = [p for p, d in zip(sde_ppls, sde_degs) if not d and not math.isinf(p)]

    sde_better = bool(ci[0] > 0) if not math.isnan(ci[0]) else None
    sign = "+" if diff > 0 else "-"
    print(f"\n  ODE mean PPL: {np.mean(ode_valid):.1f}")
    print(f"  SDE mean PPL: {np.mean(sde_valid):.1f}")
    print(f"  log-PPL diff (ODE−SDE): {sign}{abs(diff):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  SDE better: {sde_better}")

    n_params = 342 * 1000000  # approximate
    results = {
        "model": "ELF-M", "params_approx": n_params,
        "n_samples": N, "n_steps": STEPS, "gamma": GAMMA,
        "ode_deg_rate": float(ode_degs.mean()),
        "sde_deg_rate": float(sde_degs.mean()),
        "ode_mean_ppl": float(np.mean(ode_valid)) if ode_valid else None,
        "sde_mean_ppl": float(np.mean(sde_valid)) if sde_valid else None,
        "logppl_diff_ode_minus_sde": diff,
        "ci_95": ci,
        "sde_better": sde_better,
        "n_joint_nondegen": n_joint,
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "p2b_elfm.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "p2b_elfm.md", "w") as f:
        f.write("# Phase P2b — ELF-M (350M) Crossover Point\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| ODE degeneracy | {100*ode_degs.mean():.1f}% |\n")
        f.write(f"| SDE degeneracy | {100*sde_degs.mean():.1f}% |\n")
        f.write(f"| ODE mean PPL | {np.mean(ode_valid):.1f} |\n")
        f.write(f"| SDE mean PPL | {np.mean(sde_valid):.1f} |\n")
        f.write(f"| log-PPL diff (ODE−SDE) | {diff:+.4f} |\n")
        f.write(f"| 95% CI | [{ci[0]:.4f}, {ci[1]:.4f}] |\n")
        f.write(f"| SDE better? | {'✅' if sde_better else '❌' if sde_better is False else '⚠️'} |\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
