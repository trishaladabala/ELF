#!/usr/bin/env python3
"""Phase P5b — Adaptive Sampler Sensitivity Analysis.

Tests the curvature sensitivity parameter α ∈ {1.0, 2.0, 3.0, 5.0, 8.0}
to prove the adaptive sampler result is robust and not tuned to one value.
Uses the same pipeline as P5 but with _ode_step/_sde_step for consistency.
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
from utils.sampling_utils import _ode_step, get_sampling_steps, net_out_to_v_x
from configs.config import Config

OUT = Path(__file__).resolve().parent / "results" / "p5b_sensitivity"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 256; BS = 4; STEPS = 32; L = 1024; ENC = 512
SC_CFG = 3.0; NOISE_SCALE = 2.0; T_EPS = 0.05; SIGMA = 1.0

GAMMA_MIN = 0.1; GAMMA_MAX = 2.0
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

cfg = Config()
cfg.denoiser_p_mean = -1.5; cfg.denoiser_p_std = 0.8
cfg.denoiser_noise_scale = NOISE_SCALE; cfg.t_eps = T_EPS
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

def is_degenerate(tids):
    ids = [int(t) for t in tids if int(t) != 0]
    if len(ids) < 10: return True
    if len(set(ids)) <= 5: return True
    if len(ids) >= 5:
        ng = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
        if ng and Counter(ng).most_common(1)[0][1] / len(ng) > 0.3: return True
    return False

def adaptive_sde_generate(model, noise, steps, alpha, gen):
    """Adaptive SDE with given alpha parameter."""
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask_t = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    prev_v = None
    gamma_history = []

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            tc = steps[i].item()
            tn = steps[i+1].item()
            h = tn - tc

            # Get model prediction (same as _ode_step internals)
            t_b = torch.full((bs,), tc, dtype=torch.float32, device=DEV)
            sc = torch.full((bs,), SC_CFG, dtype=torch.float32, device=DEV)

            # Unconditioned + conditioned forward (CFG)
            z_uncond = torch.cat([z, torch.zeros(bs, L, ENC, dtype=z.dtype, device=DEV)], dim=-1)
            z_cond = torch.cat([z, cond.to(z.dtype)], dim=-1)
            z_in = torch.cat([z_uncond, z_cond], dim=0)
            t_in = t_b.repeat(2)
            sc_in = sc.repeat(2)

            net_out, _ = model(z_in, t_in, deterministic=True, self_cond_cfg_scale=sc_in, decoder_step_active=None)
            out_uncond, out_cond = net_out[:bs], net_out[bs:]
            net_combined = out_uncond + SC_CFG * (out_cond - out_uncond)

            v, x_pred = net_out_to_v_x(net_combined, z.float(), t_b, T_EPS)

            # Compute curvature from velocity change
            if prev_v is not None:
                v_flat = v.float().reshape(bs, -1)
                pv_flat = prev_v.float().reshape(bs, -1)
                cos_sim = F.cosine_similarity(v_flat, pv_flat, dim=-1).clamp(-1, 1)
                curvature = (1 - cos_sim).clamp(0, 2)
                gamma_t = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * torch.exp(-alpha * curvature)
                gamma_t = gamma_t.reshape(-1, 1, 1)
            else:
                gamma_t = torch.full((bs, 1, 1), 1.0, device=DEV)

            gamma_history.append(float(gamma_t.mean()))
            prev_v = v.clone()

            # SDE step
            denom_s = max(1 - tc, T_EPS)**2 * SIGMA**2
            score = (tc * x_pred.float() - z.float()) / denom_s
            drift = v.float() + (gamma_t**2 / 2) * score
            w = torch.randn(z.shape, device=DEV, dtype=z.dtype, generator=gen)
            diffusion = gamma_t.to(z.dtype) * math.sqrt(abs(h)) * w
            z = (z + h * drift.to(z.dtype) + diffusion)

            # Self-conditioning
            x_prev = x_pred

    return z, gamma_history

def decode(model, z):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        t_f = torch.ones(z.shape[0], dtype=torch.float32, device=DEV)
        dec = torch.ones(z.shape[0], device=DEV)
        sc = torch.full((z.shape[0],), SC_CFG, dtype=torch.float32, device=DEV)
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
        return logits.argmax(dim=-1).cpu()

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
    print("Phase P5b — Adaptive Sampler Sensitivity (α sweep)")
    print(f"  N={N}, L={L}, {STEPS} steps")
    print(f"  Alphas: {ALPHAS}")
    print(f"  γ_min={GAMMA_MIN}, γ_max={GAMMA_MAX}")
    print("=" * 70)
    t0 = time.time()

    model = load_model()
    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    # Pre-generate noise
    torch.manual_seed(42)
    all_noise = torch.randn(N, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE
    n_batches = (N + BS - 1) // BS
    all_steps = []
    for bi in range(n_batches):
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)
        all_steps.append(steps)

    results = []
    for alpha in ALPHAS:
        print(f"\n  ── α = {alpha} ──")
        gt0 = time.time()
        all_tokens = []; all_gamma_hist = []

        for bi in range(n_batches):
            bs = min(BS, N - bi * BS)
            noise = all_noise[bi * BS : bi * BS + bs].clone()
            steps = all_steps[bi]
            gen = torch.Generator(device=DEV)
            gen.manual_seed(7777 + bi)
            z, gh = adaptive_sde_generate(model, noise, steps, alpha, gen)
            all_gamma_hist.extend(gh)
            tokens = decode(model, z)
            all_tokens.append(tokens)

        tokens = torch.cat(all_tokens)
        texts = []; degs = []
        for i in range(N):
            tids = tokens[i].tolist()
            texts.append(t5tok.decode(tids, skip_special_tokens=True))
            degs.append(is_degenerate(tids))

        degs = np.array(degs)
        mean_gamma = float(np.mean(all_gamma_hist))
        gen_time = time.time() - gt0

        print(f"    Deg: {100*degs.mean():.1f}%, Mean γ: {mean_gamma:.3f}, Time: {gen_time:.0f}s")

        results.append({
            "alpha": alpha, "texts": texts, "degs": degs,
            "deg_rate": float(degs.mean()), "mean_gamma": mean_gamma,
            "gen_time": gen_time,
        })

    # Score with GPT-2 Large
    print(f"\n  ── PPL Scoring ──")
    del model; torch.cuda.empty_cache()

    for r in results:
        print(f"    Scoring α={r['alpha']}...")
        r["ppls"] = score_texts(r["texts"], DEV)
        valid = [p for p, d in zip(r["ppls"], r["degs"]) if not d and not math.isinf(p)]
        r["mean_ppl"] = float(np.mean(valid)) if valid else float('inf')
        r["mean_logppl"] = float(np.mean(np.log(valid))) if valid else float('inf')
        r["n_valid"] = len(valid)
        print(f"      PPL: {r['mean_ppl']:.1f} (N_valid={r['n_valid']})")

    # Summary
    print(f"\n  {'α':>6s}  {'Mean γ':>8s}  {'Deg%':>6s}  {'PPL':>8s}  {'logPPL':>8s}")
    for r in results:
        print(f"  {r['alpha']:6.1f}  {r['mean_gamma']:8.3f}  {100*r['deg_rate']:5.1f}%  {r['mean_ppl']:8.1f}  {r['mean_logppl']:8.3f}")

    best = min(results, key=lambda r: r["mean_logppl"])
    print(f"\n  ★ Best α = {best['alpha']} (PPL = {best['mean_ppl']:.1f})")

    # Save
    summary = {
        "n_samples": N, "gamma_min": GAMMA_MIN, "gamma_max": GAMMA_MAX,
        "best_alpha": best["alpha"],
        "sweep": [{
            "alpha": r["alpha"], "deg_rate": r["deg_rate"],
            "mean_ppl": r["mean_ppl"], "mean_logppl": r["mean_logppl"],
            "mean_gamma": r["mean_gamma"], "n_valid": r["n_valid"],
        } for r in results],
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "p5b_sensitivity.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "p5b_sensitivity.md", "w") as f:
        f.write("# Phase P5b — Adaptive Sampler Sensitivity\n\n")
        f.write("| α | Mean γ | Degeneracy | Mean PPL | Mean log-PPL |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            star = " ★" if r["alpha"] == best["alpha"] else ""
            f.write(f"| {r['alpha']}{star} | {r['mean_gamma']:.3f} | "
                    f"{100*r['deg_rate']:.1f}% | {r['mean_ppl']:.1f} | {r['mean_logppl']:.3f} |\n")
        f.write(f"\n**Best α = {best['alpha']}** (PPL = {best['mean_ppl']:.1f})\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
