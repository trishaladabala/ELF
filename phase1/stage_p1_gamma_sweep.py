#!/usr/bin/env python3
"""Phase P1 — Gamma Sweep at ELF-B (105M).

Runs unconditional generation with γ ∈ {0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
using identical initial noise across all γ values. Measures PPL, degeneracy,
and mean token entropy for each.
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

OUT = Path(__file__).resolve().parent / "results" / "p1_gamma"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 512; BS = 4; STEPS = 32; L = 1024; ENC = 512
GAMMAS = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
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

def is_degenerate(tids):
    ids = [int(t) for t in tids if int(t) != 0]
    if len(ids) < 10: return True
    if len(set(ids)) <= 5: return True
    if len(ids) >= 5:
        ng = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
        if ng and Counter(ng).most_common(1)[0][1] / len(ng) > 0.3: return True
    return False

def generate_with_gamma(model, gamma, all_noise, all_steps):
    """Generate N samples with given gamma, using pre-generated noise and time steps."""
    n_batches = (N + BS - 1) // BS
    all_tokens = []
    all_entropy = []

    for bi in range(n_batches):
        bs = min(BS, N - bi * BS)
        noise = all_noise[bi * BS : bi * BS + bs].clone()
        steps = all_steps[bi]

        z = noise.clone()
        cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
        mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
        x_prev = None

        if gamma == 0.0:
            # Pure ODE
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
                for i in range(STEPS):
                    z, x_prev = _ode_step(model, z, steps[i].item(), steps[i+1].item(),
                                          x_prev, cfg, 1.0, SC_CFG, cond, mask)
        else:
            # SDE with given gamma
            gen = torch.Generator(device=DEV)
            gen.manual_seed(7777 + bi)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
                for i in range(STEPS):
                    z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(),
                                          x_prev, cfg, 1.0, SC_CFG, cond, mask, gamma, gen)

        # Decode
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
            t_f = torch.ones(bs, dtype=torch.float32, device=DEV)
            dec = torch.ones(bs, device=DEV)
            sc = torch.full((bs,), SC_CFG, dtype=torch.float32, device=DEV)
            z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
            _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
            tokens = logits.argmax(dim=-1).cpu()
            probs = F.softmax(logits.float(), dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean(dim=-1).cpu()

        all_tokens.append(tokens)
        all_entropy.append(ent)

    return torch.cat(all_tokens), torch.cat(all_entropy)

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
    print("="*70)
    print("Phase P1 — Gamma Sweep at ELF-B (N=%d, L=%d)" % (N, L))
    print("  Gammas: %s" % GAMMAS)
    print("="*70)
    t0 = time.time()

    model = load_model()
    print(f"  Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    # Pre-generate all initial noise (same for all γ values)
    print("  Pre-generating noise and time steps...")
    torch.manual_seed(42)
    all_noise = torch.randn(N, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE

    n_batches = (N + BS - 1) // BS
    all_steps = []
    for bi in range(n_batches):
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)
        all_steps.append(steps)

    # Run each gamma
    results = []
    for gamma in GAMMAS:
        print(f"\n  ── γ = {gamma} ──")
        gt0 = time.time()

        tokens, entropies = generate_with_gamma(model, gamma, all_noise, all_steps)

        # Decode + degeneracy
        texts = []; degs = []
        for i in range(N):
            tids = tokens[i].tolist()
            texts.append(t5tok.decode(tids, skip_special_tokens=True))
            degs.append(is_degenerate(tids))

        degs = np.array(degs)
        deg_rate = float(degs.mean())
        mean_ent = float(entropies.mean())
        gen_time = time.time() - gt0

        print(f"    Degeneracy: {100*deg_rate:.1f}% ({int(degs.sum())}/{N})")
        print(f"    Mean entropy: {mean_ent:.3f}")
        print(f"    Gen time: {gen_time:.1f}s")

        results.append({
            "gamma": gamma, "texts": texts, "degs": degs,
            "deg_rate": deg_rate, "mean_entropy": mean_ent, "gen_time": gen_time,
        })

    # Score all with GPT-2 Large
    print(f"\n  ── PPL Scoring with GPT-2 Large ──")
    del model; torch.cuda.empty_cache()

    for r in results:
        print(f"    Scoring γ={r['gamma']}...")
        r["ppls"] = score_texts(r["texts"], DEV)
        nondeg = ~r["degs"]
        valid_ppls = [p for p, d in zip(r["ppls"], r["degs"]) if not d and not math.isinf(p)]
        r["mean_ppl"] = float(np.mean(valid_ppls)) if valid_ppls else float('inf')
        r["median_ppl"] = float(np.median(valid_ppls)) if valid_ppls else float('inf')
        r["mean_logppl"] = float(np.mean(np.log(valid_ppls))) if valid_ppls else float('inf')
        r["n_valid"] = len(valid_ppls)
        print(f"      Mean PPL: {r['mean_ppl']:.1f} (N_valid={r['n_valid']})")

    # Find optimal gamma
    valid_results = [r for r in results if r["n_valid"] > 10]
    if valid_results:
        best = min(valid_results, key=lambda r: r["mean_logppl"])
        gamma_star = best["gamma"]
        print(f"\n  ★ Optimal γ* = {gamma_star} (mean log-PPL = {best['mean_logppl']:.3f})")
    else:
        gamma_star = None
        print("\n  ⚠️ Not enough valid samples to determine γ*")

    # Paired comparison: ODE (γ=0) vs each γ
    ode_r = next(r for r in results if r["gamma"] == 0.0)
    comparisons = []
    for r in results:
        if r["gamma"] == 0.0: continue
        # Paired bootstrap on jointly non-degenerate
        joint_ok = (~ode_r["degs"]) & (~r["degs"])
        n_joint = int(joint_ok.sum())
        if n_joint > 10:
            ode_lp = np.log(np.array(ode_r["ppls"]))[joint_ok]
            sde_lp = np.log(np.array(r["ppls"]))[joint_ok]
            # Filter inf/nan
            valid = np.isfinite(ode_lp) & np.isfinite(sde_lp)
            if valid.sum() > 10:
                diff = float(np.mean(ode_lp[valid] - sde_lp[valid]))
                np.random.seed(42)
                boots = [np.mean(np.random.choice(ode_lp[valid] - sde_lp[valid], int(valid.sum()), replace=True))
                         for _ in range(5000)]
                ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            else:
                diff = float('nan'); ci = [float('nan'), float('nan')]
        else:
            diff = float('nan'); ci = [float('nan'), float('nan')]
            n_joint = 0

        comp = {"gamma": r["gamma"], "n_joint": n_joint,
                "logppl_diff_ode_minus_sde": diff, "ci_95": ci,
                "sde_better": ci[0] > 0 if not math.isnan(ci[0]) else None}
        comparisons.append(comp)
        sign = "+" if diff > 0 else "-"
        print(f"    γ={r['gamma']}: ODE−SDE logPPL = {sign}{abs(diff):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
              + (" ← SDE better" if comp.get("sde_better") else ""))

    # Save results
    summary = {
        "n_samples": N, "max_length": L, "n_steps": STEPS,
        "gamma_star": gamma_star,
        "sweep": [{
            "gamma": r["gamma"], "deg_rate": r["deg_rate"],
            "mean_ppl": r["mean_ppl"], "median_ppl": r["median_ppl"],
            "mean_logppl": r["mean_logppl"], "mean_entropy": r["mean_entropy"],
            "n_valid": r["n_valid"], "gen_time": r["gen_time"],
        } for r in results],
        "comparisons": comparisons,
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "p1_gamma_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    with open(OUT / "p1_gamma_sweep.md", "w") as f:
        f.write("# Phase P1 — Gamma Sweep at ELF-B\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Config:** N={N}, L={L}, {STEPS} steps\n\n")
        f.write("## PPL vs γ\n\n")
        f.write("| γ | Degeneracy | Mean PPL | Median PPL | Mean log-PPL | Entropy | N_valid |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            star = " ★" if r["gamma"] == gamma_star else ""
            f.write(f"| {r['gamma']}{star} | {100*r['deg_rate']:.1f}% | "
                    f"{r['mean_ppl']:.1f} | {r['median_ppl']:.1f} | "
                    f"{r['mean_logppl']:.3f} | {r['mean_entropy']:.3f} | {r['n_valid']} |\n")
        f.write(f"\n**Optimal γ* = {gamma_star}**\n\n")
        f.write("## Paired Comparisons (ODE vs SDE)\n\n")
        f.write("| γ | log-PPL diff (ODE−SDE) | 95% CI | SDE better? |\n")
        f.write("|---|---|---|---|\n")
        for c in comparisons:
            f.write(f"| {c['gamma']} | {c['logppl_diff_ode_minus_sde']:.3f} | "
                    f"[{c['ci_95'][0]:.3f}, {c['ci_95'][1]:.3f}] | "
                    f"{'✅' if c.get('sde_better') else '❌'} |\n")
        f.write(f"\n**Total time:** {time.time()-t0:.0f}s\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
