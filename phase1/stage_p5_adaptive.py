#!/usr/bin/env python3
"""Phase P5 — Adaptive Noise Sampler at ELF-B (105M).

Implements a curvature-aware adaptive γ(t) sampler:
- Computes per-step trajectory curvature in real-time
- Modulates noise intensity: γ_t = γ_base × f(curvature_t)
- Compares three strategies:
  1. ODE (γ=0)
  2. Fixed SDE (γ=1.0, the best reliable fixed γ)
  3. Adaptive SDE (γ_t varies with curvature)

The hypothesis: by injecting less noise in high-curvature regions (where the
flow needs careful navigation) and more noise in low-curvature regions (where
regularization helps most), we can beat the fixed SDE.
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
from utils.sampling_utils import get_sampling_steps, net_out_to_v_x
from configs.config import Config

OUT = Path(__file__).resolve().parent / "results" / "p5_adaptive"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 256; BS = 4; STEPS = 32; L = 1024; ENC = 512
SC_CFG = 3.0; NOISE_SCALE = 2.0
T_EPS = 0.05; SIGMA = 1.0

# Adaptive sampler parameters
GAMMA_BASE = 1.0      # Base noise level
GAMMA_MIN = 0.1       # Minimum noise (high curvature → careful)
GAMMA_MAX = 2.0       # Maximum noise (low curvature → regularize)

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


def model_forward(model, z, t_scalar, sc_cfg):
    """Run model forward pass, return v and x_pred."""
    bs = z.shape[0]
    t_b = torch.full((bs,), t_scalar, dtype=torch.float32, device=DEV)
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    sc = torch.full((bs,), sc_cfg, dtype=torch.float32, device=DEV)

    z_uncond = torch.cat([z, torch.zeros_like(z)], dim=-1)
    z_cond = torch.cat([z, cond], dim=-1)
    z_in = torch.cat([z_uncond, z_cond], dim=0)
    t_in = t_b.repeat(2)
    sc_in = sc.repeat(2)
    dec_in = None

    net_out, _ = model(z_in, t_in, deterministic=True, self_cond_cfg_scale=sc_in, decoder_step_active=dec_in)
    out_uncond, out_cond = net_out[:bs], net_out[bs:]

    # CFG
    net_out_cfg = out_uncond + sc_cfg * (out_cond - out_uncond)
    v, x_pred = net_out_to_v_x(net_out_cfg, z, t_b, T_EPS)
    return v, x_pred


def generate_ode(model, noise, steps):
    """Pure ODE sampling."""
    z = noise.clone()
    bs = z.shape[0]
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            v, _ = model_forward(model, z, tc, SC_CFG)
            z = z + h * v
    return z


def generate_fixed_sde(model, noise, steps, gamma, gen):
    """Fixed-gamma SDE sampling."""
    z = noise.clone()
    bs = z.shape[0]
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            v, x_pred = model_forward(model, z, tc, SC_CFG)
            denom_s = max(1 - tc, T_EPS)**2 * SIGMA**2
            score = (tc * x_pred - z.float()) / denom_s
            drift = v + (gamma**2 / 2) * score
            w = torch.randn(z.shape, device=DEV, dtype=z.dtype, generator=gen)
            diffusion = gamma * math.sqrt(abs(h)) * w
            z = z + h * drift.to(z.dtype) + diffusion
    return z


def generate_adaptive_sde(model, noise, steps, gen):
    """Adaptive-gamma SDE sampling with curvature-modulated noise."""
    z = noise.clone()
    bs = z.shape[0]
    prev_v = None
    gamma_history = []

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            tc, tn = float(steps[i]), float(steps[i+1])
            h = tn - tc
            v, x_pred = model_forward(model, z, tc, SC_CFG)

            # Compute curvature from velocity change
            if prev_v is not None:
                # Cosine similarity between consecutive velocity vectors
                v_flat = v.float().reshape(bs, -1)
                pv_flat = prev_v.float().reshape(bs, -1)
                cos_sim = F.cosine_similarity(v_flat, pv_flat, dim=-1)  # [bs]
                # curvature ∈ [0, 2]: 0 = straight, 2 = reversal
                curvature = (1 - cos_sim).clamp(0, 2)  # [bs]

                # Adaptive gamma: inversely proportional to curvature
                # High curvature → low noise (careful navigation)
                # Low curvature → high noise (regularize)
                # gamma_t = GAMMA_MAX * exp(-alpha * curvature)
                alpha = 3.0  # Controls sensitivity
                gamma_t = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * torch.exp(-alpha * curvature)
                gamma_t = gamma_t.reshape(-1, 1, 1)  # [bs, 1, 1]
            else:
                # First step: use base gamma
                gamma_t = torch.full((bs, 1, 1), GAMMA_BASE, device=DEV)
                curvature = torch.zeros(bs, device=DEV)

            gamma_history.append(float(gamma_t.mean()))
            prev_v = v.clone()

            # SDE step with per-sample gamma
            denom_s = max(1 - tc, T_EPS)**2 * SIGMA**2
            score = (tc * x_pred - z.float()) / denom_s
            drift = v + (gamma_t.to(v.dtype)**2 / 2) * score.to(v.dtype)
            w = torch.randn(z.shape, device=DEV, dtype=z.dtype, generator=gen)
            diffusion = gamma_t.to(z.dtype) * math.sqrt(abs(h)) * w
            z = z + h * drift.to(z.dtype) + diffusion

    return z, gamma_history


def decode_tokens(model, z):
    """Decode latent z to token IDs."""
    bs = z.shape[0]
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        t_f = torch.ones(bs, dtype=torch.float32, device=DEV)
        dec = torch.ones(bs, device=DEV)
        sc = torch.full((bs,), SC_CFG, dtype=torch.float32, device=DEV)
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
        tokens = logits.argmax(dim=-1).cpu()
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(-1).mean(-1).cpu()
    return tokens, entropy


def score_texts(texts, device):
    """Score texts with GPT-2 Large for PPL."""
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
    print("Phase P5 — Adaptive Noise Sampler at ELF-B")
    print(f"  N={N}, L={L}, {STEPS} steps")
    print(f"  Adaptive: γ_base={GAMMA_BASE}, γ_min={GAMMA_MIN}, γ_max={GAMMA_MAX}")
    print("=" * 70)
    t0 = time.time()

    model = load_model()
    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    # Pre-generate noise and steps
    torch.manual_seed(42)
    all_noise = torch.randn(N, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE

    n_batches = (N + BS - 1) // BS
    all_steps = []
    for bi in range(n_batches):
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)
        all_steps.append(steps)

    # ── Generate with three strategies ──
    strategies = {}

    for strategy_name, strategy_fn in [
        ("ODE (γ=0)", lambda noise, steps, bi: generate_ode(model, noise, steps)),
        ("Fixed SDE (γ=1.0)", lambda noise, steps, bi: generate_fixed_sde(
            model, noise, steps, 1.0,
            torch.Generator(device=DEV).manual_seed(7777 + bi))),
        ("Adaptive SDE", lambda noise, steps, bi: generate_adaptive_sde(
            model, noise, steps,
            torch.Generator(device=DEV).manual_seed(7777 + bi))),
    ]:
        print(f"\n  ── {strategy_name} ──")
        gt0 = time.time()
        all_tokens = []; all_entropy = []; gamma_hist = None

        for bi in range(n_batches):
            bs = min(BS, N - bi * BS)
            noise = all_noise[bi * BS : bi * BS + bs].clone()
            steps = all_steps[bi]

            result = strategy_fn(noise, steps, bi)
            if isinstance(result, tuple):
                z, gh = result
                if gamma_hist is None: gamma_hist = []
                gamma_hist.extend(gh)
            else:
                z = result

            tokens, entropy = decode_tokens(model, z)
            all_tokens.append(tokens)
            all_entropy.append(entropy)

        tokens = torch.cat(all_tokens)
        entropies = torch.cat(all_entropy)

        texts = []; degs = []
        for i in range(N):
            tids = tokens[i].tolist()
            texts.append(t5tok.decode(tids, skip_special_tokens=True))
            degs.append(is_degenerate(tids))

        degs = np.array(degs)
        gen_time = time.time() - gt0
        print(f"    Degeneracy: {100*degs.mean():.1f}% ({int(degs.sum())}/{N})")
        print(f"    Mean entropy: {float(entropies.mean()):.3f}")
        print(f"    Gen time: {gen_time:.1f}s")
        if gamma_hist:
            print(f"    Mean adaptive γ: {np.mean(gamma_hist):.3f} ± {np.std(gamma_hist):.3f}")

        strategies[strategy_name] = {
            "texts": texts, "degs": degs,
            "deg_rate": float(degs.mean()),
            "mean_entropy": float(entropies.mean()),
            "gen_time": gen_time,
            "gamma_history": gamma_hist,
        }

    # ── Score all with GPT-2 Large ──
    print(f"\n  ── PPL Scoring ──")
    del model; torch.cuda.empty_cache()

    for name, s in strategies.items():
        print(f"    Scoring {name}...")
        s["ppls"] = score_texts(s["texts"], DEV)
        valid_ppls = [p for p, d in zip(s["ppls"], s["degs"]) if not d and not math.isinf(p)]
        s["mean_ppl"] = float(np.mean(valid_ppls)) if valid_ppls else float('inf')
        s["mean_logppl"] = float(np.mean(np.log(valid_ppls))) if valid_ppls else float('inf')
        s["n_valid"] = len(valid_ppls)
        print(f"      Mean PPL: {s['mean_ppl']:.1f} (N_valid={s['n_valid']})")

    # ── Paired comparisons ──
    ode = strategies["ODE (γ=0)"]
    for name, s in strategies.items():
        if "ODE" in name: continue
        joint_ok = (~ode["degs"]) & (~s["degs"])
        n_joint = int(joint_ok.sum())
        if n_joint > 10:
            ode_lp = np.log(np.array(ode["ppls"]))[joint_ok]
            sde_lp = np.log(np.array(s["ppls"]))[joint_ok]
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

        s["vs_ode_diff"] = diff
        s["vs_ode_ci"] = ci
        sign = "+" if diff > 0 else "-"
        print(f"    {name} vs ODE: logPPL diff = {sign}{abs(diff):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
              + (" ← beats ODE" if ci[0] > 0 else ""))

    # Adaptive vs Fixed comparison
    fixed = strategies["Fixed SDE (γ=1.0)"]
    adaptive = strategies["Adaptive SDE"]
    joint_ok = (~fixed["degs"]) & (~adaptive["degs"])
    n_joint = int(joint_ok.sum())
    if n_joint > 10:
        f_lp = np.log(np.array(fixed["ppls"]))[joint_ok]
        a_lp = np.log(np.array(adaptive["ppls"]))[joint_ok]
        valid = np.isfinite(f_lp) & np.isfinite(a_lp)
        if valid.sum() > 10:
            diff_fa = float(np.mean(f_lp[valid] - a_lp[valid]))
            np.random.seed(42)
            boots = [np.mean(np.random.choice(f_lp[valid] - a_lp[valid], int(valid.sum()), replace=True))
                     for _ in range(5000)]
            ci_fa = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        else:
            diff_fa = float('nan'); ci_fa = [float('nan'), float('nan')]
    else:
        diff_fa = float('nan'); ci_fa = [float('nan'), float('nan')]

    print(f"\n  ★ Adaptive vs Fixed SDE: logPPL diff = {diff_fa:+.3f} [{ci_fa[0]:.3f}, {ci_fa[1]:.3f}]")
    if not math.isnan(ci_fa[0]) and ci_fa[0] > 0:
        print("    → ADAPTIVE BEATS FIXED SDE!")
    elif not math.isnan(ci_fa[1]) and ci_fa[1] < 0:
        print("    → Fixed SDE is better (adaptive doesn't help)")
    else:
        print("    → No significant difference")

    # Save results
    summary = {
        "n_samples": N, "max_length": L, "n_steps": STEPS,
        "adaptive_params": {"gamma_base": GAMMA_BASE, "gamma_min": GAMMA_MIN, "gamma_max": GAMMA_MAX},
        "strategies": {
            name: {
                "deg_rate": s["deg_rate"],
                "mean_ppl": s["mean_ppl"],
                "mean_logppl": s["mean_logppl"],
                "mean_entropy": s["mean_entropy"],
                "n_valid": s["n_valid"],
                "gen_time": s["gen_time"],
                "vs_ode_diff": s.get("vs_ode_diff"),
                "vs_ode_ci": s.get("vs_ode_ci"),
                "mean_gamma": float(np.mean(s["gamma_history"])) if s.get("gamma_history") else None,
            }
            for name, s in strategies.items()
        },
        "adaptive_vs_fixed": {"diff": diff_fa, "ci_95": ci_fa},
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "p5_adaptive.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    with open(OUT / "p5_adaptive.md", "w") as f:
        f.write("# Phase P5 — Adaptive Noise Sampler\n\n")
        f.write(f"**Config:** N={N}, L={L}, {STEPS} steps\n\n")
        f.write("| Strategy | Degeneracy | Mean PPL | Mean log-PPL | vs ODE | 95% CI |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, s in strategies.items():
            d = s.get("vs_ode_diff", "-")
            ci = s.get("vs_ode_ci", ["-", "-"])
            if isinstance(d, float) and not math.isnan(d):
                f.write(f"| {name} | {100*s['deg_rate']:.1f}% | {s['mean_ppl']:.1f} | "
                        f"{s['mean_logppl']:.3f} | {d:+.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |\n")
            else:
                f.write(f"| {name} | {100*s['deg_rate']:.1f}% | {s['mean_ppl']:.1f} | "
                        f"{s['mean_logppl']:.3f} | — | — |\n")
        f.write(f"\n**Adaptive vs Fixed SDE:** {diff_fa:+.3f} [{ci_fa[0]:.3f}, {ci_fa[1]:.3f}]\n")
        f.write(f"\n**Total time:** {time.time()-t0:.0f}s\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
