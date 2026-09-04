#!/usr/bin/env python3
"""Phase P5 (FIXED) — Adaptive Noise Sampler using the official SDE formulation.

Fixes the critical bug in the original P5: now uses the exact same "churn" SDE
as _sde_step (alpha-blending with noise, then ODE forward), not a Langevin SDE.

Baselines use the official _ode_step and _sde_step directly to ensure consistency
with P1 results.

Also logs per-step γ values to verify the sampler is genuinely adapting.
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
from utils.sampling_utils import (
    _ode_step, _sde_step, _forward_sample,
    get_sampling_steps, net_out_to_v_x, restore_cond
)
from configs.config import Config

OUT = Path(__file__).resolve().parent / "results" / "p5_fixed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 512; BS = 4; STEPS = 32; L = 1024; ENC = 512
SC_CFG = 3.0; NOISE_SCALE = 2.0; T_EPS = 0.05

GAMMA_FIXED = 1.0       # Fixed SDE baseline
GAMMA_MIN = 0.1
GAMMA_MAX = 3.0          # Wider range to allow real adaptation
ALPHA_CURV = 2.0         # Curvature sensitivity

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


def _adaptive_sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma_per_sample, generator,
):
    """Adaptive SDE step using the OFFICIAL churn formulation but with per-sample gamma.

    This is identical to _sde_step except gamma is a [bs] tensor, not a scalar.
    The formulation: alpha = clamp(1 - gamma * h), z_back = alpha*z + (1-alpha)*eps,
    then ODE step from z_back.
    """
    bs = z.shape[0]
    h = float(t_next - t)

    # Per-sample alpha: [bs, 1, 1] for broadcasting
    gamma_expanded = gamma_per_sample.reshape(-1, 1, 1).to(z.dtype)
    alpha = (1.0 - gamma_expanded * h).clamp(0.0, 1.0)

    t_back_per_sample = alpha.squeeze(-1).squeeze(-1) * float(t)  # [bs]

    eps = torch.randn(z.shape, generator=generator, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    z_back = alpha * z + (1.0 - alpha) * eps
    z_back = restore_cond(z_back, cond_seq, cond_seq_mask)

    # Use mean t_back for the model forward (the model expects a single t per sample)
    t_batch = t_back_per_sample.to(z.dtype)
    v_pred, x_pred = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    # ODE step from z_back at t_back to t_next
    t_next_tensor = torch.full((bs,), float(t_next), dtype=z.dtype, device=z.device)
    dt = (t_next_tensor - t_batch).reshape(-1, 1, 1)
    z_next = z_back + dt * v_pred

    return z_next, x_pred


def generate_ode(model, noise, steps):
    """ODE using official _ode_step — identical to P1."""
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            z, x_prev = _ode_step(model, z, steps[i].item(), steps[i+1].item(),
                                  x_prev, cfg, 1.0, SC_CFG, cond, mask)
    return z, None


def generate_fixed_sde(model, noise, steps, gamma, gen):
    """Fixed SDE using official _sde_step — identical to P1."""
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(),
                                  x_prev, cfg, 1.0, SC_CFG, cond, mask, gamma, gen)
    return z, None


def generate_adaptive_sde(model, noise, steps, gen):
    """Adaptive SDE using the CORRECT churn formulation with curvature-modulated gamma."""
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    prev_v = None
    step_gammas = []     # Per-step mean gamma
    step_gamma_stds = [] # Per-step std of gamma across samples

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            tc, tn = steps[i].item(), steps[i+1].item()

            # First, get velocity for curvature computation (using ODE-style forward)
            t_batch = torch.full((bs,), tc, dtype=torch.float32, device=DEV)
            v_pred, x_pred_cur = _forward_sample(
                model=model, z=z, t_batch=t_batch, x_pred_prev=x_prev,
                config=cfg, cfg_scale=1.0, self_cond_cfg_scale=SC_CFG,
                cond_seq=cond, cond_seq_mask=mask,
            )

            # Compute curvature
            if prev_v is not None:
                v_flat = v_pred.float().reshape(bs, -1)
                pv_flat = prev_v.float().reshape(bs, -1)
                cos_sim = F.cosine_similarity(v_flat, pv_flat, dim=-1).clamp(-1, 1)
                curvature = (1 - cos_sim).clamp(0, 2)  # [bs]

                # Adaptive gamma: high curvature → less noise, low curvature → more noise
                gamma_per_sample = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * torch.exp(-ALPHA_CURV * curvature)
            else:
                gamma_per_sample = torch.full((bs,), GAMMA_FIXED, device=DEV)
                curvature = torch.zeros(bs, device=DEV)

            step_gammas.append(float(gamma_per_sample.mean()))
            step_gamma_stds.append(float(gamma_per_sample.std()))
            prev_v = v_pred.clone()

            # Now do the actual churn SDE step with per-sample gamma
            z, x_prev = _adaptive_sde_step(
                model, z, tc, tn, x_prev,
                cfg, 1.0, SC_CFG, cond, mask, gamma_per_sample, gen,
            )

    gamma_info = {
        "per_step_mean": step_gammas,
        "per_step_std": step_gamma_stds,
        "overall_mean": float(np.mean(step_gammas)),
        "overall_std": float(np.std(step_gammas)),
        "mean_within_step_std": float(np.mean(step_gamma_stds)),
    }
    return z, gamma_info


def decode(model, z):
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
    print("Phase P5 (FIXED) — Adaptive Sampler with Official SDE Formulation")
    print(f"  N={N}, L={L}, {STEPS} steps")
    print(f"  Fixed γ={GAMMA_FIXED}, Adaptive γ∈[{GAMMA_MIN}, {GAMMA_MAX}], α={ALPHA_CURV}")
    print("=" * 70)
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

    strategy_fns = [
        ("ODE (γ=0)", lambda noise, steps, bi: generate_ode(model, noise, steps)),
        ("Fixed SDE (γ=1.0)", lambda noise, steps, bi: generate_fixed_sde(
            model, noise, steps, GAMMA_FIXED,
            torch.Generator(device=DEV).manual_seed(7777 + bi))),
        ("Fixed SDE (γ=1.5)", lambda noise, steps, bi: generate_fixed_sde(
            model, noise, steps, 1.5,
            torch.Generator(device=DEV).manual_seed(7777 + bi))),
        ("Adaptive SDE", lambda noise, steps, bi: generate_adaptive_sde(
            model, noise, steps,
            torch.Generator(device=DEV).manual_seed(7777 + bi))),
    ]

    strategies = {}
    for strategy_name, strategy_fn in strategy_fns:
        print(f"\n  ── {strategy_name} ──")
        gt0 = time.time()
        all_tokens = []; all_entropy = []; gamma_info = None

        for bi in range(n_batches):
            bs = min(BS, N - bi * BS)
            noise = all_noise[bi * BS : bi * BS + bs].clone()
            steps = all_steps[bi]

            z, info = strategy_fn(noise, steps, bi)
            if info is not None and gamma_info is None:
                gamma_info = info

            tokens, entropy = decode(model, z)
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
        if gamma_info:
            print(f"    Mean γ: {gamma_info['overall_mean']:.3f} ± {gamma_info['overall_std']:.3f}")
            print(f"    Mean within-step γ std: {gamma_info['mean_within_step_std']:.4f}")
            print(f"    Per-step γ: {[f'{g:.2f}' for g in gamma_info['per_step_mean']]}")

        strategies[strategy_name] = {
            "texts": texts, "degs": degs,
            "deg_rate": float(degs.mean()),
            "mean_entropy": float(entropies.mean()),
            "gen_time": gen_time,
            "gamma_info": gamma_info,
        }

    # Score with GPT-2 Large
    print(f"\n  ── PPL Scoring ──")
    del model; torch.cuda.empty_cache()

    for name, s in strategies.items():
        print(f"    Scoring {name}...")
        s["ppls"] = score_texts(s["texts"], DEV)
        valid = [p for p, d in zip(s["ppls"], s["degs"]) if not d and not math.isinf(p)]
        s["mean_ppl"] = float(np.mean(valid)) if valid else float('inf')
        s["mean_logppl"] = float(np.mean(np.log(valid))) if valid else float('inf')
        s["n_valid"] = len(valid)
        print(f"      Mean PPL: {s['mean_ppl']:.1f} (N_valid={s['n_valid']})")

    # Paired comparisons
    ode = strategies["ODE (γ=0)"]
    comparisons = {}
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

        s["vs_ode_diff"] = diff; s["vs_ode_ci"] = ci
        comparisons[name] = {"diff": diff, "ci": ci, "n_joint": n_joint}
        sign = "+" if diff > 0 else "-"
        better = "← beats ODE" if (not math.isnan(ci[0]) and ci[0] > 0) else ""
        print(f"    {name} vs ODE: {sign}{abs(diff):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {better}")

    # Adaptive vs Fixed(1.0) and Fixed(1.5)
    for fixed_name in ["Fixed SDE (γ=1.0)", "Fixed SDE (γ=1.5)"]:
        fixed = strategies[fixed_name]
        adaptive = strategies["Adaptive SDE"]
        joint_ok = (~fixed["degs"]) & (~adaptive["degs"])
        n_joint = int(joint_ok.sum())
        if n_joint > 10:
            f_lp = np.log(np.array(fixed["ppls"]))[joint_ok]
            a_lp = np.log(np.array(adaptive["ppls"]))[joint_ok]
            valid = np.isfinite(f_lp) & np.isfinite(a_lp)
            if valid.sum() > 10:
                diff = float(np.mean(f_lp[valid] - a_lp[valid]))
                np.random.seed(42)
                boots = [np.mean(np.random.choice(f_lp[valid] - a_lp[valid], int(valid.sum()), replace=True))
                         for _ in range(5000)]
                ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            else:
                diff = float('nan'); ci = [float('nan'), float('nan')]
        else:
            diff = float('nan'); ci = [float('nan'), float('nan')]
        comparisons[f"Adaptive vs {fixed_name}"] = {"diff": diff, "ci": ci}
        sign = "+" if diff > 0 else "-"
        verdict = "ADAPTIVE BETTER" if (not math.isnan(ci[0]) and ci[0] > 0) else \
                  "FIXED BETTER" if (not math.isnan(ci[1]) and ci[1] < 0) else "NO DIFFERENCE"
        print(f"\n  ★ Adaptive vs {fixed_name}: {sign}{abs(diff):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] → {verdict}")

    # Save
    summary = {
        "n_samples": N, "n_steps": STEPS, "pipeline": "official _ode_step/_sde_step",
        "strategies": {
            name: {
                "deg_rate": s["deg_rate"], "mean_ppl": s["mean_ppl"],
                "mean_logppl": s["mean_logppl"], "mean_entropy": s["mean_entropy"],
                "n_valid": s["n_valid"], "gen_time": s["gen_time"],
                "vs_ode_diff": s.get("vs_ode_diff"), "vs_ode_ci": s.get("vs_ode_ci"),
                "gamma_info": s.get("gamma_info"),
            } for name, s in strategies.items()
        },
        "comparisons": comparisons,
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "p5_fixed.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown
    with open(OUT / "p5_fixed.md", "w") as f:
        f.write("# Phase P5 (FIXED) — Adaptive Sampler with Official SDE\n\n")
        f.write(f"**Pipeline:** Official `_ode_step`/`_sde_step` (matches P1)\n")
        f.write(f"**Config:** N={N}, L={L}, {STEPS} steps\n\n")
        f.write("## Results\n\n")
        f.write("| Strategy | Deg | Mean PPL | log-PPL | vs ODE | 95% CI |\n")
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
        f.write(f"\n## Comparisons\n\n")
        for name, c in comparisons.items():
            f.write(f"- **{name}:** {c['diff']:+.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]\n")
        if strategies["Adaptive SDE"].get("gamma_info"):
            gi = strategies["Adaptive SDE"]["gamma_info"]
            f.write(f"\n## Adaptive γ Analysis\n\n")
            f.write(f"- Overall mean γ: {gi['overall_mean']:.3f} ± {gi['overall_std']:.3f}\n")
            f.write(f"- Mean within-step γ std: {gi['mean_within_step_std']:.4f}\n")
            f.write(f"- Per-step γ: {[round(g, 2) for g in gi['per_step_mean']]}\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
