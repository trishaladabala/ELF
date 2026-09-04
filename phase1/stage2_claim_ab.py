#!/usr/bin/env python3
"""Stage 2 — Claims A+B: ODE vs SDE Degeneracy & Quality Penalty.

Also computes per-token curvature and entropy for Claims C and D.
"""
import json, sys, math, time, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from modules.model import ELF_models
from utils.sampling_utils import _sde_step, _ode_step, get_sampling_steps
from configs.config import Config

OUT = Path(__file__).resolve().parent / "results" / "stage2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 1024; BS = 4; STEPS = 32; L = 1024; ENC = 512
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
    repo = "embedded-language-flows/ELF-B-owt-torch"
    files = list_repo_files(repo)
    ckpt_files = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    path = hf_hub_download(repo, ckpt_files[0]) if ckpt_files else None
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
        if ng:
            mc = Counter(ng).most_common(1)[0][1]
            if mc / len(ng) > 0.3: return True
    return False

def ode_generate(model, noise, steps):
    """ODE sampling with on-the-fly curvature computation."""
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    curv_sum = torch.zeros(bs, L, device="cpu")
    curv_n = 0
    z_prev = None; z_prev2 = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            t_c, t_n = steps[i].item(), steps[i+1].item()
            z, x_prev = _ode_step(model, z, t_c, t_n, x_prev, cfg, 1.0, SC_CFG, cond, mask)
            # Incremental curvature
            if z_prev is not None and z_prev2 is not None:
                v1 = z_prev - z_prev2; v2 = z - z_prev
                cs = F.cosine_similarity(v1, v2, dim=-1).clamp(-1,1)
                curv_sum += torch.arccos(cs).float().cpu()
                curv_n += 1
            z_prev2 = z_prev; z_prev = z.clone()
    per_tok = curv_sum / max(curv_n, 1)
    return z, per_tok

def sde_generate(model, noise, steps, gen):
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            t_c, t_n = steps[i].item(), steps[i+1].item()
            z, x_prev = _sde_step(model, z, t_c, t_n, x_prev, cfg, 1.0, SC_CFG, cond, mask, GAMMA, gen)
    return z

def decode_with_entropy(model, z):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        t_f = torch.ones(z.shape[0], dtype=torch.float32, device=DEV)
        dec = torch.ones(z.shape[0], device=DEV)
        sc = torch.full((z.shape[0],), SC_CFG, dtype=torch.float32, device=DEV)
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
        tokens = logits.argmax(dim=-1).cpu()
        probs = F.softmax(logits.float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).cpu()
    return tokens, ent

def score_texts(texts, device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).half().eval()
    ppls = []
    for t in texts:
        if not t.strip():
            ppls.append(float('inf')); continue
        enc = tok(t, return_tensors="pt", truncation=True, max_length=1024)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 2:
            ppls.append(float('inf')); continue
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=device=='cuda'):
            loss = gpt(ids, labels=ids).loss
        ppls.append(math.exp(loss.float().item()))
    del gpt; torch.cuda.empty_cache()
    return ppls

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    w = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c - w) / d, (c + w) / d)

def bootstrap_ci(a, b, n_boot=10000):
    diffs = np.array(a) - np.array(b)
    boots = [np.mean(np.random.choice(diffs, len(diffs), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Stage 2 — Claims A+B: ODE vs SDE (N=%d, L=%d, %d steps)" % (N, L, STEPS))
    print("="*70)
    t0 = time.time()

    model = load_model()
    print(f"  Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    # Storage
    ode_texts, sde_texts = [], []
    ode_degs, sde_degs = [], []
    all_curv = []       # [N, L] per-token curvature
    all_ode_ent = []    # [N, L]
    all_sde_ent = []    # [N, L]

    n_batches = (N + BS - 1) // BS
    gen_t0 = time.time()

    for bi in range(n_batches):
        bs = min(BS, N - bi * BS)
        if (bi+1) % 32 == 0 or bi == 0:
            print(f"  Batch {bi+1}/{n_batches} (sample {bi*BS+1}-{bi*BS+bs})...")

        # Same noise for ODE and SDE
        torch.manual_seed(42 + bi)
        noise = torch.randn(bs, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE

        # Same time steps for fair comparison
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)

        # ODE generation + curvature
        z_ode, per_tok_curv = ode_generate(model, noise, steps)
        all_curv.append(per_tok_curv.numpy())

        # SDE generation
        sde_gen = torch.Generator(device=DEV)
        sde_gen.manual_seed(7777 + bi)
        z_sde = sde_generate(model, noise, steps, sde_gen)

        # Decode + entropy
        ode_toks, ode_ent = decode_with_entropy(model, z_ode)
        sde_toks, sde_ent = decode_with_entropy(model, z_sde)
        all_ode_ent.append(ode_ent.numpy())
        all_sde_ent.append(sde_ent.numpy())

        # Texts + degeneracy
        for i in range(bs):
            ot = ode_toks[i].tolist(); st = sde_toks[i].tolist()
            ode_texts.append(t5tok.decode(ot, skip_special_tokens=True))
            sde_texts.append(t5tok.decode(st, skip_special_tokens=True))
            ode_degs.append(is_degenerate(ot))
            sde_degs.append(is_degenerate(st))

    gen_dt = time.time() - gen_t0
    print(f"\n  Generation done: {gen_dt:.1f}s ({gen_dt/N:.2f}s/sample)")

    # Concatenate arrays
    curv_arr = np.concatenate(all_curv, axis=0)     # [N, L]
    ode_ent_arr = np.concatenate(all_ode_ent, axis=0)  # [N, L]
    sde_ent_arr = np.concatenate(all_sde_ent, axis=0)  # [N, L]
    ode_degs = np.array(ode_degs)
    sde_degs = np.array(sde_degs)

    # Save metrics for Stages 3+4
    np.savez_compressed(OUT / "stage2_metrics.npz",
                        curvature=curv_arr, ode_entropy=ode_ent_arr,
                        sde_entropy=sde_ent_arr, ode_degs=ode_degs, sde_degs=sde_degs,
                        per_sample_curvature=np.mean(curv_arr, axis=1))
    print(f"  Saved metrics: {OUT / 'stage2_metrics.npz'}")

    # ── Claim A: ODE Degeneracy Rate ──
    ode_deg_rate = float(ode_degs.mean())
    sde_deg_rate = float(sde_degs.mean())
    ode_ci = wilson_ci(int(ode_degs.sum()), N)
    sde_ci = wilson_ci(int(sde_degs.sum()), N)
    print(f"\n  ── Claim A: ODE Degeneracy ──")
    print(f"  ODE: {100*ode_deg_rate:.1f}% [{100*ode_ci[0]:.1f}%, {100*ode_ci[1]:.1f}%]")
    print(f"  SDE: {100*sde_deg_rate:.1f}% [{100*sde_ci[0]:.1f}%, {100*sde_ci[1]:.1f}%]")
    claim_a = ode_deg_rate > 0.01  # > 1% bar
    print(f"  Claim A {'REPLICATES' if claim_a else 'DOES NOT REPLICATE'} (bar: >1%)")

    # ── Claim B: SDE Quality Penalty (PPL scoring) ──
    print(f"\n  ── Scoring with GPT-2 Large... ──")
    del model; torch.cuda.empty_cache()
    ode_ppls = score_texts(ode_texts, DEV)
    sde_ppls = score_texts(sde_texts, DEV)

    # Jointly non-degenerate subset
    joint_ok = (~ode_degs) & (~sde_degs)
    n_joint = int(joint_ok.sum())
    print(f"  Jointly non-degenerate: {n_joint}/{N}")

    if n_joint > 10:
        ode_lp = np.log(np.array(ode_ppls))[joint_ok]
        sde_lp = np.log(np.array(sde_ppls))[joint_ok]
        diff = float(np.mean(ode_lp - sde_lp))  # negative = SDE worse
        ci_lo, ci_hi = bootstrap_ci(ode_lp, sde_lp)
        print(f"  Mean log-PPL diff (ODE-SDE): {diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        claim_b = ci_hi < 0  # CI strictly negative = SDE worse
        print(f"  Claim B {'REPLICATES' if claim_b else 'DOES NOT REPLICATE'}")
    else:
        diff = ci_lo = ci_hi = float('nan')
        claim_b = False
        print("  Claim B: insufficient jointly non-degenerate samples")

    # ── Save all results ──
    results = {
        "n_samples": N, "max_length": L, "n_steps": STEPS,
        "claim_a": {
            "ode_deg_rate": ode_deg_rate, "ode_deg_ci": list(ode_ci),
            "sde_deg_rate": sde_deg_rate, "sde_deg_ci": list(sde_ci),
            "ode_deg_count": int(ode_degs.sum()), "sde_deg_count": int(sde_degs.sum()),
            "replicates": claim_a,
        },
        "claim_b": {
            "n_joint_nondegen": n_joint,
            "mean_logppl_diff": diff, "ci": [ci_lo, ci_hi],
            "replicates": claim_b,
        },
        "timing": {"gen_s": gen_dt, "total_s": time.time() - t0},
    }

    with open(OUT / "claim_ab_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save texts
    with open(OUT / "ode_outputs.jsonl", "w") as f:
        for i, t in enumerate(ode_texts):
            json.dump({"id": i, "text": t[:500], "deg": bool(ode_degs[i]),
                        "ppl": ode_ppls[i] if not math.isinf(ode_ppls[i]) else None}, f)
            f.write("\n")
    with open(OUT / "sde_outputs.jsonl", "w") as f:
        for i, t in enumerate(sde_texts):
            json.dump({"id": i, "text": t[:500], "deg": bool(sde_degs[i]),
                        "ppl": sde_ppls[i] if not math.isinf(sde_ppls[i]) else None}, f)
            f.write("\n")

    # Verdict report
    with open(OUT / "claim_ab_verdict.md", "w") as f:
        f.write("# Stage 2 — Claims A+B Verdict\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Config:** N={N}, L={L}, {STEPS} steps, γ={GAMMA}\n\n")
        f.write("## Claim A: ODE Degeneracy\n\n")
        f.write(f"| Sampler | Degeneracy | 95% CI |\n|---|---|---|\n")
        f.write(f"| ODE | {100*ode_deg_rate:.1f}% ({int(ode_degs.sum())}/{N}) | [{100*ode_ci[0]:.1f}%, {100*ode_ci[1]:.1f}%] |\n")
        f.write(f"| SDE | {100*sde_deg_rate:.1f}% ({int(sde_degs.sum())}/{N}) | [{100*sde_ci[0]:.1f}%, {100*sde_ci[1]:.1f}%] |\n\n")
        f.write(f"**Verdict:** {'✅ REPLICATES' if claim_a else '❌ DOES NOT REPLICATE'} (bar: >1%)\n\n")
        f.write("## Claim B: SDE Quality Penalty\n\n")
        f.write(f"- Jointly non-degenerate: {n_joint}/{N}\n")
        f.write(f"- Mean log-PPL diff (ODE−SDE): {diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]\n")
        f.write(f"- **Verdict:** {'✅ REPLICATES' if claim_b else '❌ DOES NOT REPLICATE'}\n\n")
        f.write(f"**Total time:** {time.time()-t0:.0f}s\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
