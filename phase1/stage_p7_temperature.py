#!/usr/bin/env python3
"""Phase P7 — Temperature-Calibrated Decoding vs SDE Noise.

Core experiment for the paper's algorithmic contribution.

Hypothesis: Temperature scaling at decode time achieves a better PPL-MAUVE
Pareto frontier than SDE noise in latent space, at zero extra compute cost.

Generates N samples with ODE (γ=0), then decodes at different temperatures.
Also generates with SDE at different γ values for comparison.
Measures both PPL and MAUVE for each condition.

This directly addresses the PPL-MAUVE disagreement found in P6.
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

OUT = Path(__file__).resolve().parent / "results" / "p7_temperature"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 512; BS = 4; STEPS = 32; L = 1024; ENC = 512
SC_CFG = 3.0; NOISE_SCALE = 2.0

TEMPERATURES = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
SDE_GAMMAS = [0.5, 1.0, 1.5]

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


def generate_ode(model, noise, steps):
    """ODE generation using official _ode_step. Returns final z vectors."""
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


def generate_sde(model, noise, steps, gamma, gen):
    """SDE generation using official _sde_step."""
    z = noise.clone()
    bs = z.shape[0]
    cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
    mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
    x_prev = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        for i in range(STEPS):
            z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(),
                                  x_prev, cfg, 1.0, SC_CFG, cond, mask, gamma, gen)
    return z


def decode_with_temperature(model, z, temperature=1.0, top_p=0.9, use_sampling=True):
    """Decode latent z to tokens using temperature-scaled nucleus sampling.
    
    IMPORTANT: argmax is temperature-invariant (argmax(logits/T) = argmax(logits)).
    Temperature only matters when SAMPLING from the distribution.
    We use nucleus (top-p) sampling so temperature has a real effect.
    
    temperature < 1.0 → sharper (more confident, less diverse)
    temperature = 1.0 → standard nucleus sampling
    temperature > 1.0 → softer (less confident, more diverse)
    """
    bs = z.shape[0]
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        t_f = torch.ones(bs, dtype=torch.float32, device=DEV)
        dec = torch.ones(bs, device=DEV)
        sc = torch.full((bs,), SC_CFG, dtype=torch.float32, device=DEV)
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)

        # Apply temperature
        scaled_logits = logits.float() / temperature

        if use_sampling:
            # Nucleus (top-p) sampling per position
            # Shape: [bs, seq_len, vocab_size]
            probs = F.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens with cumulative probability above top_p
            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Sample from filtered distribution
            # Reshape for multinomial: [bs * seq_len, vocab_size]
            bs_seq = bs * scaled_logits.shape[1]
            flat_probs = sorted_probs.reshape(bs_seq, -1)
            sampled_sorted_idx = torch.multinomial(flat_probs, 1).squeeze(-1)  # [bs*seq]
            # Map back to original indices
            flat_sorted_indices = sorted_indices.reshape(bs_seq, -1)
            tokens = flat_sorted_indices[torch.arange(bs_seq, device=DEV), sampled_sorted_idx]
            tokens = tokens.reshape(bs, -1).cpu()
        else:
            # Argmax (temperature has no effect here)
            tokens = scaled_logits.argmax(dim=-1).cpu()

        # Measure entropy
        probs_for_entropy = F.softmax(scaled_logits, dim=-1)
        entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-12)).sum(-1).mean(-1).cpu()

    return tokens, entropy


def score_texts_ppl(texts, device):
    """Score texts with GPT-2 Large for perplexity."""
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


def score_texts_mauve(generated_texts, reference_texts):
    """Compute MAUVE score."""
    import mauve
    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        device_id=0,
        max_text_length=256,
        verbose=False,
        featurize_model_name="gpt2-large",
    )
    return float(result.mauve)


def load_reference_texts(n=500):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = []
    for row in ds:
        t = row["text"].strip()
        if len(t) > 100:
            words = t.split()[:256]
            texts.append(" ".join(words))
            if len(texts) >= n:
                break
    return texts


def run_condition(model, t5tok, all_noise, all_steps, label, gen_fn, decode_fn):
    """Generate + decode + collect texts for one experimental condition."""
    n_batches = (N + BS - 1) // BS
    all_tokens = []; all_entropy = []

    for bi in range(n_batches):
        bs = min(BS, N - bi * BS)
        noise = all_noise[bi * BS : bi * BS + bs].clone()
        steps = all_steps[bi]

        z = gen_fn(noise, steps, bi)
        tokens, entropy = decode_fn(z)
        all_tokens.append(tokens)
        all_entropy.append(entropy)

    tokens = torch.cat(all_tokens)
    entropies = torch.cat(all_entropy)

    texts = []; degs = []
    for i in range(N):
        tids = tokens[i].tolist()
        texts.append(t5tok.decode(tids, skip_special_tokens=True))
        degs.append(is_degenerate(tids))

    return texts, np.array(degs), float(entropies.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Phase P7 — Temperature-Calibrated Decoding vs SDE Noise")
    print(f"  N={N}, {STEPS} steps")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  SDE gammas: {SDE_GAMMAS}")
    print("=" * 70)
    t0 = time.time()

    model = load_model()
    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    # Pre-generate noise and steps (same for all conditions)
    torch.manual_seed(42)
    all_noise = torch.randn(N, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE
    n_batches = (N + BS - 1) // BS
    all_steps = []
    for bi in range(n_batches):
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)
        all_steps.append(steps)

    # ── Phase A: Generate ODE z-vectors once ──
    print("\n  ── Generating ODE z-vectors ──")
    gt0 = time.time()
    ode_z_batches = []
    for bi in range(n_batches):
        bs = min(BS, N - bi * BS)
        noise = all_noise[bi * BS : bi * BS + bs].clone()
        steps = all_steps[bi]
        z = generate_ode(model, noise, steps)
        ode_z_batches.append(z)
    print(f"    Done ({time.time()-gt0:.0f}s)")

    # ── Phase B0: ODE + Argmax baseline (matches P1 exactly) ──
    print(f"\n  ── ODE + Argmax (baseline) ──")
    conditions = []
    all_tokens = []; all_entropy = []
    for z_batch in ode_z_batches:
        tokens, entropy = decode_with_temperature(model, z_batch, temperature=1.0, use_sampling=False)
        all_tokens.append(tokens)
        all_entropy.append(entropy)
    tokens_base = torch.cat(all_tokens)
    entropies_base = torch.cat(all_entropy)
    texts_b = []; degs_b = []
    for i in range(N):
        tids = tokens_base[i].tolist()
        texts_b.append(t5tok.decode(tids, skip_special_tokens=True))
        degs_b.append(is_degenerate(tids))
    degs_b = np.array(degs_b)
    print(f"    Deg: {100*degs_b.mean():.1f}%, Entropy: {float(entropies_base.mean()):.3f}")
    conditions.append({
        "label": "ODE+argmax", "method": "argmax",
        "gamma": 0.0, "temperature": 1.0,
        "texts": texts_b, "degs": degs_b, "mean_entropy": float(entropies_base.mean()),
    })

    # ── Phase B: Decode ODE at different temperatures (nucleus sampling) ──
    for temp in TEMPERATURES:
        print(f"\n  ── ODE + T={temp} (nucleus p=0.9) ──")
        torch.manual_seed(9999)  # Reproducible sampling
        all_tokens = []; all_entropy = []
        for z_batch in ode_z_batches:
            tokens, entropy = decode_with_temperature(model, z_batch, temperature=temp)
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
        mean_ent = float(entropies.mean())
        print(f"    Deg: {100*degs.mean():.1f}%, Entropy: {mean_ent:.3f}")
        conditions.append({
            "label": f"ODE+T={temp}", "method": "temperature_nucleus",
            "gamma": 0.0, "temperature": temp,
            "texts": texts, "degs": degs, "mean_entropy": mean_ent,
        })

    # ── Phase C: Generate SDE at different gammas ──
    for gamma in SDE_GAMMAS:
        print(f"\n  ── SDE γ={gamma} + T=1.0 ──")
        all_tokens = []; all_entropy = []
        for bi in range(n_batches):
            bs = min(BS, N - bi * BS)
            noise = all_noise[bi * BS : bi * BS + bs].clone()
            steps = all_steps[bi]
            gen = torch.Generator(device=DEV)
            gen.manual_seed(7777 + bi)
            z = generate_sde(model, noise, steps, gamma, gen)
            # SDE uses argmax decode (same as P1) for fair comparison
            tokens, entropy = decode_with_temperature(model, z, temperature=1.0, use_sampling=False)
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
        mean_ent = float(entropies.mean())
        print(f"    Deg: {100*degs.mean():.1f}%, Entropy: {mean_ent:.3f}")
        conditions.append({
            "label": f"SDE γ={gamma}", "method": "sde",
            "gamma": gamma, "temperature": 1.0,
            "texts": texts, "degs": degs, "mean_entropy": mean_ent,
        })

    # ── Phase D: Score everything ──
    print(f"\n  ── PPL Scoring ──")
    del model; torch.cuda.empty_cache()

    for c in conditions:
        print(f"    Scoring {c['label']}...")
        c["ppls"] = score_texts_ppl(c["texts"], DEV)
        valid = [p for p, d in zip(c["ppls"], c["degs"]) if not d and not math.isinf(p)]
        c["mean_ppl"] = float(np.mean(valid)) if valid else float('inf')
        c["mean_logppl"] = float(np.mean(np.log(valid))) if valid else float('inf')
        c["n_valid"] = len(valid)
        print(f"      PPL: {c['mean_ppl']:.1f} (N_valid={c['n_valid']})")

    print(f"\n  ── MAUVE Scoring ──")
    ref_texts = load_reference_texts(n=500)
    for c in conditions:
        print(f"    MAUVE for {c['label']}...")
        # Use first 500 non-degenerate texts
        valid_texts = [t for t, d in zip(c["texts"], c["degs"]) if not d][:500]
        if len(valid_texts) >= 100:
            c["mauve"] = score_texts_mauve(valid_texts, ref_texts)
        else:
            c["mauve"] = 0.0
        print(f"      MAUVE: {c['mauve']:.4f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  {'Condition':>20s}  {'PPL':>8s}  {'MAUVE':>8s}  {'Entropy':>8s}  {'Deg':>6s}")
    print(f"{'='*70}")
    for c in conditions:
        print(f"  {c['label']:>20s}  {c['mean_ppl']:8.1f}  {c['mauve']:8.4f}  {c['mean_entropy']:8.3f}  {100*c['degs'].mean():5.1f}%")

    # Find Pareto-optimal conditions (best MAUVE at each PPL level)
    # A condition is Pareto-dominated if another has both better PPL AND better MAUVE
    pareto = []
    for c in conditions:
        dominated = False
        for c2 in conditions:
            if c2 is c: continue
            if c2["mean_ppl"] <= c["mean_ppl"] and c2["mauve"] >= c["mauve"]:
                if c2["mean_ppl"] < c["mean_ppl"] or c2["mauve"] > c["mauve"]:
                    dominated = True; break
        if not dominated:
            pareto.append(c["label"])
    print(f"\n  Pareto-optimal: {pareto}")

    # Save
    summary = {
        "n_samples": N, "n_steps": STEPS,
        "conditions": [{
            "label": c["label"], "method": c["method"],
            "gamma": c["gamma"], "temperature": c["temperature"],
            "deg_rate": float(c["degs"].mean()),
            "mean_ppl": c["mean_ppl"], "mean_logppl": c["mean_logppl"],
            "mauve": c["mauve"], "mean_entropy": c["mean_entropy"],
            "n_valid": c["n_valid"],
        } for c in conditions],
        "pareto_optimal": pareto,
        "total_time_s": time.time() - t0,
    }
    with open(OUT / "p7_temperature.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "p7_temperature.md", "w") as f:
        f.write("# Phase P7 — Temperature vs SDE Noise (Pareto Frontier)\n\n")
        f.write("| Condition | PPL ↓ | MAUVE ↑ | Entropy | Deg | Pareto? |\n")
        f.write("|---|---|---|---|---|---|\n")
        for c in conditions:
            p = "✅" if c["label"] in pareto else ""
            f.write(f"| {c['label']} | {c['mean_ppl']:.1f} | {c['mauve']:.4f} | "
                    f"{c['mean_entropy']:.3f} | {100*c['degs'].mean():.1f}% | {p} |\n")
        f.write(f"\n**Pareto-optimal:** {', '.join(pareto)}\n")
        f.write(f"\n**Total time:** {time.time()-t0:.0f}s\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
