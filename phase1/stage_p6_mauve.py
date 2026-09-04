#!/usr/bin/env python3
"""Phase P6 — MAUVE Score (distribution-level quality metric).

Computes MAUVE between generated text and real text (WikiText-103) for
each γ value from P1. MAUVE measures how close the generated distribution
is to the real data distribution using KL-divergence frontiers.

This addresses the "single metric" weakness — PPL alone is insufficient.
"""
import json, sys, math, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "p6_mauve"
P1_DIR = Path(__file__).resolve().parent / "results" / "p1_gamma"


def load_reference_texts(n=500, max_length=256):
    """Load real text from WikiText-103 as reference corpus."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = []
    for row in ds:
        t = row["text"].strip()
        if len(t) > 100:  # Skip short fragments
            # Truncate to max_length words for fair comparison
            words = t.split()[:max_length]
            texts.append(" ".join(words))
            if len(texts) >= n:
                break
    print(f"  Loaded {len(texts)} reference texts from WikiText-103")
    return texts


def load_p1_texts():
    """Load generated texts from P1 gamma sweep."""
    # P1 saved texts as part of the sweep — we need to regenerate or load them
    # The P1 script didn't save raw texts to disk, so we'll load from the P1 json
    # which doesn't have texts either. We need to regenerate.
    # Actually, let's check if there are saved texts
    import glob
    # The gamma sweep json has the sweep summary but not raw texts
    # We'll need to generate fresh for MAUVE
    return None


def generate_texts_for_gamma(gamma, n=500):
    """Generate texts at a specific gamma value."""
    import torch
    import torch.nn.functional as F

    SRC = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(SRC))
    from modules.model import ELF_models
    from utils.sampling_utils import _sde_step, _ode_step, get_sampling_steps
    from configs.config import Config

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    L = 1024; ENC = 512; BS = 4; STEPS = 32; SC_CFG = 3.0; NOISE_SCALE = 2.0

    cfg = Config()
    cfg.denoiser_p_mean = -1.5; cfg.denoiser_p_std = 0.8
    cfg.denoiser_noise_scale = NOISE_SCALE; cfg.t_eps = 0.05
    cfg.num_self_cond_cfg_tokens = 4; cfg.self_cond_prob = 0.5

    COMMON = dict(text_encoder_dim=ENC, max_length=L, bottleneck_dim=128,
                  num_time_tokens=4, num_self_cond_cfg_tokens=4,
                  num_model_mode_tokens=4, vocab_size=32100)

    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "embedded-language-flows/ELF-B-owt-torch"
    files = list_repo_files(repo)
    ckpts = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    path = hf_hub_download(repo, ckpts[0])
    model = ELF_models["ELF-B"](**COMMON)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_params1") or ckpt.get("params") or ckpt
    model.load_state_dict(sd, strict=False)
    model = model.to(DEV, torch.float32).eval()

    from transformers import T5Tokenizer
    t5tok = T5Tokenizer.from_pretrained("t5-small")

    texts = []
    n_batches = (n + BS - 1) // BS
    for bi in range(n_batches):
        bs = min(BS, n - bi * BS)
        torch.manual_seed(42 + bi)
        noise = torch.randn(bs, L, ENC, dtype=torch.float32, device=DEV) * NOISE_SCALE
        torch.manual_seed(1000 + bi)
        steps = get_sampling_steps(STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)

        cond = torch.zeros(bs, L, ENC, dtype=torch.float32, device=DEV)
        mask = torch.zeros(bs, L, dtype=torch.float32, device=DEV)
        x_prev = None
        z = noise.clone()

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
            if gamma == 0.0:
                for i in range(STEPS):
                    z, x_prev = _ode_step(model, z, steps[i].item(), steps[i+1].item(),
                                          x_prev, cfg, 1.0, SC_CFG, cond, mask)
            else:
                gen = torch.Generator(device=DEV)
                gen.manual_seed(7777 + bi)
                for i in range(STEPS):
                    z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(),
                                          x_prev, cfg, 1.0, SC_CFG, cond, mask, gamma, gen)

            t_f = torch.ones(bs, dtype=torch.float32, device=DEV)
            dec = torch.ones(bs, device=DEV)
            sc = torch.full((bs,), SC_CFG, dtype=torch.float32, device=DEV)
            z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
            _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
            tokens = logits.argmax(dim=-1).cpu()

        for i in range(bs):
            texts.append(t5tok.decode(tokens[i].tolist(), skip_special_tokens=True))

    del model; torch.cuda.empty_cache()
    return texts


def compute_mauve(generated_texts, reference_texts):
    """Compute MAUVE score between generated and reference texts."""
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Phase P6 — MAUVE Score (distribution-level quality)")
    print("=" * 70)
    t0 = time.time()

    # Load reference
    ref_texts = load_reference_texts(n=500)

    gammas = [0.0, 0.5, 1.0, 1.5]
    results = []

    for gamma in gammas:
        print(f"\n  ── γ = {gamma} ──")
        gt0 = time.time()

        # Generate texts
        print(f"    Generating 500 texts...")
        gen_texts = generate_texts_for_gamma(gamma, n=500)
        gen_time = time.time() - gt0

        # Compute MAUVE
        print(f"    Computing MAUVE score...")
        mauve_score = compute_mauve(gen_texts, ref_texts)
        total_time = time.time() - gt0

        print(f"    MAUVE: {mauve_score:.4f} (time: {total_time:.0f}s)")
        results.append({
            "gamma": gamma, "mauve": mauve_score,
            "n_generated": len(gen_texts), "n_reference": len(ref_texts),
            "time_s": total_time,
        })

    # Summary
    print(f"\n  ── MAUVE Summary ──")
    print(f"  {'γ':>6s}  {'MAUVE':>8s}")
    for r in results:
        print(f"  {r['gamma']:6.1f}  {r['mauve']:8.4f}")

    best = max(results, key=lambda r: r["mauve"])
    print(f"\n  ★ Best γ = {best['gamma']} (MAUVE = {best['mauve']:.4f})")

    with open(OUT / "p6_mauve.json", "w") as f:
        json.dump({"results": results, "total_time_s": time.time() - t0}, f, indent=2)

    with open(OUT / "p6_mauve.md", "w") as f:
        f.write("# Phase P6 — MAUVE Score\n\n")
        f.write("| γ | MAUVE ↑ | N_gen | N_ref |\n|---|---|---|---|\n")
        for r in results:
            star = " ★" if r["gamma"] == best["gamma"] else ""
            f.write(f"| {r['gamma']}{star} | {r['mauve']:.4f} | {r['n_generated']} | {r['n_reference']} |\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
