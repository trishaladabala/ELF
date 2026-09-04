#!/usr/bin/env python3
"""Stage 5 — Conditional Task Generalization (WMT14 + XSum).

Adapted from phase0m/testM1_wmt.py and testM2_xsum.py for CUDA.
Tests whether ODE/SDE phenomena extend to conditional generation.
"""
import json, sys, math, time, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.stats import pearsonr

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from modules.model import ELF_models
from utils.sampling_utils import _sde_step, get_sampling_steps
from configs.config import Config

OUT = Path(__file__).resolve().parent / "results" / "stage5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N_STEPS = 64; SDE_GAMMA = 1.5; SC_CFG = 1.0; CFG = 2.0
COMMON = dict(text_encoder_dim=512, bottleneck_dim=128, num_time_tokens=4,
              num_self_cond_cfg_tokens=4, num_model_mode_tokens=4, vocab_size=32100)

cfg = Config()
cfg.denoiser_p_mean = -1.5; cfg.denoiser_p_std = 0.8
cfg.denoiser_noise_scale = 2.0; cfg.t_eps = 0.05
cfg.num_self_cond_cfg_tokens = 4; cfg.self_cond_prob = 0.5

def load_model(name, seq_len):
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = f"embedded-language-flows/{name}-torch"
    files = list_repo_files(repo)
    ckpts = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    path = hf_hub_download(repo, ckpts[0]) if ckpts else None
    model = ELF_models["ELF-B"](max_length=seq_len, **COMMON)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_params1") or ckpt.get("params") or ckpt
    model.load_state_dict(sd, strict=False)
    return model.to(DEV, torch.float32).eval()

def is_degenerate(tids, seq_len):
    ids = [t for t in tids if t not in (0, 1)]
    if not ids: return True
    if len(set(int(t) for t in ids)) <= 2: return True
    scale = max(1, seq_len // 128)
    if len(ids) >= 4:
        ng = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
        if Counter(ng).most_common(1)[0][1] >= 5 * scale: return True
    return False

def sample(model, n, seq_len, enc_dim, cond_seq, cond_mask, use_sde=True):
    gen = torch.Generator(device=DEV); gen.manual_seed(42)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        z = torch.randn(n, seq_len, enc_dim, generator=gen, dtype=torch.float32, device=DEV) * cfg.denoiser_noise_scale
        z = torch.where(cond_mask.unsqueeze(-1) > 0, cond_seq, z)
        steps = get_sampling_steps(N_STEPS, "logit_normal", cfg.denoiser_p_mean, cfg.denoiser_p_std, DEV, torch.float32)
        x_prev = None; gamma = SDE_GAMMA if use_sde else 0.0
        for i in range(N_STEPS):
            z, x_prev = _sde_step(model, z, steps[i].item(), steps[i+1].item(), x_prev,
                                   cfg, CFG, SC_CFG, cond_seq, cond_mask, gamma, gen)
    return z

def decode(model, z):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=DEV=='cuda'):
        t_f = torch.ones(z.shape[0], dtype=torch.float32, device=DEV)
        dec = torch.ones(z.shape[0], device=DEV)
        sc = torch.full((z.shape[0],), SC_CFG, dtype=torch.float32, device=DEV)
        z_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_in, t_f, deterministic=True, self_cond_cfg_scale=sc, decoder_step_active=dec)
        return logits.argmax(dim=-1).cpu()

def run_wmt(n=512, bs=8):
    import sacrebleu
    from datasets import load_dataset
    from transformers import T5Tokenizer
    from modules.t5_encoder import get_encoder
    from utils.encoder_utils import encode_text
    from utils.generation_utils import shift_left, mask_after_eos
    import utils.data_utils as du

    print("\n" + "="*70 + "\n  WMT14 De-En (N=%d)\n" % n + "="*70)
    t5tok = T5Tokenizer.from_pretrained("t5-small")
    _, t5enc = get_encoder("t5-small", dtype=torch.float32)
    t5enc = t5enc.to(DEV).eval()

    c_len, t_len = 64, 64; seq_len = c_len + t_len
    model = load_model("ELF-B-de-en", seq_len)

    ds = load_dataset("wmt/wmt14", "de-en", split="test")
    samples = ds.select(range(n))
    de = [s["translation"]["de"] for s in samples]
    en = [s["translation"]["en"] for s in samples]

    data = [{"index": i, "input": de[i], "target": en[i],
             "condition_input_ids": t5tok(de[i], add_special_tokens=True)["input_ids"][:c_len],
             "input_ids": t5tok(en[i], add_special_tokens=True)["input_ids"][:t_len]} for i in range(n)]

    dl = du.get_dataloader(data, batch_size=bs, shuffle=False, num_workers=0,
                           max_seq_length=seq_len, pad_token_id=t5tok.eos_token_id,
                           max_input_seq_length=c_len, distributed=False)

    ode_texts, sde_texts, ode_degs, sde_degs = [], [], [], []
    for bi, batch in enumerate(dl):
        print(f"  Batch {bi+1}/{len(dl)}...")
        bsz = batch["input_ids"].shape[0]
        iids = torch.from_numpy(np.array(batch["input_ids"])).to(DEV).long()
        amask = torch.from_numpy(np.array(batch["attention_mask"])).to(DEV).long()
        cmask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(DEV).float()
        cond = encode_text(input_ids=iids, attention_mask=amask, encoder=t5enc,
                           latent_mean=0.0, latent_std=0.2).to(torch.float32)

        z_ode = sample(model, bsz, seq_len, 512, cond, cmask, use_sde=False)
        z_sde = sample(model, bsz, seq_len, 512, cond, cmask, use_sde=True)
        t_ode = decode(model, z_ode); t_sde = decode(model, z_sde)
        cpl = cmask.to(torch.int32).sum(dim=1)
        t_ode = mask_after_eos(shift_left(t_ode, cpl, 0)[:, :t_len], t5tok.eos_token_id, t5tok.eos_token_id)
        t_sde = mask_after_eos(shift_left(t_sde, cpl, 0)[:, :t_len], t5tok.eos_token_id, t5tok.eos_token_id)
        for i in range(bsz):
            ode_texts.append(t5tok.decode(t_ode[i].tolist(), skip_special_tokens=True))
            sde_texts.append(t5tok.decode(t_sde[i].tolist(), skip_special_tokens=True))
            ode_degs.append(is_degenerate(t_ode[i].tolist(), t_len))
            sde_degs.append(is_degenerate(t_sde[i].tolist(), t_len))

    bleu_ode = sacrebleu.corpus_bleu(ode_texts, [en]).score
    bleu_sde = sacrebleu.corpus_bleu(sde_texts, [en]).score
    np.random.seed(42)
    diffs = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        ob = sacrebleu.corpus_bleu([ode_texts[i] for i in idx], [[en[i] for i in idx]]).score
        sb = sacrebleu.corpus_bleu([sde_texts[i] for i in idx], [[en[i] for i in idx]]).score
        diffs.append(sb - ob)
    ci_lo, ci_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

    r = {"ode_bleu": bleu_ode, "sde_bleu": bleu_sde, "diff_mean": float(np.mean(diffs)),
         "diff_ci": [float(ci_lo), float(ci_hi)],
         "ode_deg": sum(ode_degs)/n, "sde_deg": sum(sde_degs)/n}
    print(f"  ODE BLEU: {bleu_ode:.1f}, SDE BLEU: {bleu_sde:.1f}, diff CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  ODE deg: {100*r['ode_deg']:.1f}%, SDE deg: {100*r['sde_deg']:.1f}%")
    del model; torch.cuda.empty_cache()
    return r

def run_xsum(n=512, bs=8):
    from rouge_score import rouge_scorer
    from datasets import load_dataset
    from transformers import T5Tokenizer
    from modules.t5_encoder import get_encoder
    from utils.encoder_utils import encode_text
    from utils.generation_utils import shift_left, mask_after_eos
    import utils.data_utils as du

    print("\n" + "="*70 + "\n  XSum Summarization (N=%d)\n" % n + "="*70)
    t5tok = T5Tokenizer.from_pretrained("t5-small")
    _, t5enc = get_encoder("t5-small", dtype=torch.float32)
    t5enc = t5enc.to(DEV).eval()

    c_len, t_len = 64, 64; seq_len = c_len + t_len
    model = load_model("ELF-B-xsum", seq_len)

    ds = load_dataset("EdinburghNLP/xsum", split="validation")
    samples = ds.select(range(n))
    docs = [s["document"] for s in samples]
    sums = [s["summary"] for s in samples]

    data = [{"index": i, "input": docs[i], "target": sums[i],
             "condition_input_ids": t5tok(docs[i], add_special_tokens=True)["input_ids"][:c_len],
             "input_ids": t5tok(sums[i], add_special_tokens=True)["input_ids"][:t_len]} for i in range(n)]

    dl = du.get_dataloader(data, batch_size=bs, shuffle=False, num_workers=0,
                           max_seq_length=seq_len, pad_token_id=t5tok.eos_token_id,
                           max_input_seq_length=c_len, distributed=False)

    ode_texts, sde_texts, ode_degs, sde_degs = [], [], [], []
    for bi, batch in enumerate(dl):
        if (bi+1) % 8 == 0: print(f"  Batch {bi+1}/{len(dl)}...")
        bsz = batch["input_ids"].shape[0]
        iids = torch.from_numpy(np.array(batch["input_ids"])).to(DEV).long()
        amask = torch.from_numpy(np.array(batch["attention_mask"])).to(DEV).long()
        cmask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(DEV).float()
        cond = encode_text(input_ids=iids, attention_mask=amask, encoder=t5enc,
                           latent_mean=0.0, latent_std=0.2).to(torch.float32)

        z_ode = sample(model, bsz, seq_len, 512, cond, cmask, use_sde=False)
        z_sde = sample(model, bsz, seq_len, 512, cond, cmask, use_sde=True)
        t_ode = decode(model, z_ode); t_sde = decode(model, z_sde)
        cpl = cmask.to(torch.int32).sum(dim=1)
        t_ode = mask_after_eos(shift_left(t_ode, cpl, 0)[:, :t_len], t5tok.eos_token_id, t5tok.eos_token_id)
        t_sde = mask_after_eos(shift_left(t_sde, cpl, 0)[:, :t_len], t5tok.eos_token_id, t5tok.eos_token_id)
        for i in range(bsz):
            ode_texts.append(t5tok.decode(t_ode[i].tolist(), skip_special_tokens=True))
            sde_texts.append(t5tok.decode(t_sde[i].tolist(), skip_special_tokens=True))
            ode_degs.append(is_degenerate(t_ode[i].tolist(), t_len))
            sde_degs.append(is_degenerate(t_sde[i].tolist(), t_len))

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    def avg_rouge(hyps, refs):
        scores = [scorer.score(r, h) for h, r in zip(hyps, refs)]
        return {k: float(np.mean([s[k].fmeasure for s in scores])) for k in ["rouge1","rouge2","rougeL"]}

    ode_rouge = avg_rouge(ode_texts, sums)
    sde_rouge = avg_rouge(sde_texts, sums)

    # Bootstrap CI for ROUGE-1 diff
    np.random.seed(42)
    diffs = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        or_ = avg_rouge([ode_texts[i] for i in idx], [sums[i] for i in idx])["rouge1"]
        sr_ = avg_rouge([sde_texts[i] for i in idx], [sums[i] for i in idx])["rouge1"]
        diffs.append(sr_ - or_)
    ci_lo, ci_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

    r = {"ode_rouge": ode_rouge, "sde_rouge": sde_rouge,
         "rouge1_diff_ci": [float(ci_lo), float(ci_hi)],
         "ode_deg": sum(ode_degs)/n, "sde_deg": sum(sde_degs)/n}
    print(f"  ODE ROUGE-1: {ode_rouge['rouge1']:.3f}, SDE ROUGE-1: {sde_rouge['rouge1']:.3f}")
    print(f"  ROUGE-1 diff CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  ODE deg: {100*r['ode_deg']:.1f}%, SDE deg: {100*r['sde_deg']:.1f}%")
    del model; torch.cuda.empty_cache()
    return r

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wmt_r = run_wmt(n=512, bs=8)
    xsum_r = run_xsum(n=512, bs=8)

    results = {"wmt14": wmt_r, "xsum": xsum_r, "total_time_s": time.time() - t0}
    with open(OUT / "stage5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT / "stage5_verdict.md", "w") as f:
        f.write("# Stage 5 — Conditional Task Generalization\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## WMT14 De-En\n\n")
        f.write(f"| Sampler | BLEU | Degeneracy |\n|---|---|---|\n")
        f.write(f"| ODE | {wmt_r['ode_bleu']:.1f} | {100*wmt_r['ode_deg']:.1f}% |\n")
        f.write(f"| SDE | {wmt_r['sde_bleu']:.1f} | {100*wmt_r['sde_deg']:.1f}% |\n")
        f.write(f"| Diff CI | [{wmt_r['diff_ci'][0]:.2f}, {wmt_r['diff_ci'][1]:.2f}] | |\n\n")
        f.write("## XSum\n\n")
        f.write(f"| Sampler | ROUGE-1 | ROUGE-2 | ROUGE-L | Degeneracy |\n|---|---|---|---|---|\n")
        f.write(f"| ODE | {xsum_r['ode_rouge']['rouge1']:.3f} | {xsum_r['ode_rouge']['rouge2']:.3f} | {xsum_r['ode_rouge']['rougeL']:.3f} | {100*xsum_r['ode_deg']:.1f}% |\n")
        f.write(f"| SDE | {xsum_r['sde_rouge']['rouge1']:.3f} | {xsum_r['sde_rouge']['rouge2']:.3f} | {xsum_r['sde_rouge']['rougeL']:.3f} | {100*xsum_r['sde_deg']:.1f}% |\n")
        f.write(f"| R1 diff CI | [{xsum_r['rouge1_diff_ci'][0]:.4f}, {xsum_r['rouge1_diff_ci'][1]:.4f}] | | | |\n\n")
        f.write(f"**Total time:** {time.time()-t0:.0f}s\n")

    print(f"\n  Saved → {OUT}")
    print(f"  Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
