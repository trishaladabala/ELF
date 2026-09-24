#!/usr/bin/env python3
"""Generate qualitative examples for the paper.

Compares outputs from:
1. Ground Truth Reference
2. Teacher (64 steps)
3. Consistency Distillation (4 steps)
4. Progressive Distillation (4 steps)
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos, shift_left
from utils.encoder_utils import encode_text
from utils.data_utils import get_dataloader, get_pad_token_id
from configs.config import SamplingConfig

CHECKPOINT = "ELF-B-de-en"
CONSISTENCY_DIR = _SCRIPT_DIR.parent / "runs" / "consistency_distill"
PROGRESSIVE_DIR = _SCRIPT_DIR.parent / "runs" / "progressive_distill"
OUT_DIR = _SCRIPT_DIR.parent / "runs" / "analysis"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load wrapper and tokenizer
    base_wrapper = ELFWrapper(CHECKPOINT, device=device)
    config = base_wrapper.config
    config.use_bf16 = True
    config.online_eval = True
    tokenizer = base_wrapper.tokenizer
    encoder = base_wrapper.encoder
    pad_token_id = get_pad_token_id(tokenizer, "eos")
    eos_token_id = tokenizer.eos_token_id

    # Load dataset (first few examples of the validation set we used)
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id="embedded-language-flows/wmt14_de-en_validation_t5", repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        table = ipc.RecordBatchStreamReader(f).read_all()
    eval_dataset = table.to_pandas().to_dict('records')
    
    # Grab 5 random examples from the test set (last 1000)
    import random
    random.seed(42)
    test_data = eval_dataset[-1000:]
    sample_indices = random.sample(range(len(test_data)), 10)
    samples = [test_data[i] for i in sample_indices]

    loader = get_dataloader(
        samples, batch_size=10, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=getattr(config, 'max_input_length', 64), distributed=False,
    )
    batch = next(iter(loader))

    # Load models
    models = {}
    models["Teacher (64-step)"] = (base_wrapper.model, 64)
    
    c_path = CONSISTENCY_DIR / "ema_best.pt"
    if c_path.exists():
        c_wrapper = ELFWrapper(CHECKPOINT, device=device)
        c_wrapper.model.load_state_dict(torch.load(c_path, map_location=device))
        models["Ours (4-step)"] = (c_wrapper.model, 4)
        
    p_path = PROGRESSIVE_DIR / "student_4step.pt"
    if p_path.exists():
        p_wrapper = ELFWrapper(CHECKPOINT, device=device)
        p_wrapper.model.load_state_dict(torch.load(p_path, map_location=device))
        models["Progressive (4-step)"] = (p_wrapper.model, 4)

    # Generate
    results = {name: [] for name in models}
    sources = []
    targets = []
    
    bsz = batch["input_ids"].shape[0]
    input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
    encoder_attn_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
    cond_seq_mask = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()
    cond_seq = encode_text(
        input_ids=input_ids, attention_mask=encoder_attn_mask,
        encoder=encoder, latent_mean=config.latent_mean, latent_std=config.latent_std,
    ).to(torch.bfloat16)

    # Reconstruct original text
    for i in range(bsz):
        src = tokenizer.decode(input_ids[i][input_ids[i] != pad_token_id], skip_special_tokens=True)
        sources.append(src)
        targets.append(samples[i]["target"])

    for name, (model, steps) in models.items():
        torch.manual_seed(42)
        generator = torch.Generator(device="cuda").manual_seed(42)
        
        t_steps = get_sampling_steps(
            n_steps=steps, time_schedule="logit_normal",
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=torch.bfloat16,
        )
        z = (torch.randn((bsz, config.max_length, 512), generator=generator, dtype=torch.bfloat16, device=device)
             * config.denoiser_noise_scale)

        s_config = SamplingConfig(sampling_method="ode", num_sampling_steps=[steps], cfgs=[2.0], self_cond_cfg_scales=[1.0], time_schedule="logit_normal")
        
        with torch.no_grad():
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                config=config, sampling_config=s_config, cfg_scale=2.0, self_cond_cfg_scale=1.0,
            )
            cond_len_per_sample = cond_seq_mask.to(torch.int32).sum(dim=1)
            predicted_ids = _dlm_decode_batch(z=latent, model=model, t_final_val=t_steps[-1].item(), config=config, self_cond_cfg_scale=1.0)
            predicted_ids = shift_left(predicted_ids, cond_len_per_sample, 0)[:, :config.max_length-64]
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            
            for i in range(bsz):
                results[name].append(tokenizer.decode(predicted_ids[i].cpu().numpy(), skip_special_tokens=True))

    # Write to markdown
    md_path = OUT_DIR / "qualitative_examples.md"
    with open(md_path, "w") as f:
        f.write("# Qualitative Examples: Distillation Comparison\n\n")
        for i in range(bsz):
            f.write(f"### Example {i+1}\n")
            f.write(f"**Source (DE):** {sources[i]}\n\n")
            f.write(f"**Reference (EN):** {targets[i]}\n\n")
            for name in models:
                f.write(f"**{name}:** {results[name][i]}\n\n")
            f.write("---\n\n")

    print(f"Qualitative examples saved to {md_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
