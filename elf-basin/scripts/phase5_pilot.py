#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.sampling.anchored_sampler import anchored_resume_generate
from elfbasin.eval.metrics import compute_bleu
from utils.sampling_utils import get_sampling_steps

def main():
    print("="*70)
    print("  Phase 5: Pilot Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELFWrapper("ELF-B-de-en", device=device)
    config = wrapper.config
    
    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_deen"
    if not RESULTS_DIR.exists():
        print("Harvested data not found.")
        return 1
        
    print("Loading test dataset (last 1000 samples of val)...")
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id="embedded-language-flows/wmt14_de-en_validation_t5", repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    eval_dataset = table.to_pandas().to_dict('records')
    ds = eval_dataset[2000:3000]
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    references = [item["target"] for item in ds]
    
    # Load condition sequences for the 1000 samples
    sources = [item["input"] for item in ds]
    input_encodings = tokenizer(sources, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    cond_tokens = input_encodings.input_ids.to(device)
    
    batch_size = 32
    num_samples = 1000
    
    cond_tokens_full = torch.zeros(num_samples, 128, dtype=torch.long, device=device)
    cond_tokens_full[:, :64] = cond_tokens
    
    with torch.no_grad():
        cond_seq = wrapper.encode(cond_tokens_full)
        
    cond_seq_mask = torch.zeros(num_samples, 128, device=device)
    cond_seq_mask[:, :64] = 1.0
    
    # Load z at step 40
    step = 40
    z_data = np.lib.format.open_memmap(RESULTS_DIR / f"step_{step}" / "z.npy", mode='r')[2000:3000]
    
    # The sampling steps used during harvest
    TIME_SCHEDULE = "logit_normal"
    t_steps = get_sampling_steps(
        n_steps=64, time_schedule=TIME_SCHEDULE,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=device, dtype=torch.float32,
    )
    
    # We want to start from index 40
    # Wait, De-En has 64 steps, so t_steps has length 65.
    # The ODE loop runs for i in range(len(t_steps)-1)
    # The harvested step 40 means `i = 40`.
    start_step_idx = 40
    
    variants = ["none", "freeze", "reencode"]
    results = {}
    
    for var in variants:
        print(f"\nRunning Pilot Variant: {var}")
        hyps = []
        
        for i in tqdm(range(0, num_samples, batch_size)):
            end = min(num_samples, i + batch_size)
            z_batch = torch.from_numpy(np.array(z_data[i:end])).to(torch.float32).to(device)
            c_seq = cond_seq[i:end]
            c_mask = cond_seq_mask[i:end]
            
            with torch.no_grad():
                # Compute margin to get top-decile positions
                # (For speed in this pilot, we use margin instead of rho)
                # First get x_hat at step 40
                t_val = t_steps[start_step_idx].item()
                from utils.sampling_utils import _ode_step
                _, x_hat = _ode_step(
                    z=z_batch, t=t_val, t_next=t_val, x_pred_prev=None,
                    model=wrapper.model, config=config,
                    cfg_scale=2.0, self_cond_cfg_scale=1.0,
                    cond_seq=c_seq, cond_seq_mask=c_mask
                )
                
                margin = wrapper.margin(x_hat, sc_cfg=1.0)
                
                # We only anchor the target tokens (64:128)
                margin[:, :64] = -1e9
                
                # Top decile: threshold at 90th percentile across the batch? Or per sequence?
                # The paper says "top-decile-ρ positions". Usually this means per sequence.
                # Since we only have 64 target tokens, top decile is top 6 tokens.
                k = 6
                topk_vals = margin.topk(k, dim=1).values
                thresh = topk_vals[:, -1:]
                anchor_positions = (margin >= thresh) & (margin > -1e8)
                
                # Run the anchored sampler
                z_final, _ = anchored_resume_generate(
                    model=wrapper.model, wrapper=wrapper, z=z_batch,
                    start_step_idx=start_step_idx, t_steps=t_steps,
                    cond_seq=c_seq, cond_seq_mask=c_mask,
                    config=config, cfg_scale=2.0, self_cond_cfg_scale=1.0,
                    intervention_type=var,
                    anchor_positions=anchor_positions,
                    blend_lambda=1.0
                )
                
                logits = wrapper.decode(z_final, sc_cfg=1.0)
                ids = logits.argmax(dim=-1)
                
                # Proper decode pipeline matching Phase 0
                from utils.generation_utils import mask_after_eos, shift_left
                B_cur = ids.shape[0]
                batch_ds = ds[i:end]
                cond_len_t = torch.tensor([item["condition_sequence_length"] for item in batch_ds], dtype=torch.int32, device=device)
                pred_ids = shift_left(ids, cond_len_t, 0)[:, :64]
                pred_ids = mask_after_eos(
                    pred_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
                for j in range(B_cur):
                    text = tokenizer.decode(pred_ids[j].cpu().numpy(), skip_special_tokens=True)
                    hyps.append(text)
                    
        bleu = compute_bleu(hyps, references)
        print(f"  BLEU: {bleu:.2f}")
        results[var] = bleu
        
    print("\n" + "="*70)
    print("  PHASE 5 PILOT RESULTS")
    print("="*70)
    print(f"  (a) Baseline (no anchor) BLEU : {results['none']:.2f}")
    print(f"  (b) Freeze in place BLEU      : {results['freeze']:.2f}")
    print(f"  (c) Re-encode & Anchor BLEU   : {results['reencode']:.2f}")
    
    pass_pilot = results['reencode'] > results['freeze']
    
    if pass_pilot:
        print("  ✓ PILOT PASSED — Re-encoding is better than freezing.")
    else:
        print("  ✗ PILOT FAILED — Re-encoding does not beat freezing. Stop Track B.")
        
    out_dir = _SCRIPT_DIR.parent / "runs" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return 0 if pass_pilot else 1

if __name__ == "__main__":
    sys.exit(main())
