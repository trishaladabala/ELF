#!/usr/bin/env python3
"""Gate 3: Headroom / ceiling test
Measurements:
1. Oracle-decoder ceiling
2. Early-state ceiling
3. Error taxonomy
"""
import sys
import os
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.eval.metrics import compute_bleu
from transformers import AutoTokenizer
from utils.generation_utils import mask_after_eos, shift_left

def _ids_to_texts(ids_np, tokenizer, cond_lens, seq_len=128):
    """Convert raw (B, L) token IDs to proper hypothesis texts.
    Applies shift_left + mask_after_eos to match Phase 0 pipeline."""
    import torch
    ids_t = torch.from_numpy(ids_np).long()
    B = ids_t.shape[0]
    
    cond_lens_t = torch.tensor(cond_lens, dtype=torch.int32) if not isinstance(cond_lens, torch.Tensor) else cond_lens
    pred_ids = shift_left(ids_t, cond_lens_t, 0)[:, :64]
    
    pred_ids = mask_after_eos(
        pred_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    texts = []
    for j in range(B):
        texts.append(tokenizer.decode(pred_ids[j].numpy(), skip_special_tokens=True))
    return texts

def main():
    print("="*70)
    print("  Gate 3: Headroom / Ceiling Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELFWrapper("ELF-B-de-en", device=device)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_deen"
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist. Run harvest_deen.py first.")
        return 1
        
    num_samples = 3000
    seq_len = 128
    batch_size = 64
    
    print("Loading eval dataset for references...")
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id="embedded-language-flows/wmt14_de-en_validation_t5", repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    eval_dataset = table.to_pandas().to_dict('records')
    
    ds = eval_dataset[:num_samples]
    references = [item["target"] for item in ds]
        
    # 1. Native decoder BLEU
    final_tokens_path = RESULTS_DIR / "final_tokens.npy"
    final_tokens_mmap = np.lib.format.open_memmap(final_tokens_path, mode='r')
    
    native_hyps = []
    for i in range(0, num_samples, batch_size):
        end = min(num_samples, i + batch_size)
        ids_batch = final_tokens_mmap[i:end]
        batch_cond_lens = [item["condition_sequence_length"] for item in ds[i:end]]
        native_hyps.extend(_ids_to_texts(ids_batch, tokenizer, batch_cond_lens))
        
    native_bleu = compute_bleu(native_hyps, references)
    print(f"Native Decoder BLEU: {native_bleu:.2f}")
    
    # 2. Oracle decoder BLEU
    # Decode gt_encoder_states
    gt_enc_path = RESULTS_DIR / "gt_encoder_states.npy"
    gt_enc_mmap = np.lib.format.open_memmap(gt_enc_path, mode='r')
    
    oracle_hyps = []
    oracle_ids = np.zeros((num_samples, seq_len), dtype=np.int16)
    
    batch_size = 64
    print("\nComputing oracle decoder ceiling...")
    for i in tqdm(range(0, num_samples, batch_size)):
        end = min(num_samples, i + batch_size)
        states = torch.from_numpy(gt_enc_mmap[i:end].astype(np.float32)).to(device)
        batch_cond_lens = [item["condition_sequence_length"] for item in ds[i:end]]
        with torch.no_grad():
            logits = wrapper.decode(states, sc_cfg=1.0)
            ids = logits.argmax(dim=-1).cpu().numpy()
            oracle_ids[i:end] = ids
            
            oracle_hyps.extend(_ids_to_texts(ids, tokenizer, batch_cond_lens))
                
    oracle_bleu = compute_bleu(oracle_hyps, references)
    print(f"Oracle Decoder BLEU: {oracle_bleu:.2f}")
    print(f"Oracle Improvement: +{oracle_bleu - native_bleu:.2f} BLEU")
    
    # 3. Early state ceilings
    # For De-En 64-step, our harvested steps are [8, 16, 24, 32, 40, 48, 56, 62]
    # Let's decode x_hat at 40, 48, 56
    early_steps = [40, 48, 56]
    early_bleus = {}
    
    print("\nComputing early-state native ceilings...")
    for step in early_steps:
        x_hat_path = RESULTS_DIR / f"step_{step}" / "x_hat.npy"
        if not x_hat_path.exists():
            continue
        x_hat_mmap = np.lib.format.open_memmap(x_hat_path, mode='r')
        
        early_hyps = []
        for i in tqdm(range(0, num_samples, batch_size), desc=f"Step {step}"):
            end = min(num_samples, i + batch_size)
            states = torch.from_numpy(x_hat_mmap[i:end].astype(np.float32)).to(device)
            batch_cond_lens = [item["condition_sequence_length"] for item in ds[i:end]]
            with torch.no_grad():
                logits = wrapper.decode(states, sc_cfg=1.0)
                ids = logits.argmax(dim=-1).cpu().numpy()
                early_hyps.extend(_ids_to_texts(ids, tokenizer, batch_cond_lens))
        
        eb = compute_bleu(early_hyps, references)
        early_bleus[step] = eb
        print(f"Step {step} Native BLEU: {eb:.2f} (Savings: {100*(64-step)/64:.1f}% NFEs)")
        
    # 4. Error taxonomy
    print("\nComputing error taxonomy...")
    # Count positions where native_ids != oracle_ids
    # We only care about the target side (64:128) and non-padding
    errors = 0
    total = 0
    for i in range(num_samples):
        native = final_tokens_mmap[i, 64:128]
        oracle = oracle_ids[i, 64:128]
        gt = gt_tokens_mmap[i, 64:128]
        
        # Valid positions are where gt is not padding (0 is usually pad for T5)
        valid = (gt != 0)
        total += valid.sum()
        errors += (native[valid] != oracle[valid]).sum()
        
    print(f"Total valid target tokens: {total}")
    print(f"Total mismatch errors (native vs oracle): {errors} ({errors/total*100:.1f}%)")
    
    # Gate 3 Decision
    improvement = oracle_bleu - native_bleu
    pass_gate = improvement >= 1.0
    
    print("\n" + "="*70)
    print("  GATE 3 — DECISION POINT")
    print("="*70)
    print(f"  Oracle Improvement: +{improvement:.2f} BLEU")
    if pass_gate:
        print("  ✓ GATE 3 PASSED — Significant headroom found.")
        print("  Decision: Proceed to Phase 4 (Track A).")
    else:
        print("  ✗ GATE 3 FAILED — Headroom is too small (< 1.0 BLEU).")
        print("  Decision: Skip Phase 4. Go straight to Phase 5 (Track B).")
        
    # Save results
    results = {
        "native_bleu": native_bleu,
        "oracle_bleu": oracle_bleu,
        "oracle_improvement": improvement,
        "early_bleus": early_bleus,
        "error_taxonomy": {
            "total_tokens": int(total),
            "total_errors": int(errors),
            "error_rate": float(errors/max(1, total))
        },
        "gate_3_pass": bool(pass_gate)
    }
    with open(RESULTS_DIR / "gate3_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return 0 if pass_gate else 1

if __name__ == "__main__":
    sys.exit(main())
