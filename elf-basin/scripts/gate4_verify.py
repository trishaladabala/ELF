#!/usr/bin/env python3
import sys
import os
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.train.adapter import DecoderAdapter
from elfbasin.eval.metrics import compute_bleu

def _compute_entropy(logits):
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log2(probs + 1e-12)).sum(dim=-1)

def main():
    print("="*70)
    print("  Gate 4: Adapter Verification")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELFWrapper("ELF-B-de-en", device=device)
    
    TRAIN_DIR = _SCRIPT_DIR.parent / "runs" / "train_adapter"
    if not TRAIN_DIR.exists():
        print("Trained adapters not found. Run train_adapter.py first.")
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
    
    # We will just evaluate BLEU and Entropy on the test set using the adapters.
    # To do that properly, we should run the full sampler with the adapted decode head.
    # However, since the denoiser is frozen, the states x_hat_s are identical for all adapters
    # up until the final step? No, the adapter is only used on the *decode* path!
    # "The denoiser trunk stays frozen. Only the decode path is adapted."
    # So the generation trajectory is exactly the same for all variants if we don't change
    # the self-conditioning! But wait, self-conditioning uses decode mode? No, SC uses denoiser.
    # Therefore, x_hat_final is identical for all adapters. We just decode the final x_hat.
    
    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_deen"
    final_tokens_path = RESULTS_DIR / "final_tokens.npy"
    if not final_tokens_path.exists():
         print("Missing final_tokens.npy")
         return 1
         
    # Let's load the x_hat at the final step (step 62 or step 64 depending on what was harvested)
    # The last harvested step is 62
    x_hat_data = np.lib.format.open_memmap(RESULTS_DIR / "step_62" / "x_hat.npy", mode='r')[2000:3000]
    
    references = [item["target"] for item in ds]
    
    results = {}
    
    variants = [0, 1, 2, 3, 4, 5, 6]
    for v in variants:
        print(f"\nEvaluating Variant {v}...")
        
        if v > 0:
            ckpt_path = TRAIN_DIR / f"adapter_v{v}.pt"
            if not ckpt_path.exists():
                print(f"  Missing {ckpt_path}, skipping.")
                continue
                
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            
            # Detect adapter type and dimension
            if any("lora" in k for k in state_dict.keys()):
                # It's a LoRA adapter
                from elfbasin.train.lora_adapter import LoRADecoderAdapter
                # We can infer r from the shape of lora_A: [in_features, r]
                sample_key = next(k for k in state_dict.keys() if "lora_A" in k)
                r = state_dict[sample_key].shape[1]
                adapter = LoRADecoderAdapter(wrapper, r=r).to(device)
            else:
                # It's a bottleneck adapter
                from elfbasin.train.adapter import DecoderAdapter
                r = state_dict["adapter.0.weight"].shape[0]
                adapter = DecoderAdapter(wrapper, r=r).to(device)
                
            adapter.load_state_dict(state_dict)
        else:
            # For variant 0, just use the wrapper (no adapter)
            adapter = None
            
        if adapter is not None:
            adapter.eval()
        
        hyps = []
        entropies = []
        
        batch_size = 32
        for i in tqdm(range(0, 1000, batch_size)):
            end = min(1000, i + batch_size)
            x_hat_batch = torch.from_numpy(np.array(x_hat_data[i:end])).to(torch.float32).to(device)
            batch_ds = ds[i:end]
            cond_lens = torch.tensor([item["condition_sequence_length"] for item in batch_ds], dtype=torch.int32, device=device)
            
            with torch.no_grad():
                if v == 0:
                    logits = wrapper.decode(x_hat_batch)
                else:
                    logits = adapter(x_hat_batch)
                    
                ent = _compute_entropy(logits).mean().item()
                entropies.append(ent)
                
                # Proper decode pipeline matching Phase 0
                from utils.generation_utils import mask_after_eos, shift_left
                all_ids = logits.argmax(dim=-1)
                B_cur = all_ids.shape[0]
                pred_ids = shift_left(all_ids, cond_lens, 0)[:, :64]
                pred_ids = mask_after_eos(
                    pred_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
                for j in range(B_cur):
                    text = tokenizer.decode(pred_ids[j].cpu().numpy(), skip_special_tokens=True)
                    hyps.append(text)
                    
        bleu = compute_bleu(hyps, references)
        mean_ent = np.mean(entropies)
        
        # Compute diversity metrics
        from elfbasin.eval.metrics import compute_diversity
        diversity = compute_diversity(hyps)
        
        print(f"  BLEU: {bleu:.2f} | Entropy: {mean_ent:.3f} | Dist-2: {diversity['distinct_2']:.4f} | Rep-4: {diversity['repeated_4gram_fraction']:.4f}")
        results[f"v{v}"] = {"bleu": bleu, "entropy": mean_ent, "distinct_2": diversity["distinct_2"], "rep_4gram": diversity["repeated_4gram_fraction"]}
        
    v0_bleu = results.get("v0", {}).get("bleu", 0)
    v5_bleu = results.get("v5", {}).get("bleu", 0)
    v6_bleu = results.get("v6", {}).get("bleu", 0)
    v0_ent = results.get("v0", {}).get("entropy", 0)
    v5_ent = results.get("v5", {}).get("entropy", 0)
    v6_ent = results.get("v6", {}).get("entropy", 0)
    
    best_bleu = max(v5_bleu, v6_bleu)
    best_v = 5 if v5_bleu > v6_bleu else 6
    best_ent = v5_ent if best_v == 5 else v6_ent
    
    pass_gate = (best_bleu > v0_bleu) and (best_ent >= v0_ent - 0.05) # Allow tiny variance
    
    print("\n" + "="*70)
    print("  GATE 4 — VERIFICATION RESULTS")
    print("="*70)
    print(f"  Baseline (v0) BLEU: {v0_bleu:.2f}, Entropy: {v0_ent:.3f}")
    print(f"  Best Method (v{best_v}) BLEU: {best_bleu:.2f}, Entropy: {best_ent:.3f}")
    
    if pass_gate:
        print(f"  ✓ GATE 4 PASSED — Method v{best_v} improves BLEU without degrading entropy.")
    else:
        print("  ✗ GATE 4 FAILED — Method does not beat baseline or degrades entropy.")
        
    with open(TRAIN_DIR / "gate4_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return 0 if pass_gate else 1

if __name__ == "__main__":
    sys.exit(main())
