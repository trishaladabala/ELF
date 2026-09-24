import os, sys, json
import torch
import numpy as np
from transformers import AutoTokenizer, GPT2TokenizerFast, GPT2LMHeadModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ELF_SRC = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'src'))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from efm_pipeline.efm_model import EFMModel, EFM_Tiny, EFM_Small
from efm_pipeline.eval_runner import generate_efm_samples
from efm_pipeline.utils import set_seed, get_device
from utils.generation_utils import mask_after_eos

def main():
    device = get_device()
    print(f"Device: {device}")
    
    # Check 1: Tokenizer consistency
    print("\n=== Check 1: Tokenizer consistency ===")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"T5 Tokenizer Vocab Size: {t5_tokenizer.vocab_size} (Expected ~32128)")
    print(f"GPT2 Tokenizer Vocab Size: {gpt2_tokenizer.vocab_size} (Used for Gen-PPL)")
    
    # Load Model
    ckpt_dir = os.path.join(_PROJ_ROOT, 'checkpoints', 'stage1_continuous_baseline')
    ckpt_path = os.path.join(ckpt_dir, 'model_final.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Auto-detect size
    n_embd = state_dict['token_emb.weight'].shape[1]
    if n_embd == 384:
        model = EFM_Tiny()
        print("Detected model size: EFM_Tiny (n_embd=384)")
    elif n_embd == 768:
        model = EFM_Small()
        print("Detected model size: EFM_Small (n_embd=768)")
    else:
        model = EFMModel()
        print(f"Detected model size: EFMModel (n_embd={n_embd})")
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Check 2: BOS/EOS handling
    print("\n=== Check 2: BOS/EOS handling ===")
    set_seed(42)
    with torch.no_grad():
        samples = generate_efm_samples(model, num_samples=32, device=device)
    
    eos_id = t5_tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 1
    
    has_eos_before = (samples == eos_id).any(dim=1).sum().item()
    masked_samples = mask_after_eos(samples.clone(), eos_id)
    has_eos_after = (masked_samples == eos_id).any(dim=1).sum().item()
    
    print(f"Samples with EOS before masking: {has_eos_before}/32")
    print(f"Samples with EOS after masking: {has_eos_after}/32 (Wait, mask_after_eos keeps the FIRST eos, so this should be same if EOS was present)")
    
    # Check 3: Sequence length distribution
    print("\n=== Check 3: Sequence length distribution ===")
    lengths = []
    decoded_texts = []
    for seq in masked_samples:
        valid_tokens = seq[seq != t5_tokenizer.pad_token_id]
        lengths.append(len(valid_tokens))
        decoded_texts.append(t5_tokenizer.decode(valid_tokens, skip_special_tokens=True))
    
    lengths = np.array(lengths)
    word_lengths = [len(text.split()) for text in decoded_texts]
    print(f"Token lengths - Min: {lengths.min()}, Max: {lengths.max()}, Mean: {lengths.mean():.2f}, Std: {lengths.std():.2f}")
    print(f"Word lengths - Min: {np.min(word_lengths)}, Max: {np.max(word_lengths)}, Mean: {np.mean(word_lengths):.2f}, Std: {np.std(word_lengths):.2f}")
    
    print("\nLength Histogram (tokens):")
    hist, bins = np.histogram(lengths, bins=5)
    for count, bin_edge in zip(hist, bins):
        print(f"{int(bin_edge)}+: {'*' * count}")

    # Check 4: Decoded samples
    print("\n=== Check 4: Decoded samples ===")
    samples_to_save = []
    for i in range(10):
        seq = masked_samples[i].tolist()
        text = decoded_texts[i]
        words = len(text.split())
        has_special = eos_id in seq
        
        sample_info = {
            "index": i,
            "token_ids": seq[:20],
            "word_count": words,
            "has_eos": has_special,
            "text": text
        }
        samples_to_save.append(sample_info)
        print(f"Sample {i}:")
        print(f"  Tokens (first 20): {seq[:20]}")
        print(f"  Words: {words} | Has EOS: {has_special}")
        
    # Check 5: Gen-PPL cross-validation
    print("\n=== Check 5: Gen-PPL cross-validation ===")
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    eval_model.eval()
    
    def calc_ppl(text):
        if not text.strip():
            return float('inf')
        encodings = gpt2_tokenizer(text, return_tensors="pt")
        max_length = eval_model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = eval_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
                
        if not nlls:
            return float('inf')
        return torch.exp(torch.stack(nlls).mean()).item()

    good_text = "The quick brown fox jumps over the lazy dog. It is a well-known sentence used to test typewriters and computer keyboards."
    random_text = "xjfq wl kf p a z c b q n r k"
    
    ppl_good = calc_ppl(good_text)
    ppl_random = calc_ppl(random_text)
    print(f"Gen-PPL on good text: {ppl_good:.2f} (Expected < 50)")
    print(f"Gen-PPL on random text: {ppl_random:.2f} (Expected >> 50)")

    # Check 6: Sampling seed determinism
    print("\n=== Check 6: Sampling seed determinism ===")
    set_seed(42)
    with torch.no_grad():
        s1 = generate_efm_samples(model, num_samples=8, device=device)
    set_seed(42)
    with torch.no_grad():
        s2 = generate_efm_samples(model, num_samples=8, device=device)
    
    if s1.shape == s2.shape:
        is_identical = torch.allclose(s1.float(), s2.float())
    else:
        is_identical = False
    print(f"Seed determinism check passed: {is_identical}")

    # Check 7: ODE integration comparison
    print("\n=== Check 7: ODE integration comparison ===")
    print("Time schedule: uniform, N steps (e.g. 250 in eval_runner usually)")
    print("Velocity computation: v = (flow_out - z) / max(1 - t, 5e-2)")
    print("FLAG: eval_runner uses `v = (flow_out - z) / max(1 - t, 5e-2)` which differs from ELF's `net_out_to_v_x`!")

    # Check 8: ELF-B vs EFM gap analysis
    print("\n=== Check 8: The ELF-B vs EFM gap analysis ===")
    print("ELF-B Gen-PPL: 18.3 vs EFM continuous: 6568")
    print("Possible causes:")
    causes = [
        "Model size: 105M vs ~20M",
        "Training data: full OWT vs 5k wikitext2 samples",
        "Training steps: millions vs 50k",
        "No self-conditioning or CFG in EFM eval",
        "No bf16 autocast in EFM eval",
        "Different velocity computation",
        "Decoder quality differences"
    ]
    for c in causes:
        print(f" - {c}")
        
    # Save reports
    val_dir = os.path.join(_PROJ_ROOT, 'validation_phase')
    os.makedirs(val_dir, exist_ok=True)
    
    report = {
        "tokenizer": {
            "t5_vocab": t5_tokenizer.vocab_size,
            "gpt2_vocab": gpt2_tokenizer.vocab_size
        },
        "bos_eos": {
            "before_mask": has_eos_before,
            "after_mask": has_eos_after
        },
        "lengths": {
            "token_min": int(lengths.min()),
            "token_max": int(lengths.max()),
            "token_mean": float(lengths.mean()),
            "word_min": int(np.min(word_lengths)),
            "word_max": int(np.max(word_lengths)),
            "word_mean": float(np.mean(word_lengths))
        },
        "ppl": {
            "good": ppl_good,
            "random": ppl_random
        },
        "determinism": is_identical,
        "gap_analysis_causes": causes
    }
    
    report_path = os.path.join(val_dir, 'audit_eval_report.json')
    samples_path = os.path.join(val_dir, 'decoded_samples.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    with open(samples_path, 'w') as f:
        json.dump(samples_to_save, f, indent=2)
        
    print(f"\nSaved report to {report_path}")
    print(f"Saved decoded samples to {samples_path}")

if __name__ == "__main__":
    main()
