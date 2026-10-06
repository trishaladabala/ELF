#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Tiny
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation

def test_tiny(W):
    device = get_device()
    seed = 42
    set_seed(seed)
    
    out_dir = f"tiny_pilot_W{W}"
    ensure_dir(out_dir)
    
    dataset = OrderedAssemblyDataset(num_samples=5000, seq_len=64, vocab_size=256, embed_dim=512, W=W, seed=seed)
    model = EFM_Tiny(vocab_size=256, local_time_mode="continuous").to(device)
    
    print(f"\nTraining EFM_Tiny Continuous W={W}...")
    train(
        model=model,
        dataset=dataset,
        device=device,
        max_steps=5000,
        batch_size=64,
        save_every=1000,
        log_every=500,
        checkpoint_dir=f"{out_dir}/ckpt",
        experiment_name=f"tiny_W{W}",
        output_dir=out_dir,
        local_time_training="expansion",
        use_amp=True
    )
    
    model.eval()
    
    print(f"\nEvaluating EFM_Tiny Continuous W={W}...")
    # Evaluate generation metrics
    generated_ids = generate_synthetic_samples(
        model=model, dataset=dataset, device=device,
        num_samples=1000, local_time_mode="expansion", W=W
    )
    
    metrics = full_evaluation(
        generated_token_ids=generated_ids.cpu(),
        true_token_ids=dataset.token_ids[:1000].cpu(),
        wave_ids=dataset.wave_ids[:1000].cpu(),
        vocab_size=256,
    )
    
    # Evaluate exact Teacher-Forced Exact Match and NLL
    bs = 64
    nlls = []
    exact_matches = []
    with torch.no_grad():
        for i in range(0, 1000, bs):
            z = dataset.embeddings[i:i+bs].to(device)
            targets = dataset.token_ids[i:i+bs].to(device)
            times = dataset.wave_ids[i:i+bs].to(device).float() / max(W - 1, 1)
            
            _, logits = model(z, torch.ones(len(z), device=device), local_times=times, decoder_step_active=True)
            loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1), reduction='none').view(targets.shape)
            nlls.append(loss.mean().item())
            
            preds = logits.argmax(dim=-1)
            exact_matches.append((preds == targets).float().mean().item())
            
    nll = sum(nlls) / len(nlls)
    tf_acc = sum(exact_matches) / len(exact_matches)
    
    print(f"Results for W={W}:")
    print(f"  Teacher-Forced NLL: {nll:.4f}")
    print(f"  Teacher-Forced Exact Match: {tf_acc:.4f}")
    print(f"  Dependency Accuracy: {metrics['overall_dep_acc']:.4f}")
    print(f"  Generated Exact Match: {metrics['overall_exact_match']:.4f}")

if __name__ == "__main__":
    test_tiny(4)
    test_tiny(16)
