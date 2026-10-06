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

def evaluate_tf(model, dataset, device):
    model.eval()
    bs = min(1000, len(dataset.embeddings))
    x0 = dataset.embeddings[:bs].to(device)
    times = dataset.insertion_times[:bs].to(device)
    targets = dataset.token_ids[:bs].to(device)
    
    with torch.no_grad():
        t = torch.ones((bs,), device=device)
        _, logits = model(x0, t, local_times=times, decoder_step_active=True)
        loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1), reduction='none').view(targets.shape)
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        
    return loss.mean().item(), acc

def main():
    device = get_device()
    set_seed(42)
    steps = 5000
    batch_size = 32
    
    out_dir = "task2a_results"
    ensure_dir(out_dir)
    
    sigmas = [0.0, 0.05, 0.1, 0.2]
    Ws = [4, 16]
    conditions = [
        ("global_time_only", "none", 0),
        ("continuous", "continuous", 0),
        ("lowrank_K4", "lowrank", 4),
    ]
    
    print("="*60)
    print("TASK 2a: SIGMA SWEEP")
    print("="*60)
    
    results = {}
    
    for W in Ws:
        results[W] = {}
        for sigma in sigmas:
            results[W][sigma] = {}
            dataset = OrderedAssemblyDataset(
                num_samples=5000, seq_len=64, vocab_size=256,
                embed_dim=512, W=W, seed=42, sigma=sigma
            )
            
            for name, mode, K in conditions:
                exp_name = f"W{W}_sig{sigma}_{name}"
                ckpt_dir = os.path.join(out_dir, "checkpoints", exp_name)
                
                model = EFM_Tiny(
                    vocab_size=256,
                    local_time_mode=mode,
                    local_time_K=K,
                    lowrank_basis="learned" if mode == "lowrank" else "fourier",
                ).to(device)
                
                local_time_training = "expansion" if mode != "none" else "none"
                train(
                    model=model, dataset=dataset, device=device,
                    max_steps=steps, batch_size=batch_size,
                    save_every=steps+1, log_every=1000,
                    checkpoint_dir=ckpt_dir, experiment_name=exp_name,
                    output_dir=out_dir, local_time_training=local_time_training,
                    use_amp=True
                )
                
                nll, acc = evaluate_tf(model, dataset, device)
                results[W][sigma][name] = {"nll": nll, "acc": acc}
                print(f"W={W} | Sigma={sigma} | {name:16s} => TF NLL: {nll:.4f}, TF Acc: {acc:.4f}")
                
                with open(os.path.join(out_dir, "summary.json"), "w") as f:
                    json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
