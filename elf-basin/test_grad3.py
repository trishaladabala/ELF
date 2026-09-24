import sys
sys.path.insert(0, '/home/snu/trishal_sanjay_proj/ELF/elf-basin/src')
sys.path.insert(0, '/home/snu/trishal_sanjay_proj/ELF/src')
import torch
from elfbasin.model.elf import ELFWrapper
w = ELFWrapper('ELF-B-de-en')
model = w.model

x = torch.randn(1, 128, 512, device='cuda', requires_grad=True)
B = x.shape[0]
sc_cfg = 1.0
t_final = torch.ones((B,), dtype=x.dtype, device=x.device)
sc_batch = torch.full((B,), float(sc_cfg), dtype=x.dtype, device=x.device)

x_input = torch.cat([x, torch.zeros_like(x)], dim=-1)

print("x_input requires_grad:", x_input.requires_grad)

with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=w.use_bf16):
    mode_tokens = model.model_mode_embed.weight.unsqueeze(0).expand(B, -1, -1)
    mode_gate = model.model_mode_gate.weight.unsqueeze(0).expand(B, -1, -1)
    active_gate = torch.where(torch.tensor([True], device='cuda')[:, None, None], mode_gate, torch.zeros_like(mode_gate))
    mode_tokens = mode_tokens * active_gate
    
    x2 = torch.cat([mode_tokens, x_input], dim=1)
    print("after concat 1 requires_grad:", x2.requires_grad)
    
    sc_tokens = model.self_cond_cfg_embed(sc_batch)[:, None, :].expand(-1, 4, -1)
    x3 = torch.cat([sc_tokens, x2], dim=1)
    print("after concat 2 requires_grad:", x3.requires_grad)
    
    time_tokens = model.time_embed(t_final)[:, None, :].expand(-1, 4, -1)
    x4 = torch.cat([time_tokens, x3], dim=1)
    print("after concat 3 requires_grad:", x4.requires_grad)
    
    x5 = model.text_proj(x4.float())
    print("after text_proj requires_grad:", x5.requires_grad)
    
    x6 = model.blocks[0](x5, rope_fn=model.feat_rope)
    print("after block0 requires_grad:", x6.requires_grad)

