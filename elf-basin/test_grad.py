import sys
sys.path.insert(0, '/home/snu/trishal_sanjay_proj/ELF/elf-basin/src')
sys.path.insert(0, '/home/snu/trishal_sanjay_proj/ELF/src')
import torch
from elfbasin.model.elf import ELFWrapper
w = ELFWrapper('ELF-B-de-en')
x = torch.randn(1, 128, 512, device='cuda', requires_grad=True)

print("x.requires_grad:", x.requires_grad)
x_input = torch.cat([x, torch.zeros_like(x)], dim=-1)
print("x_input.requires_grad:", x_input.requires_grad)

out, dec = w.model(x_input, torch.ones(1, device='cuda'), deterministic=True, decoder_step_active=True)
print("out.requires_grad:", out.requires_grad)
print("dec.requires_grad:", dec.requires_grad)

