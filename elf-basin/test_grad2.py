import sys
sys.path.insert(0, '/home/snu/trishal_sanjay_proj/ELF/elf-basin/src')
sys.path.insert(0, '/home/snu/trishal_sanjay_proj/ELF/src')
import torch
from elfbasin.model.elf import ELFWrapper
w = ELFWrapper('ELF-B-de-en')
x = torch.randn(1, 128, 512, device='cuda', requires_grad=True)
logits = w.decode(x, sc_cfg=1.0)
print('logits grad_fn:', logits.grad_fn)
