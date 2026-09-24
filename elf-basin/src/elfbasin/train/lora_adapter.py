"""LoRA (Low-Rank Adaptation) for ELF decode-mode layers.

Instead of a small bottleneck applied to x before decode, LoRA modifies
the attention projections inside the transformer during decode mode.
This is far more expressive while staying under the 5M param cap.

Architecture:
  For each of the 12 ELF-B layers, we add LoRA to:
    - qkv projection (768 -> 2304): LoRA adds A(768xr) * B(rx2304)
    - out projection (768 -> 768):  LoRA adds A(768xr) * B(rx768)

  At r=32: 12 layers * (768*32 + 32*2304 + 768*32 + 32*768) * 4 bytes
         = 12 * (24576 + 73728 + 24576 + 24576) = 12 * 147456 = ~1.77M params
  Well under the 5M budget.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


class LoRALinear(nn.Module):
    """A LoRA adapter wrapping an existing frozen linear layer."""

    def __init__(self, original: nn.Linear, r: int = 32, alpha: float = 1.0):
        super().__init__()
        self.original = original
        in_features = original.in_features
        out_features = original.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, out_features))

        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original (frozen) output + LoRA delta
        out = self.original(x)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scaling
        return out + lora_out


class LoRADecoderAdapter(nn.Module):
    """Applies LoRA to all attention layers of the ELF model for decode mode.

    The model's original parameters stay frozen. Only the LoRA matrices are trained.
    The decode is done by calling the full model forward with decoder_step_active=True.
    """

    def __init__(self, wrapper, r: int = 32, alpha: float = 1.0):
        super().__init__()
        self.wrapper = wrapper
        self.r = r
        self.alpha = alpha

        # Collect all attention layers and wrap them with LoRA
        self.lora_modules: List[Tuple[str, LoRALinear]] = []

        model = wrapper.model
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                lora_qkv = LoRALinear(module.qkv, r=r, alpha=alpha)
                module.qkv = lora_qkv
                self.lora_modules.append((f"{name}.qkv", lora_qkv))

            if hasattr(module, 'proj') and isinstance(module.proj, nn.Linear):
                # 'proj' exists in Attention — but also in other modules.
                # Only wrap if parent is an Attention module.
                from modules.layers import Attention
                if isinstance(module, Attention):
                    lora_proj = LoRALinear(module.proj, r=r, alpha=alpha)
                    module.proj = lora_proj
                    self.lora_modules.append((f"{name}.proj", lora_proj))

        # Register all LoRA parameters so they appear in self.parameters()
        for i, (name, lora_mod) in enumerate(self.lora_modules):
            self.add_module(f"lora_{i}", lora_mod)

        # Freeze everything in the wrapper model except our LoRA params
        for param in wrapper.model.parameters():
            param.requires_grad = False
        for _, lora_mod in self.lora_modules:
            lora_mod.lora_A.requires_grad = True
            lora_mod.lora_B.requires_grad = True

    def forward(self, x, sc_cfg=1.0):
        """Decode x through the LoRA-augmented model."""
        B = x.shape[0]
        t_final = torch.ones((B,), dtype=x.dtype, device=x.device)
        sc_batch = (
            torch.full((B,), float(sc_cfg), dtype=x.dtype, device=x.device)
            if self.wrapper.config.num_self_cond_cfg_tokens > 0 else None
        )
        if self.wrapper.config.self_cond_prob > 0 or self.wrapper.config.num_self_cond_cfg_tokens > 0:
            x_input = torch.cat([x, torch.zeros_like(x)], dim=-1)
        else:
            x_input = x

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.wrapper.use_bf16):
            _, decoder_logits = self.wrapper.model(
                x_input, t_final, deterministic=True,
                self_cond_cfg_scale=sc_batch,
                decoder_step_active=True,
            )
        return decoder_logits.float()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def remove_lora(self):
        """Remove LoRA and restore original linear layers."""
        for _, lora_mod in self.lora_modules:
            if isinstance(lora_mod, LoRALinear):
                # Merge LoRA weights into original
                merged = lora_mod.original.weight.data + (
                    lora_mod.lora_B.data.T @ lora_mod.lora_A.data.T
                ) * lora_mod.scaling
                lora_mod.original.weight.data = merged
