import torch
import torch.nn as nn

class DecoderAdapter(nn.Module):
    """
    Sensitivity-regularized decoder adapter.
    We apply a small adapter block to the latent `x` before passing it to the frozen native decoder.
    This effectively adapts the decode path without modifying the denoiser trunk.
    """
    def __init__(self, wrapper, r=256):
        super().__init__()
        self.wrapper = wrapper
        d = wrapper.d_model
        
        # A simple bottleneck adapter
        self.adapter = nn.Sequential(
            nn.Linear(d, r),
            nn.GELU(),
            nn.Linear(r, d)
        )
        
        # Initialize to identity (zero out the second linear layer)
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        
    def forward(self, x, sc_cfg=1.0):
        # Adapt the latent state
        x_adapted = x + self.adapter(x.float())
        # Pass through the native decoder directly to avoid @torch.no_grad in wrapper.decode
        B = x_adapted.shape[0]
        t_final = torch.ones((B,), dtype=x_adapted.dtype, device=x_adapted.device)
        sc_batch = (
            torch.full((B,), float(sc_cfg), dtype=x_adapted.dtype, device=x_adapted.device)
            if self.wrapper.config.num_self_cond_cfg_tokens > 0 else None
        )
        if self.wrapper.config.self_cond_prob > 0 or self.wrapper.config.num_self_cond_cfg_tokens > 0:
            x_input = torch.cat([x_adapted, torch.zeros_like(x_adapted)], dim=-1)
        else:
            x_input = x_adapted
            
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.wrapper.use_bf16):
            _, decoder_logits = self.wrapper.model(
                x_input, t_final, deterministic=True,
                self_cond_cfg_scale=sc_batch,
                decoder_step_active=True,
            )
        return decoder_logits.float()
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
