"""EFM-style variable-length flow model.

Wraps ELF's Transformer blocks (Attention, SwiGLU, RMSNorm, RoPE) with
the new EFM expansion machinery (local-time conditioning, insertion head,
expand operator).

Architecture adapted to real ELF-B interfaces (see elf_interface_audit.py):

    Input pipeline:
        token_ids → T5 encoder (frozen, reused) → x₀ ∈ (B, S, 512)

    EFM model internals:
        x₀ → self_cond_proj if self-conditioned (2C → C)
           → BottleneckTextProj (C → bottleneck → hidden_size)
           + global time prefix tokens (from TimestepEmbedder)
           + local-time conditioning (added per-token after bottleneck)  [NEW]
           → N × ELFBlock (Attention + SwiGLU + RMSNorm, with RoPE)
           → FinalLayer (hidden → text_encoder_dim)           [flow output]
           → factored decoder (hidden → proj → unembed → vocab) [decode output]

    Expansion loop:
        InsertionHead predicts counts → ExpandOperator inserts noise
        → denoise → repeat until target length.

Reused from ELF:
    ELFBlock, Attention, SwiGLU, RMSNorm, RoPE, TimestepEmbedder,
    BottleneckTextProj, FinalLayer, decoder unembedding.

New:
    LocalTimeConditioner, InsertionHead integration, expand loop.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ELF components (imported via efm_pipeline/__init__.py sys.path).
from modules.model import ELFBlock
from modules.layers import (
    TimestepEmbedder, TextRotaryEmbeddingFast,
    BottleneckTextProj, FinalLayer, RMSNorm,
    _make_linear, NORMAL_INIT_002, DEFAULT_KERNEL_INIT,
    DEFAULT_BIAS_INIT, ZERO_INIT, rotate_half,
)

# EFM-specific components.
from efm_pipeline.local_time import LocalTimeConditioner
from efm_pipeline.insertion_head import InsertionHead
from efm_pipeline.expand_op import ExpandOperator


class VariableLengthRoPE(nn.Module):
    """Wraps ELF's RoPE to support dynamic sequence lengths during expansion."""
    def __init__(self, base_rope: TextRotaryEmbeddingFast):
        super().__init__()
        self.base_rope = base_rope

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        S = t.shape[2]
        freqs_cos = self.base_rope.freqs_cos[:S].to(t.dtype)
        freqs_sin = self.base_rope.freqs_sin[:S].to(t.dtype)
        return t * freqs_cos + rotate_half(t) * freqs_sin


class EFMModel(nn.Module):
    """EFM-style flow model with variable-length expansion.

    Combines ELF's Transformer blocks with EFM's expansion machinery.
    Supports continuous, quantized, and low-rank local-time conditioning.

    Args:
        text_encoder_dim:   T5 encoder output dimension (512 for T5-Small).
        max_length:         Maximum sequence length.
        hidden_size:        Transformer hidden dimension.
        depth:              Number of ELFBlock layers.
        num_heads:          Number of attention heads.
        mlp_ratio:          FFN expansion ratio.
        bottleneck_dim:     Bottleneck dimension for text projection.
        num_time_tokens:    Number of global time prefix tokens.
        vocab_size:         Vocabulary size (for decoder head).
        local_time_mode:    "continuous", "quantized", or "lowrank".
        local_time_K:       K parameter for quantized/lowrank modes.
        lowrank_basis:      "fourier" or "learned" (for lowrank mode).
        insertion_enabled:  Whether to use learned insertion.
        max_insert_per_gap: Max tokens to insert per gap.
        gradient_checkpointing: Enable gradient checkpointing.
        attn_drop:          Attention dropout.
        proj_drop:          Projection dropout.
    """

    def __init__(
        self,
        text_encoder_dim: int = 512,
        max_length: int = 128,
        hidden_size: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        vocab_size: int = 32128,
        local_time_mode: str = "continuous",
        local_time_K: int = 16,
        lowrank_basis: str = "fourier",
        insertion_enabled: bool = True,
        max_insert_per_gap: int = 4,
        gradient_checkpointing: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.bottleneck_dim = bottleneck_dim
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.insertion_enabled = insertion_enabled

        # --- Reused from ELF ---

        # Bottleneck text projection: text_encoder_dim → bottleneck → hidden_size.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Global time embedding (scalar t → prefix tokens).
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)
        self.num_time_tokens = num_time_tokens

        # RoPE (1D positional encoding for text).
        head_dim = hidden_size // num_heads
        base_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=num_time_tokens,
        )
        self.feat_rope = VariableLengthRoPE(base_rope)

        # Transformer blocks (identical to ELF).
        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        # Flow matching output head.
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # Factored decoder unembedding: hidden → text_encoder_dim → vocab.
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, text_encoder_dim))
        self.proj_bias = nn.Parameter(torch.empty(text_encoder_dim))
        self.unembed_kernel = nn.Parameter(torch.empty(text_encoder_dim, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

        # --- New EFM components ---

        # Per-token local-time conditioning.
        self.local_time_conditioner = LocalTimeConditioner(
            hidden_size=hidden_size,
            mode=local_time_mode,
            K=local_time_K,
            lowrank_basis=lowrank_basis,
        )

        # Insertion head (predicts per-gap counts).
        if insertion_enabled:
            self.insertion_head = InsertionHead(
                hidden_size=hidden_size,
                max_insert_per_gap=max_insert_per_gap,
            )
        else:
            self.insertion_head = None

        # Expand operator (inserts noise at predicted positions).
        self.expand_op = ExpandOperator(
            embed_dim=text_encoder_dim,
            max_total_length=max_length,
        )

    def build_time_prefix(self, t: torch.Tensor) -> torch.Tensor:
        """Build global time conditioning prefix tokens.

        Args:
            t: (B,) — global diffusion time.

        Returns:
            prefix: (B, num_time_tokens, hidden_size)
        """
        B = t.shape[0]
        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix = self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        return prefix

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        local_times: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the EFM model.

        Args:
            x:                (B, S, text_encoder_dim) or (B, S, D) — input embeddings.
            t:                (B,) — global diffusion timestep.
            local_times:      (B, S) — per-token local times τ_i ∈ [0,1].
                              If None, all tokens get τ=1.0 (fully known).
            attention_mask:   (B, S) — 1=valid, 0=padded.
            decoder_step_active: If True, also compute decoder logits.
            deterministic:    If True, disable dropout.

        Returns:
            flow_output:      (B, S, text_encoder_dim) — predicted clean x₀.
            decoder_logits:   (B, S, vocab_size) or None.
        """
        B, S = x.shape[:2]

        # --- Bottleneck projection ---
        with torch.amp.autocast('cuda', enabled=False):
            x = self.text_proj(x.float())  # (B, S, hidden_size)

        # --- Local-time conditioning (NEW) ---
        # Add per-token local-time embeddings to the hidden states.
        if local_times is not None:
            lt_embed = self.local_time_conditioner(local_times)  # (B, S, hidden_size)
            x = x + lt_embed

        # --- Global time prefix ---
        prefix = self.build_time_prefix(t)  # (B, num_time_tokens, hidden_size)
        x = torch.cat([prefix, x], dim=1)   # (B, prefix + S, hidden_size)

        # Update attention mask for prefix.
        if attention_mask is not None:
            prefix_mask = torch.ones(B, self.num_time_tokens,
                                      dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # --- Transformer blocks ---
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                def _fwd(h, blk=block):
                    return blk(h, rope_fn=self.feat_rope, attention_mask=attention_mask,
                               deterministic=deterministic)
                x = checkpoint(_fwd, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask,
                           deterministic=deterministic)

        # Strip prefix tokens.
        x = x[:, self.num_time_tokens:]  # (B, S, hidden_size)

        # --- Output heads ---
        with torch.amp.autocast('cuda', enabled=False):
            # Decoder logits (if requested).
            decoder_logits = None
            if decoder_step_active:
                x_f32 = x.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias

            # Flow matching output.
            flow_output = self.final_layer(x.float())

        return flow_output, decoder_logits

    def predict_insertions(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict insertion counts using the insertion head.

        Args:
            hidden_states: (B, S, hidden_size) — from a forward pass
                           (before stripping prefix).

        Returns:
            counts: (B, S-1) — integer insertion counts.
        """
        if self.insertion_head is None:
            raise ValueError("Insertion head is disabled.")
        return self.insertion_head.predict_counts(hidden_states)

    def get_insertion_rates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get raw insertion rates (for training loss).

        Args:
            hidden_states: (B, S, hidden_size)

        Returns:
            rates: (B, S-1) — continuous non-negative rates.
        """
        if self.insertion_head is None:
            raise ValueError("Insertion head is disabled.")
        return self.insertion_head(hidden_states)


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def EFM_Tiny(**kwargs):
    """Tiny model for Mac smoke tests: 4 layers, 256 hidden, ~5M params."""
    defaults = dict(
        depth=4, hidden_size=256, num_heads=4,
        bottleneck_dim=64, num_time_tokens=2,
    )
    defaults.update(kwargs)
    return EFMModel(**defaults)


def EFM_Small(**kwargs):
    """Small model for A4000 experiments: 6 layers, 512 hidden, ~36M params."""
    defaults = dict(
        depth=6, hidden_size=512, num_heads=8,
        bottleneck_dim=128, num_time_tokens=4,
    )
    defaults.update(kwargs)
    return EFMModel(**defaults)


def EFM_Base(**kwargs):
    """Base model matching ELF-B scale: 12 layers, 768 hidden, ~105M params."""
    defaults = dict(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, num_time_tokens=4,
        gradient_checkpointing=True,  # Required to fit on 16GB A4000
    )
    defaults.update(kwargs)
    return EFMModel(**defaults)


EFM_models = {
    "EFM-Tiny": EFM_Tiny,
    "EFM-Small": EFM_Small,
    "EFM-Base": EFM_Base,
}
