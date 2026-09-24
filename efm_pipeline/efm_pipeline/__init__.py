"""EFM Pipeline — Compressed Local-Time Conditioning for Expanding Flow Maps.

This package implements the experimental pipeline for studying compressed
local-time conditioning in EFM-style variable-length flow models, using
ELF's pretrained components as the foundation.

Reused from ELF (ELF/src/):
    modules.layers   — ELFBlock, Attention, SwiGLU, RMSNorm, RoPE, TimestepEmbedder
    modules.model    — ELF (full model), ELF_B/M/L factories, decoder unembedding
    utils.*          — sampling, metrics, data, encoder, generation utilities

New in this package:
    local_time       — LocalTimeConditioner (continuous / quantized / lowrank)
    insertion_head   — Per-gap insertion count predictor
    expand_op        — Expand operator (insert noise at predicted positions)
    efm_model        — EFM wrapper combining ELF blocks + expansion machinery
"""

import sys
import os

# Make ELF source importable.
_ELF_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if os.path.isdir(_ELF_SRC) and _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)
