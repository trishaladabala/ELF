"""Configuration for EFM pipeline experiments.

Extends the conceptual structure of ELF's Config / SamplingConfig with
EFM-specific settings (local-time mode, insertion head, expansion).
Uses plain dataclasses — no YAML dependency.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Architecture parameters for the EFM flow transformer."""

    # --- Model size presets ---
    # "tiny" = 4 layers × 256 hidden (Mac smoke tests, ~5M params)
    # "small" = 6 layers × 512 hidden (A4000 experiments, ~20M params)
    size: str = "tiny"

    # Explicit overrides (set automatically from `size` if left at 0).
    depth: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    # ELF-compatible settings.
    text_encoder_dim: int = 512   # T5-Small output dim
    bottleneck_dim: int = 128     # Text → hidden bottleneck
    num_time_tokens: int = 4      # Prefix time conditioning tokens
    num_self_cond_cfg_tokens: int = 0  # Disabled for EFM by default
    num_model_mode_tokens: int = 0
    vocab_size: int = 32128       # T5 vocab size
    gradient_checkpointing: bool = False

    def __post_init__(self):
        presets = {
            "tiny":  {"depth": 4,  "hidden_size": 256, "num_heads": 4},
            "small": {"depth": 6,  "hidden_size": 512, "num_heads": 8},
            "elf_b": {"depth": 12, "hidden_size": 768, "num_heads": 12},
        }
        if self.size in presets and self.depth == 0:
            for k, v in presets[self.size].items():
                setattr(self, k, v)
        if self.depth == 0:
            raise ValueError(f"Unknown model size '{self.size}' and no explicit depth set.")


@dataclass
class LocalTimeConfig:
    """Local-time conditioning configuration."""

    # Mode: "continuous" | "quantized" | "lowrank"
    mode: str = "continuous"

    # Number of quantization bins (quantized mode) or basis dimensions (lowrank mode).
    K: int = 16

    # Low-rank basis type: "fourier" or "learned"
    lowrank_basis: str = "fourier"

    # For quantized mode: whether to freeze the backbone and only train bin embeddings.
    quantize_freeze_backbone: bool = False


@dataclass
class InsertionConfig:
    """Insertion head configuration."""

    enabled: bool = True          # Whether to use learned insertion
    max_insert_per_gap: int = 4   # Max tokens to insert per gap
    hidden_dim_ratio: float = 0.25  # MLP hidden dim = hidden_size * ratio


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimizer.
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"      # "adamw"
    max_grad_norm: float = 1.0

    # Schedule.
    warmup_fraction: float = 0.1  # Fraction of total steps for warmup
    lr_schedule: str = "cosine"   # "cosine" or "linear"

    # Training budget.
    max_steps: int = 50000
    batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # Precision.
    use_amp: bool = True          # FP16 on CUDA, FP32 elsewhere
    amp_dtype: str = "float16"    # "float16" or "bfloat16"

    # Checkpointing.
    save_every_steps: int = 5000
    keep_last_n: int = 3
    log_every_steps: int = 100
    eval_every_steps: int = 5000

    # Loss weights.
    flow_loss_weight: float = 1.0
    decode_loss_weight: float = 0.5
    insertion_loss_weight: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration."""

    # Dataset name: "wikitext2" (smoke test) or "openwebtext"
    dataset: str = "wikitext2"

    # Sequence lengths.
    max_seq_length: int = 64
    initial_seq_length: int = 16  # Starting length for expansion experiments

    # T5 encoder.
    encoder_model_name: str = "t5-small"
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Generation / eval.
    gen_ppl_model: str = "gpt2-large"  # Model for computing Gen-PPL
    eval_batch_size: int = 8


@dataclass
class SamplingConfig:
    """Sampling (inference) configuration."""

    method: str = "ode"             # "ode" or "sde"
    num_steps: int = 32
    time_schedule: str = "logit_normal"
    sde_gamma: float = 1.5
    cfg_scale: float = 1.0
    self_cond_cfg_scale: float = 1.0
    num_samples: int = 512          # Number of samples for evaluation
    t_eps: float = 5e-2


@dataclass
class ExperimentConfig:
    """Top-level config combining all sub-configs."""

    # Sub-configs.
    model: ModelConfig = field(default_factory=ModelConfig)
    local_time: LocalTimeConfig = field(default_factory=LocalTimeConfig)
    insertion: InsertionConfig = field(default_factory=InsertionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Experiment metadata.
    experiment_name: str = "default"
    seed: int = 42
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    device: Optional[str] = None   # None = auto-detect

    def to_dict(self) -> dict:
        """Flatten config to a plain dict for logging."""
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Preset configs for common experiment types
# ---------------------------------------------------------------------------

def elf_baseline_config() -> ExperimentConfig:
    """Config for Stage 0: ELF-B baseline reproduction."""
    return ExperimentConfig(
        experiment_name="stage0_elf_baseline",
        model=ModelConfig(size="elf_b"),
        local_time=LocalTimeConfig(mode="continuous"),
        insertion=InsertionConfig(enabled=False),
        sampling=SamplingConfig(num_steps=32, method="ode", num_samples=512),
        data=DataConfig(dataset="openwebtext", max_seq_length=1024),
    )


def efm_tiny_config() -> ExperimentConfig:
    """Config for smoke tests on Mac (tiny model, WikiText-2)."""
    return ExperimentConfig(
        experiment_name="smoke_test",
        model=ModelConfig(size="tiny"),
        local_time=LocalTimeConfig(mode="continuous"),
        insertion=InsertionConfig(enabled=True),
        training=TrainingConfig(max_steps=100, batch_size=2, use_amp=False),
        sampling=SamplingConfig(num_steps=8, num_samples=4),
        data=DataConfig(dataset="wikitext2", max_seq_length=32),
    )


def efm_small_config(time_mode: str = "continuous", K: int = 16) -> ExperimentConfig:
    """Config for A4000 experiments (small model, OpenWebText)."""
    return ExperimentConfig(
        experiment_name=f"efm_{time_mode}_K{K}",
        model=ModelConfig(size="small", gradient_checkpointing=True),
        local_time=LocalTimeConfig(mode=time_mode, K=K),
        insertion=InsertionConfig(enabled=True),
        training=TrainingConfig(
            max_steps=50000, batch_size=4, use_amp=True,
        ),
        sampling=SamplingConfig(num_steps=32, num_samples=512),
        data=DataConfig(dataset="openwebtext", max_seq_length=64),
    )
