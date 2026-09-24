# Walkthrough: EFM Pipeline Implementation

## What Was Built

**19 files** created in [`efm_pipeline/`](file:///Users/adabalatrishal/7semminor/efm_pipeline):

| File | Purpose | New vs Reused |
|---|---|---|
| [`__init__.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/__init__.py) | Package init, wires ELF/src into sys.path | New |
| [`utils.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/utils.py) | Seed fixing, device detection, memory, AMP, timing | New |
| [`logger.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/logger.py) | CSV + JSON experiment logging (no WandB) | New |
| [`config.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/config.py) | Dataclass config (model/data/training/sampling) | New |
| [`local_time.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/local_time.py) | LocalTimeConditioner: continuous / quantized / lowrank | **New (core)** |
| [`insertion_head.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/insertion_head.py) | Per-gap insertion count predictor | **New (core)** |
| [`expand_op.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/expand_op.py) | Expand operator (loop + batched) | **New (core)** |
| [`elf_interface_audit.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/elf_interface_audit.py) | Traces ELF-B tensor shapes for wrapper design | New |
| [`efm_model.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/efm_model.py) | EFM wrapper: ELF blocks + expansion machinery | **New (core)** / Reuses ELF blocks |
| [`train_efm.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/train_efm.py) | Training script (all time modes) | New / Reuses ELF sampling |
| [`eval_runner.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/eval_runner.py) | Standardised evaluation (Gen-PPL, entropy, etc.) | New / Reuses ELF metrics |
| [`smoke_test.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/smoke_test.py) | Validates all modules (< 2 min on CPU) | New |
| [`experiments/stage0_reproduce_elf.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage0_reproduce_elf.py) | ELF-B baseline reproduction | New / Reuses ELF model+metrics |
| [`experiments/stage1_train_efm.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage1_train_efm.py) | Train continuous-time EFM | Delegates to train_efm.py |
| [`experiments/stage2_continuous_baseline.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage2_continuous_baseline.py) | Evaluate continuous EFM | New |
| [`experiments/stage3a_quantize_frozen.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage3a_quantize_frozen.py) | Frozen quantization sweep | New |
| [`experiments/stage3b_quantize_learned.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage3b_quantize_learned.py) | Learned quantized conditioning | New |
| [`experiments/stage4_lowrank.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage4_lowrank.py) | Low-rank τ compression | New |
| [`experiments/stage5_analysis.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/experiments/stage5_analysis.py) | Result aggregation and comparison tables | New |

## Bug Fix Applied

`ELFBlock` is defined in [`modules/model.py`](file:///Users/adabalatrishal/7semminor/ELF/src/modules/model.py#L18-L49), not `modules/layers.py`. Fixed the import in [`efm_model.py`](file:///Users/adabalatrishal/7semminor/efm_pipeline/efm_model.py#L42).

---

## Commands to Run

All commands are run from the project root: `cd /Users/adabalatrishal/7semminor`

### 0. Smoke Test (Mac, < 2 min)

```bash
python -m efm_pipeline.smoke_test
```

### 1. ELF Interface Audit (Mac, ~30s)

```bash
python -m efm_pipeline.elf_interface_audit
```

This outputs a JSON report to `results/elf_interface_audit.json`.

### 2. Stage 0 — ELF-B Baseline (needs GPU or ~16GB RAM, downloads ~3GB)

```bash
# On Mac (smaller batch, fewer samples):
python -m efm_pipeline.experiments.stage0_reproduce_elf \
    --num_samples 64 --batch_size 4 --num_steps 16

# On A4000:
python -m efm_pipeline.experiments.stage0_reproduce_elf \
    --num_samples 512 --batch_size 16 --num_steps 32
```

### 3. Stage 1 — Train Continuous EFM

```bash
# Short validation (Mac, ~5 min):
python -m efm_pipeline.experiments.stage1_train_efm \
    --model_size tiny --max_steps 1000 --batch_size 2 --seq_length 32 --dataset wikitext2

# Full run (A4000):
python -m efm_pipeline.experiments.stage1_train_efm \
    --model_size small --max_steps 50000 --batch_size 4 --seq_length 64 \
    --dataset openwebtext --use_amp
```

### 4. Stage 2 — Evaluate Continuous EFM

```bash
python -m efm_pipeline.experiments.stage2_continuous_baseline \
    --checkpoint checkpoints/stage1_continuous_baseline \
    --model_size tiny --num_samples 256
```

### 5. Stage 3a — Frozen Quantization Sweep

```bash
python -m efm_pipeline.experiments.stage3a_quantize_frozen \
    --checkpoint checkpoints/stage1_continuous_baseline \
    --model_size tiny --K_values 1,2,4,8,16,64 --num_samples 256
```

### 6. Stage 3b — Learned Quantization Sweep

```bash
python -m efm_pipeline.experiments.stage3b_quantize_learned \
    --checkpoint checkpoints/stage1_continuous_baseline \
    --model_size tiny --K_values 1,2,4,8,16 --finetune_steps 5000 \
    --dataset wikitext2
```

### 7. Stage 4 — Low-Rank Sweep

```bash
# Fourier basis:
python -m efm_pipeline.experiments.stage4_lowrank \
    --checkpoint checkpoints/stage1_continuous_baseline \
    --model_size tiny --K_values 1,2,4,8,16 --basis fourier \
    --finetune_steps 5000

# Learned basis:
python -m efm_pipeline.experiments.stage4_lowrank \
    --checkpoint checkpoints/stage1_continuous_baseline \
    --model_size tiny --K_values 1,2,4,8,16 --basis learned \
    --finetune_steps 5000
```

### 8. Stage 5 — Analysis

```bash
python -m efm_pipeline.experiments.stage5_analysis --results_dir results
```

---

## Recommended Run Order

1. **Smoke test** — verify all modules work
2. **Interface audit** — inspect ELF-B shapes (informational)
3. **Stage 0** — reproduce ELF-B baseline (reference point)
4. **Stage 1** (short) — train tiny EFM for 1k steps (verify learning works)
5. **Stage 1** (full) — train for 50k steps on A4000
6. **Stage 2** — evaluate the trained model
7. **Stage 3a** → **3b** → **4** — compression experiments
8. **Stage 5** — aggregate and compare all results
