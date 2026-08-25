# ELF Checkpoint Availability Report

**Date:** 2026-07-20  
**Purpose:** Determine whether released ELF checkpoints exist before planning real-scale replication.

## Result: Checkpoints Are Publicly Available

The ELF paper's GitHub repository (https://github.com/lillian039/ELF) links to pre-trained checkpoints hosted on HuggingFace under the [`embedded-language-flows`](https://huggingface.co/embedded-language-flows) organization.

### Unconditional Generation (OpenWebText)

| Model | Parameters | HuggingFace | SDE Steps | Gen. PPL ↓ | Entropy ↑ |
|-------|-----------|-------------|-----------|-----------|-----------|
| **ELF-B** | 105M | [ELF-B-owt](https://huggingface.co/embedded-language-flows/ELF-B-owt) | 32 | 24.1 | 5.15 |
| **ELF-M** | 342M | [ELF-M-owt](https://huggingface.co/embedded-language-flows/ELF-M-owt) | 64 | 21.7 | 5.18 |
| **ELF-L** | 652M | [ELF-L-owt](https://huggingface.co/embedded-language-flows/ELF-L-owt) | 64 | 23.3 | 5.28 |

### Conditional Generation (ELF-B only)

| Task | HuggingFace | Metric | Score |
|------|-------------|--------|-------|
| WMT14 De-En | [ELF-B-de-en](https://huggingface.co/embedded-language-flows/ELF-B-de-en) | BLEU ↑ | 26.4 |
| XSum | [ELF-B-xsum](https://huggingface.co/embedded-language-flows/ELF-B-xsum) | ROUGE-1/2/L ↑ | 36.0/12.2/27.8 |

### Codebase Details

- **Primary implementation:** JAX/TPU (main branch)
- **PyTorch version:** Available on [`pytorch_elf`](https://github.com/lillian039/ELF/tree/pytorch_elf) branch
- **Distillation:** Available on [`distillation`](https://github.com/lillian039/ELF/tree/distillation) branch
- **Encoder:** Uses a frozen T5-small encoder (assumed separately downloaded via HuggingFace `transformers`)
- **Paper results computed on:** TPU v5p-64

### Implications for This Project

1. **No training required for real-scale replication.** We can download ELF-B-owt (105M params) and run our corrected-SDE vs ODE comparison directly on the released checkpoint.
2. **The PyTorch branch** should be used for compatibility with our existing MPS-based harness. Need to verify whether the HuggingFace weights are in JAX or PyTorch format (likely JAX `.msgpack`, requiring conversion).
3. **The ~840x scale jump** from our toy model (275K params) to ELF-B (105M params) can now be tested directly rather than extrapolated from intermediate-scale trends.
4. **Memory consideration:** ELF-B at 105M params is ~420MB in float32. This should fit in MPS memory for inference, though generation with N=512 samples may need batching.
