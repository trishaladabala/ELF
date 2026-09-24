# Executive Summary

This report describes a complete pipeline for **compressed local-time conditioning in Expanding Flow Maps (EFM) and transfer to continuous Embedded Language Flows (ELF)**, tailored to a single RTX A4000 (16 GB) setup. We outline software requirements, data sources, and step-by-step instructions (with commands and configs) for each experiment stage. In brief, the recommended workflow is:

- **Stage 1 (Screening):** Use the released discrete EFM model to test alternative insertion schedules and simple compression (quantization) of the local-time field. This requires only inference, not retraining, and is very cheap.
- **Stage 2 (Quantization and Low-Rank Tests):** Freeze the EFM backbone and replace its full local-time vector with a quantized or low-rank representation. Fine-tune only the time-conditioning module. Measure how much conditioning capacity can be removed before generation degrades.
- **Stage 3 (Adaptive Expansion):** Implement an uncertainty-driven expansion schedule (a fixed schedule vs. an entropy-based schedule) with the same EFM checkpoint.
- **Stage 4 (Conditional Noise Ablation):** Enable or disable a learnable noise generator in the expand operator and compare performance.
- **Stage 5 (Controlled ELF Baselines):** Implement *append-only* and *insert-only* fixed-canvas baselines using frozen T5 embeddings, to provide reference curves under continuous representations.
- **Stage 6 (Continuous EFM with Compression):** Build a **small continuous-flow model** that operates on *contextual* embeddings (e.g. from a frozen T5 encoder). Repeat insertion-generation experiments with compressed time conditioning (quantized/low-rank) and compare to uncompressed.
- **Stage 7 (Analysis and Figures):** Collect metrics (generative perplexity, sample entropy, decoder confidence, etc.), log with WandB, and plot results (including a Gen-PPL vs. Entropy trade-off curve, compression vs. quality graph, and an experiment matrix table). We include Mermaid diagrams for the pipeline and timeline.

This plan prioritizes **feasibility on an A4000**: most heavy lifting is limited to small flow models or frozen encoders. Retraining is only done on the compact time-conditioning or insertion modules until Stage 6, where the *first full training run* of a continuous flow may be attempted. We emphasize reproducible configuration (fixed seeds, deterministic flags) and provide exact commands, hyperparameters, and expected resource usage. When possible, we cite the EFM and ELF papers for reference to methods and metrics.

---

## 1. Environment Setup

1. **OS and system:** Ubuntu 22.04 LTS (tested) or similar Linux. Ensure you have CUDA drivers and an RTX A4000 installed.  
2. **Python and drivers:** Use Python 3.10 (or 3.11). Install CUDA 11.7 (A4000 supports up to 11.8) and cuDNN accordingly.  
3. **Package manager:** We recommend using Conda or venv. Example with Conda:

   ```bash
   conda create -n efm_elf python=3.10 -y
   conda activate efm_elf
   ```

4. **PyTorch and CUDA:** Install PyTorch 2.x (compatible with CUDA 11.7/11.8) and related libraries. For example, via Conda:

   ```bash
   conda install -y pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c conda-forge
   ```

   Confirm GPU availability:

   ```python
   import torch
   print(torch.cuda.get_device_name(0))
   print(torch.cuda.is_available())
   ```

5. **Other Python libraries:** We will need Hugging Face Transformers, tokenizers, and possibly `diffusers` or custom code. Install:

   ```bash
   pip install transformers==4.42.0 accelerate datasets wandb
   pip install torchtyping tqdm regex
   ```

   *(We pin to versions known stable in mid-2026; adjust if needed.)* The ELF requirements (JAX-based) are not directly used, but we follow their environment guidelines. We will also use PyTorch's mixed precision (AMP) and gradient checkpointing as needed to fit in 16GB.

6. **WandB (or MLflow) setup:** We use Weights & Biases for logging. Sign up at [wandb.ai](https://wandb.ai) and note your API key. In code we will do `wandb.login()` or set environment `WANDB_API_KEY`. You may disable logging (`wandb.init(..., mode="disabled")`) for quick tests.

7. **Git repositories:** Clone the necessary code:

   - **ELF code (PyTorch branch):**  
     ```bash
     git clone https://github.com/lillian039/ELF.git
     cd ELF
     git checkout pytorch_elf
     pip install -r requirements.txt
     ```  
     (ELF’s PyTorch branch implements flow models and T5 embedding use. Note: If access issues occur, try `gh auth login` or use the PyPI package if one is released. The tags/commits aren’t finalized, so we use `main` or the default branch of `pytorch_elf`.)

   - **EFM code:** The official EFM repository by Tang & Chatterjee is not yet publicly fleshed out (it only contains a README). However, we can emulate EFM behavior by modifying flow models. For now, assume we have an EFM checkpoint on HuggingFace or given by the authors. (If none is available, simulate by using a simple flow on one-hot token simplex as a stand-in.) The pipeline will indicate where the checkpoint should be placed.

8. **Hardware config for reproducibility:**  
   - Set `torch.backends.cudnn.deterministic = True` and `torch.use_deterministic_algorithms(True)` at the top of scripts.  
   - Fix seeds for Python, NumPy, PyTorch (see below in Reproducibility).  
   - For FP16 training, be aware of non-determinism; if absolute reproducibility is required, use FP32 or set `torch.backends.cudnn.benchmark=False`.  
   - We'll record all software versions via `pip freeze > requirements.lock` for reference.

**References:** The EFM authors provide high-level code directions but no ready-to-run scripts. The ELF repo shows example commands (pip installs, wandb) which we adapt to PyTorch.

---

## 2. Data Preparation

We focus on language data, using **One Billion Word (OWT)** or **OpenWebText** (OWT or OWT as in ELF) as our text corpus (unconditional modeling). Optionally, use smaller subsets for quick tests.

- **Dataset:** We use HuggingFace Datasets. For unconditional text:
  - *Example:* `openwebtext` (the Pile or OpenWebText from HF) is convenient. 
  - Alternatively, Wikipedia or WikiText-103 can be used. Ensure it’s tokenized consistently with T5. ELF uses `embedded-language-flows/openwebtext-t5` (pre-tokenized with T5-small).

- **Preprocessing:** Use the T5 tokenizer to map raw text to input IDs. For ELF-like experiments, we need fixed-length sequences (max length 32, 64, or 128). We will pre-tokenize:
  
  ```python
  from transformers import T5Tokenizer
  tok = T5Tokenizer.from_pretrained("t5-small")
  def tokenize_batch(examples):
      return tok(examples["text"], truncation=True, max_length=64, padding="max_length")
  dataset = datasets.load_dataset("openwebtext", split="train")
  dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
  dataset.set_format(type="torch", columns=["input_ids"])
  ```
  
  Save the dataset to disk (`dataset.save_to_disk("data/openwebtext_tok")`) so that it can be reused without re-tokenization. ELF’s README notes exactly this pipeline.
  
- **T5 encoder:** For continuous-embedding experiments, we use a *frozen* T5 model to map tokens to embeddings. The ELF work uses T5-Small as the encoder (because T5-Base might be too large). To save memory, we can precompute T5-Small's outputs for all sequences (if few enough) or use them on-the-fly. Probably we will feed token IDs through `T5EncoderModel.from_pretrained("t5-small")` to get embeddings of shape `(batch, seq_len, d_model)`.

- **Batching:** Experiments will use short sequences (16–32 tokens) initially. We expect to increment to 64–128 as needed. Keep batch size small (1–8) to fit in 16GB. For instance, T5-Small encoder + flow with `seq_len=64, batch=4` uses ~12 GB (approx). Gradient checkpointing can allow larger models or longer sequences.

No additional dataset references needed; details follow ELF’s approach.

---

## 3. Baseline and Control Implementations

We implement a series of baselines for comparison:

- **Project 1 (Fixed-Length Flow Baseline):** A simple flow matching model (small transformer) that takes a *fixed-length* continuous embedding sequence (no expansion) and learns to denoise it. This is necessary as a control but not a core result. Use the same data and architecture as in Stage 6, but never increase sequence length. We will measure its Gen-PPL to compare to expanding models.

- **Project 2 (Append-Only Expansion Baseline):** Implement an expansion rule that always appends noise at the *end* of the sequence (no learned insertion) and then denoises. For discrete data, this means generating text by appending blank slots. Since we operate in continuous embedding space, we can simulate by appending random Gaussian embeddings and then denoising. This checks whether expanding by appending (a special case of EFM’s expand operator) works at all. The EFM paper explicitly includes concatenation/append expansion as a special case. In practice, we will take a partial sequence, append `k` Gaussian vectors at the end, and run the flow on the extended sequence. This baseline has trivial schedule (always extend to final length in one go), so we skip schedule and only do insertion testing.

- **Encoder-Consistency Control:** As a check, we may run "prefix reconstruction": take a prefix of text, then later produce the full text, and compare the T5 embeddings of tokens before and after. This reveals the representation inconsistency problem. For example, embed "The cat" and "The cat sat on the mat", and measure embedding shifts of "cat". This is an analysis step rather than a training loop.

Each baseline will be quick to implement. The code structure will separate the *expansion rule* (fixed, learned) from the *denoising network* (flow transformer). 

References: EFM defines fixed and learned expand operators. The “append-only” case is covered as a trivial expand where new coordinates have no context effect. We use it only as a sanity check (low novelty).

---

## 4. Experiment 1 – Schedule Optimization (Cheap Screening)

**Goal:** Test whether EFM’s default local-time schedule is suboptimal, and whether gentler (or sharper) schedules improve sample quality. This is entirely *inference-only* using a pretrained EFM model.

1. **EFM Checkpoint:** Obtain the released EFM language model checkpoint. If not available, simulate by training a small EFM-style model on toy data (beyond scope). We’ll assume a checkpoint named `efm_text_owt` is placed in `checkpoints/EFM` (16GB should handle sampling).

2. **Schedules to Test:** The schedule determines insertion times for tokens. Common choices: *linear* (`t_ins = Uniform(0,1)` as EFM), *quadratic* (`sqrt(t)` or `t^2` on local time), or *adaptive* (as below). We parameterize a family $\alpha(t)$ as in the new report: e.g. $\alpha(t) = t^p$ with $p \in \{0.5, 1.0, 2.0\}$. Also include EFM’s original as baseline ($p=1$). 

3. **Inference Setup:** Freeze the EFM model (generator) and run it with different schedules:
   ```bash
   # Example pseudo-command (replace with actual API call)
   python run_efm.py --checkpoint checkpoints/EFM/efm_text_owt \
       --output_dir outputs/schedule_test --schedule_alpha 0.5 1.0 2.0
   ```
   The script `run_efm.py` loads the checkpoint and overrides the local-time sampling distribution by raising it to the given alpha before insertion.  
   If no official code, this involves rescaling insertion times: if EFM normally sets $t_{\text{ins}} = U(0,1)$, we can use $U^\alpha$ to bias distribution. 

4. **Metrics:** For each schedule, generate **N=1000 samples** (or as many as feasible). Compute generative perplexity and entropy (see Evaluation Metrics below). Lower Gen-PPL and reasonable entropy is better. Plot Gen-PPL vs. Entropy for each schedule. (Expect monotonic schedule might help as per [新 report], but check).  

5. **Analysis:** Identify if any schedule (e.g. $\alpha>1$ or $<1$) meaningfully improves quality. If *all* perform similarly, schedule may be unimportant. If one is clearly best, adopt it for fine-tuning in Stage 2.

This step is **very cheap** (no training, just sampling). It provides evidence whether schedule tuning is worthwhile before retraining.

> **Output:** Table or graph of Schedule Parameter $\alpha$ vs. Gen-PPL and Entropy. A sample generation figure (e.g., partial vs. final sequences to illustrate timing).

**Citation:** EFM uses per-token insertion times $t_{\rm ins}$ and local clocks. We are testing the effect of modifying that schedule, an idea inspired by the new notes. No direct citations exist for this specific schedule hack, but the concept builds on [33†L123-L127]’s local-time framework.

---

## 5. Experiment 2 – Local-Time Quantization

**Goal:** Determine how much of EFM’s per-coordinate local-time vector is essential. Replace the continuous local-time value $t_i$ with a **quantized integer bin** $q_i \in \{1,\dots,K\}$, for $K=1,2,4,8,16,\infty$ (where $\infty$=no quantization). This tests if a low-resolution time field suffices.

1. **Setup:** Starting from the best schedule (or original) from Exp 1, modify the EFM model’s input. Instead of feeding each coordinate’s normalized local time (a real in [0,1]), feed a categorical or one-hot encoding of its *bin* $q_i = \lceil K \cdot t_i \rceil$ for given $K$. For $K=1$, all tokens have the same time indicator (no time info). 

2. **Inference Test:** As in Stage 1, run inference with the EFM checkpoint, overriding its time input:
   ```bash
   python run_efm_quant.py --checkpoint checkpoints/EFM/efm_text_owt \
       --quantize_bins 1 2 4 8 16 64
   ```
   Each `--quantize_bins K` sets $t_i \to \lfloor t_i \cdot K\rfloor$ (or similar). The model’s expansion transformer is adjusted so that instead of a float $t_i$, it sees an embedding for the bin index. (This may require a small code change: we freeze all weights except a new embedding layer for $K$ bins.)

3. **Initial Inference (Frozen Backbone):** First, do *frozen* inference: feed quantized times without retraining. This shows how robust the pretrained model is to time compression.

4. **Fine-Tuning:** For those $K$ values where frozen performance drops, **fine-tune** the time-conditioning part. Specifically, freeze the main flow-transformer weights and only retrain a small module that maps $q_i$ to the needed form. For example, append a small MLP that turns the bin index (or one-hot) into a scale factor. Train on the standard objective (flow map consistency) for a few epochs (say 10k steps) with a small learning rate (1e-5 to 1e-4). Use a batch size of 2–4 with gradient accumulation if needed.

5. **Metrics:** For each $K$, compute Gen-PPL and Entropy. Plot **Quality vs. Conditioning Complexity**: e.g. $K$ (or #dimensions used) on x-axis and Gen-PPL on y-axis. A sharp increase in PPL as $K$ decreases would show high dependence on fine time values. If quality stays flat down to small $K$, we have a big win (less conditioning needed). According to the new report, this is expected to be very informative.

6. **Data:** Use the same dataset (OWT) and generation procedure as before.

> **Output:** A figure/table of $K$ vs. Gen-PPL and entropy. Observe, for example, that if $K=4$ gives nearly the same quality as $K=\infty$, most time precision is wasted.

**Citation:** This experiment directly tests the necessity of the per-coordinate local-time field introduced by EFM. We leverage the idea of quantizing the time feature as suggested in the attached proposal (upcoming experiment). This approach is novel (no prior paper explicitly quantized local-time bins) but based on [33†L123-L127]’s definition of local clocks.

---

## 6. Experiment 3 – Low-Rank Local-Time Conditioning (Core Idea)

**Goal:** Replace the full $d$-dimensional time vector $\{t_i\}_{i=1}^d$ with a *low-dimensional embedding*. Concretely, use a learned basis $\{\phi_k(i)\}_{k=1}^K$ to approximate $t_i \approx \sum_{k=1}^K a_k\, \phi_k(i)$. This tests whether the EFM’s local-time field has low-dimensional structure that a small K can capture.

1. **Design:** We implement a small neural network (or even a fixed basis) that maps a token index $i$ to a $K$-dimensional vector $\phi(i)$, and weights $a_k$ that combine them. For simplicity, one can choose $\phi_k(i) = \sin(i\cdot\omega_k)$ (a Fourier basis) or learnable positional embeddings. Then $z_i = [\phi_1(i),…,\phi_K(i)]$ is the compressed time embedding.

2. **Integration:** Modify the EFM model to remove the original $d$ inputs $t_i$ and instead take a single shared $K$-dim vector. Concretely:
   - During inference (and fine-tuning), for each insertion event, compute the insertion position indices and apply the same $\phi$ basis to them. Use fixed or learnable $\{\phi_k\}$. 
   - The EFM’s transport network will now be conditioned on $[a_1\phi_1(i),…,a_K\phi_K(i)]$ instead of $t_i$. If $K$ is small (e.g. 4–8), the conditioning overhead is greatly reduced.

3. **Training:** Fine-tune the time-conditioning parameters (the $\phi$ basis and/or weights $a_k$) on the training objective (consistency/denoising) while keeping the rest of the model fixed. Alternatively, train the entire expansion head anew with frozen flow.

4. **Run for various $K$:** As with quantization, try $K=1,2,4,8,16$. (Note $K=1$ collapses all times to a single sinusoidal with amplitude; $K=\infty$ can be the original.)

5. **Metrics:** Again measure Gen-PPL and entropy for each $K$. We expect performance to drop slowly as $K$ reduces, if any low-rank structure exists. The optimal $K$ vs. quality curve is the main result. A flat curve up to small $K$ is a very strong finding.

6. **Comparison:** Compare quantization vs. low-rank. They address similar goals: compressing the time signal. Likely low-rank (which can exploit temporal smoothness) will outperform coarse quantization in retaining quality.

> **Output:** A primary figure showing generation quality vs. $K$ for low-rank conditioning, alongside the quantization results. Possibly also a visualization of learned basis $\phi_k$ if insightful.

**Citation:** This directly implements the report’s proposed “low-rank local-time” idea. By testing whether $\{t_i\}$ can be approximated by a small number of basis functions, we address the hypothesis raised in the report that the local-time field has exploitable structure. The EFM formalism implies each $t_i$ is used separately, so reducing dimensionality is a novel contribution.

---

## 7. Experiment 4 – Adaptive Expansion Scheduling

**Goal:** Replace the fixed insertion schedule with a schedule driven by model “uncertainty” or entropy. The idea: instead of pre-specifying *when* the model should expand, let it decide to insert when its predictions become uncertain.

1. **Uncertainty Criterion:** During sampling (denoising), track the model’s token prediction entropy (or max probability) at each potential insertion point. If entropy > threshold (or max-prob < threshold), trigger an expansion event sooner; else wait longer. A simple rule: at each timestep, compute the average per-token softmax entropy; if it exceeds a preset fraction of the maximum possible entropy, insert new tokens now.

2. **Implementation:** Starting with a frozen EFM checkpoint, simulate generation with this rule:
   ```bash
   python run_efm.py --checkpoint checkpoints/EFM/efm_text_owt \
       --adaptive_expansion  \
       --entropy_threshold 0.8
   ```
   This calls the model step-by-step: after each denoising step, measure entropy and decide if it’s time to expand the sequence further. Because EFM normally alternates expand/denoise in fixed increments, this requires a small loop: do denoise, check, expand if needed, repeat.

3. **Variations:** Try a few thresholds (e.g. 0.5, 0.8, 0.95 of max entropy) and compare to the fixed schedule (from Stage 1). We might also try a learned scheduler (train a small network to predict insertion times conditioned on hidden state), but that is more complex.

4. **Comparison Baseline:** Compare to the best fixed schedule found in Exp 1. This is still an inference test (no retraining of flow/insertion head). It checks if model-driven scheduling improves Gen-PPL or other metrics.

5. **Metrics:** As usual, compute Gen-PPL, entropy, and track the average sequence length or steps used. We expect adaptive to potentially achieve similar quality with fewer expansion events.

> **Output:** A table comparing fixed vs. adaptive schedules (metric: Gen-PPL, Entropy, Avg. final length). A short example dialogue showing how uncertainty drives expansion: e.g., “At first step, entropy=2.1 (too high) -> insert 2 tokens; after second, entropy=0.5 -> stop.”

**Citation:** This implements the “adaptive uncertainty-driven expansion” idea from the new text. While related work has studied learned insertion orders, using online uncertainty as a heuristic is a novel twist. We cite [35] to justify entropy as a metric (since ELF used unigram entropy for diversity) but this specific scheme is original to our proposal.

---

## 8. Experiment 5 – Conditional Noise Learning (Ablation)

**Goal:** Test whether learning the expand operator’s injected noise distribution improves performance. In EFM, new coordinates are typically filled with i.i.d. Gaussian noise. We try making their distribution conditional on context.

1. **Modeling:** Introduce a small neural network $\mu_\theta(x_s)$ to predict a mean vector for the inserted noise (with isotropic variance). At an expansion event, instead of sampling $\epsilon \sim \mathcal{N}(0, I)$, we sample $\epsilon \sim \mathcal{N}(\mu_\theta(x_s), I)$. Train $\mu_\theta$ via an $L_2$ loss aligning it to the posterior shift (EFM already outlines a loss $L_{\text{insert}}$ for insertion counts, but not directly for noise means).

2. **Implementation:** Take the EFM insertion head code, add a small MLP ($\approx$1k weights) that takes the pre-expansion state and outputs a vector in the noise space. Freeze the rest, train only this MLP on the same insertion/denoising losses. Alternatively, for simplicity, treat this as another ablation: compare fixed Gaussian vs. learned mean (with the mean network on/off).

3. **Training:** A few epochs of additional training on $\mu_\theta$ (together with expand-transport consistency) on the same data. Use a low LR (1e-5) and small batch.

4. **Evaluation:** Compare Gen-PPL of models with fixed vs. learned noise. If performance improves slightly, then this is useful; if negligible, it suggests simple Gaussian is fine.

> **Output:** A short table: fixed-noise vs. learned-noise Gen-PPL and entropy.

**Citation:** The EFM paper mentions (for continuous EFlows) the expand operator uses conditional noise $p_{\epsilon|s,t}$. The proposed ablation is to learn this distribution, as hinted in the new plan. Since it’s an acknowledged part of the framework, we cite [33†L113-L121] for context of the expand operator.

---

## 9. Experiment 6 – Append-Only vs. Insert Expansion (ELF Control)

**Goal:** Implement simplified expansion baselines using **continuous embeddings** (not discrete tokens) to mimic ELF conditions. This checks how much learned insertion matters.

- **Append-only (Expanding-Fixed):** As in Project 2 but with T5 embeddings: start with a short prefix, append Gaussian noise embeddings (of a fixed total final length), then denoise. Do *not* learn where to append; just append at end.

- **Fixed-insert (no expansion):** Keep sequence length fixed but use an insertion-style evaluation: For example, treat “insertion head” as always outputting zero new tokens, so model is fixed-length.

These will use a small flow model that takes T5 embeddings (e.g., T5-Small 6-layer encoder, 512 hidden) as input. No training – just pre-set operations – to see baseline curves for continuous expansions.

**Implementation:** Build a PyTorch script that:
  - Loads T5-Small encoder (frozen) to produce embeddings for input text.
  - For append-only: given a prefix (from data), append N Gaussian vectors (same dim as T5). Concatenate to form full embedding tensor. Run a small continuous flow model (which we’ll also train) to denoise from noise to data.
  - For fixed-length: simply run flow denoising without expansion (like a standard flow).

**Notes:** These serve as controls. They won’t beat learned insertion but give a floor. We can optionally skip actual training on them (they can be treated as stage results from Exp.5).

> **Output:** Metrics for these baselines, to compare against later learned models.  

**Citation:** EFM’s discrete append-only case is described as one expand operator. The continuous analogy here has no specific citation, it’s just a sanity check.

---

## 10. Model Architecture (Continuous Flow + Insertion Head)

For Stages 6+, we define the continuous EFM-style model that will be trained:

- **Input:** Sequence of context embeddings $x_0 \in \mathbb{R}^{d(0) \times D}$, where $d(0)$ is initial sequence length (e.g. 16), $D$ is embedding dim (we set $D=512$ or 768 to match T5-Small/Medium).
- **Flow Network:** A transformer-like architecture that takes the current embedding sequence and local-time (or its compressed representation) as input, and outputs denoised embeddings. We follow Flow Matching or Score Matching style training (like ELF).
- **Insertion Head:** A separate network (e.g. a small feedforward over each gap) that predicts how many new tokens to insert in each gap. This is a Poisson predictor as in EFM. We implement it as an MLP with softplus to output counts.
- **Combined Step:** An EFM sampling step consists of: (a) apply the expand operator to add noise (learned or fixed) into new positions as per insertion head’s suggestion, (b) apply the flow network to denoise the expanded state to the next time. We then loop as needed until full length.

Because full EFM training (like their joint consistency + insertion loss) is complex, we simplify: we train the flow network and insertion head jointly with a combined objective (cross-entropy on final tokens, possibly plus consistency). This follows techniques from flow-map language models (ELF, Flow map discrete).

**Model Sizes:** For A4000 feasibility, we choose:
- Transformer hidden size $H=512$, layers $\sim6$, heads $8$. This yields roughly $20$–$50$M parameters. E.g., a 6×512×8 model is ~20M.
- T5 encoder: use T5-Small (6×512). Freeze it. That’s ~60M, but frozen (embedding part only, store intermediate embeddings).
- Insertion head: ~100k params (tiny).
- Local-time embedding: K bins or basis of size $K \le 8$.

We enable **mixed precision (FP16)** and **gradient checkpointing** for the transformer to handle B=4, seq_len up to 128. We expect each forward+backward on 64 tokens ~6–8 GB. Adjust batch to 2 or use micro-batches.

No direct citations needed here; this is our design. We align with ELF’s size (105M gave 24 PPL), so a 30M model on shorter sequences should be sufficient for a proof-of-concept.

---

## 11. Stage 6 – Training Continuous EFM with Compression

Having designed the model and obtained insights on local-time from previous stages, we now train the continuous-flow variable-length model. We focus on the most promising settings:

- **Baseline Flow:** Train the model on fixed length (no expansion) as a comparison. (Essentially Project 1 baseline but with T5 embeddings and final token loss.) Measure its PPL.
- **Expanding Flow (quant=∞):** Train with learned insertion and original continuous time signals.
- **Expanding Flow (quantized):** Train with $K=4$ bins (or best from Stage 2) for local time. Only the time embedding is changed; everything else same.
- **Expanding Flow (low-rank):** Train with low-rank time of dimension $K=4$–$8$ (based on Stage 3 results).
- **Adaptive vs. fixed schedule:** If one schedule from Stage 1 was best, train two versions: one where insertion head follows that schedule, another where we allow the model to decide (as per uncertainty, but here we can simulate by forcing insertion at dynamic times during training, which is complicated). This is optional.

**Training Commands:** Using `train_flow.py` (our script):

```bash
# Example: train expanding flow with T5 embeddings
python train_flow.py \
  --config configs/train_flow.yaml \
  --model.hidden_size 512 \
  --model.layers 6 \
  --dataset openwebtext \
  --schedule alpha=1.0 \
  --time_mode continuous  \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --fp16 \
  --grad_checkpoint
```

We will have separate configs (YAML or CLI flags) for each variant. Key hyperparameters (guide values):

- **Batch size:** 4 (64 tokens each, adjust if VRAM is tight).
- **Learning rate:** 1e-4 (AdamW with weight decay 0.01). Possibly start lower (5e-5) if instability.
- **Optimizer:** AdamW.
- **Steps:** ~50k steps (like ~20M tokens) for initial runs; can extend if needed.
- **LR scheduler:** Cosine decay or linear with warmup (e.g. 10% warmup, 50k steps).
- **Precision:** AMP (FP16) with gradient scaling.
- **Checkpointing:** Save model every 5k steps, keep last 3. 

We use the `accelerate` library or manual `amp` context. Example (pseudocode):
```python
from accelerate import Accelerator
accelerator = Accelerator(fp16=True)
model, optimizer = accelerator.prepare(model, optimizer)
for step, batch in enumerate(data_loader):
    optimizer.zero_grad()
    loss = model.compute_loss(batch)  # includes token and insertion loss
    accelerator.backward(loss)
    optimizer.step()
    if step % 1000 == 0: save_checkpoint()
```

**Notes:** For quantized or low-rank variants, adjust the model definition accordingly (e.g. have only $K$ time bins to learn). Training must include flow-map and insertion losses. If unstable, try freezing the insertion head for initial warmup.

**Evaluation:** After training each model, evaluate on held-out data by generating 1000 samples and computing Gen-PPL (via a GPT-2 Large or the T5 decoder). Also compute test perplexity (ELF’s “model perplexity” or simply cross-entropy on next-token). Compare:
- Fixed vs. expanding flows.
- Continuous vs. quantized vs. low-rank.

We expect: expanding with full time < fixed; quantized close; low-rank maybe slight gap.

> **Output:** A table of Gen-PPL / entropy for all trained models, plus an ablation figure showing relative drop-offs.

**Citation:** These steps follow the methodology of ELF (flow model with T5 embeddings) and EFM’s training of insertion head. We implement the compression ideas (quantized/low-rank) motivated by the combined analysis above.

---

## 12. Evaluation Metrics and Code

We will consistently use the following metrics:

- **Generative Perplexity (Gen-PPL):** The perplexity of *generated* text under a pre-trained GPT-2 Large model. Lower is better. We implement:

   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer
   import torch
   # Load once
   gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-large").eval().to(device)
   tok2 = GPT2Tokenizer.from_pretrained("gpt2-large")
   def gen_perplexity(texts):
       inputs = tok2(texts, return_tensors="pt", padding=True, truncation=True).to(device)
       with torch.no_grad():
           loss = gpt2(**inputs, labels=inputs["input_ids"]).loss
       ppl = torch.exp(loss).item()
       return ppl
   ```
  We will generate samples (with fixed random seeds) and compute Gen-PPL for each model. (ELF and EFM papers report Gen-PPL exactly this way.)

- **Unigram Entropy:** The average word-level entropy of generated samples. For an evaluation set of sentences:
  
   ```python
   from collections import Counter
   import math
   words = " ".join(samples).split()
   counts = Counter(words)
   probs = torch.tensor([c/len(words) for c in counts.values()])
   entropy = -(probs * torch.log(probs)).sum().item() / math.log(2)
   ```
  (ELF reports entropy in bits; ensure consistent base if replicating. In [32], entropy ~5.2 “nats” maybe, but we can report bits.)

- **Decoder Confidence:** Since we’re in continuous space until final step, define confidence as: after generating continuous embeddings, we feed them through the “discretization head” (usually a linear projection + softmax over vocabulary, same as T5’s output layer). For each position, take `max_prob = max(softmax(logits))`; then average those across tokens. Higher means model is more sure. Code snippet:
  
   ```python
   # Suppose 'final_embeddings' is tensor [batch, T, D]
   logits = decoder_head(final_embeddings)  # shape [batch, T, vocab_size]
   probs = torch.softmax(logits, dim=-1)
   confidence = probs.max(dim=-1).values.mean().item()
   ```

- **Auto-Perplexity (in-model):** Optionally, the held-out log-likelihood of real text under the flow model (converted to “pseudo-perplexity”). Not always defined for flows, so skip unless needed.

- **BLEU/ROUGE (if we do conditional tasks):** Not needed here, focus on unconditional generation.

All metrics should be logged in training and evaluation scripts. We’ll store results to WandB and also in CSV logs.

**Citation:** The use of GPT-2 for Gen-PPL is directly from ELF and EFM evaluations. We cite these to justify our metrics.

---

## 13. Experiment Matrix and Run Table

We summarize planned experiments in the table below. Each row is a model run with specific settings. We will note expected VRAM and time:

| Representation       | Expansion   | Time Cond.        | Layers×Hidden | Params (M) | Batch | SeqLen | Notes                             | VRAM (est)   |
|----------------------|-------------|-------------------|--------------:|-----------:|------:|-------:|-----------------------------------|-------------|
| **Fixed baseline**   | none        | N/A               | 6×512        | ~20        | 4     | 64     | fixed-length flow (no expand)     | ~8–10 GB    |
| **Append-only**      | append      | original (contin.)| 6×512        | ~20        | 4     | 64→128 | expand by appending Gaussian      | ~10–12 GB   |
| **Quantized $K=4$**  | learned     | quantized (K=4)   | 6×512        | ~20        | 4     | 64→128 | insertion head + 4-bin time       | ~10–12 GB   |
| **Low-rank $K=8$**   | learned     | low-rank (K=8)    | 6×512        | ~20        | 4     | 64→128 | insertion head + 8-dim time       | ~10–12 GB   |
| **T5 embeddings**    | fixed-insert| original          | 6×512        | ~20        | 4     | 64     | control: fixed-length (ELF style) | ~8–10 GB    |
| **Full ELF (B)**     | learned     | original          | 6×512        | ~20        | 4     | 64→128 | continuous EFM on embeddings      | ~12–14 GB   |
| **+Adaptive Sch.**   | adaptive    | original          | 6×512        | ~20        | 4     | dynamic| same as above, uncertain schedule  | ~12–14 GB   |

*Each experiment involves training or evaluation as per plan.* Actual VRAM can be reduced with FP16 and checkpointing; these estimates assume FP16 and moderate pipeline usage. Runtime per 50k steps ~1–2 days on A4000 (6–8 hours per 10k, approximated). The *content above guides exact Python commands and configs.*

---

## 14. Reproducibility and Logging

To ensure reproducibility, we fix all random seeds at script start:

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

We log all runs using **Weights & Biases**. For each run:

```python
import wandb
wandb.init(project="ExpandingFlows", config={
    "model": {...},
    "dataset": "OpenWebText",
    "batch_size": 4,
    "seq_len": 64,
    "LR": 1e-4,
    "steps": 50000
})
wandb.watch(model)
```

We save metrics (`loss`, `Gen-PPL`, `entropy`, etc.) every 500 steps. We also upload model checkpoints to W&B or our own storage.

All code and data should be version-controlled. We include a `requirements.lock` file (from `pip freeze`). We also set `accelerate config` to fix device placement.

---

## 15. Unit Tests and Smoke Tests

Before heavy runs, implement quick tests:

- **Data Loader Test:** Load a batch and check shapes.
   ```python
   for batch in train_loader:
       assert batch["input_ids"].shape[1] <= max_seq_len
       break
   ```
- **Model Forward Test:** Feed random data:
   ```python
   dummy = torch.randn(1, 32, 512).to(device)  # 32 tokens, 512-dim
   out = model(dummy, times=torch.rand(1,32,1))
   assert not torch.isnan(out).any()
   ```
- **Expansion Test:** Simulate one expand step: ensure insertion head outputs plausible counts and flow network handles added noise.

- **Sampling Script Test:** Run the generation code with a tiny toy model to ensure syntax and logic.

If errors arise (e.g., out-of-memory, NaNs, tiny BLEU), check:
- Learning rate too high: reduce to 5e-5.
- Batch too large: lower it.
- FP16 overflow: use gradient scaling or switch to FP32.
- Instability: reduce gradient clipping, use `accumulate_grad_batches`.
- Code bugs: print shapes diligently.

In case of convergence issues, try turning off some losses (only train flow vs. insertion head separately first).

---

## 16. Figures and Diagrams

### Pipeline Flowchart

```mermaid
flowchart TB
    A[Data (tokenized text)] -->|T5 encoder| B[T5 Embeddings (frozen)]
    B --> C[Flow Transformer (denoiser)]
    C --> D[Expanded Embedding Sequence]
    D --> E[Loop: Insert + Denoise]
    E --> F[Final Embedding Sequence]
    F -->|Decoder| G[Generated Text]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ff9,stroke:#333,stroke-width:2px
```
*Figure: Simplified pipeline: text tokens → T5 embeddings → flow/expansion steps → final embeddings → text output (via decoder).*

### Timeline Diagram

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    title Experiment Timeline (Weeks)
    section Screening
    Schedule Tests         :done, des1, 2026-09-18, 3d
    Quantization Tests     :done, des2, 2026-09-21, 3d
    Low-Rank Tests         :active, des3, 2026-09-24, 4d
    Adaptive/Noise Tests   :, des4, 2026-09-28, 4d
    section Training
    Baseline Training      :crit, training1, after des4, 1w
    Compressed Training    :, training2, after training1, 1w
    Continuous EFM Training:crit, training3, after training2, 2w
    section Analysis
    Evaluation Metrics     :milestone, 2026-10-22, 1d
    Figure Generation      :milestone, 2026-10-24, 1d
```
*Figure: Proposed schedule (dates are illustrative). Screening experiments are quick; main training spans weeks.* 

---

## 17. References

- **Expanding Flow Maps (Tang & Chatterjee 2026)**: Defines the EFM framework with expand and transport maps. Key concepts: per-token local clocks, insertion head.  
- **ELF: Embedded Language Flows (Hu et al. 2026)**: Introduces continuous diffusion in embedding space for text. Provides code and metrics (Gen-PPL, entropy).  
- **Flow Map Language Models (Lee et al. 2026)**: Predecessor to EFM’s language work (variable-length flow maps).  
- **How to build a consistency model (Boffi et al. 2025)**: Basis for flow map distillation (flow maps concept).  
- **Software repos:** ELF code repo (branch `pytorch_elf`). EFM repo (no code yet). HuggingFace datasets and models (T5, GPT-2).  
- **Metrics usage:** GPT-2 for Gen-PPL; entropy as diversity measure.

