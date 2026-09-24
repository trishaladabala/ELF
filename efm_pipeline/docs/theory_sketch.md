# Theory Sketch: Compression as Regularization for Temporal Conditioning

## 1. Setup
- Conditioning signal tau in [0,1] (continuous) vs. K-valued 
  (quantized) vs. d-dimensional (low-rank projection).
- Training budget: N samples, T steps.
- Goal: bound generalization gap epsilon(C, N) as a function of 
  conditioning complexity C.

## 2. Complexity measure
- Bits to specify continuous tau at precision delta: log(1/delta)
- Bits to specify K-bin quantization: log K
- Bits to specify d-dim low-rank projection at precision delta: 
  d log(1/delta)
- Define C = bits required.

## 3. Rate-distortion argument (TO BE FILLED)
- Hypothesis class size as a function of C.
- Generalization bound scaling with C and N.
- Optimal C* that minimizes epsilon(C, N).

## 4. Prediction
- C* depends on task complexity W (e.g., number of distinct 
  insertion regimes).
- Prediction: C* ~ W.
- Empirical test: does the U-shaped curve bottom out near K ~ W 
  across tasks?

## 5. Open questions


