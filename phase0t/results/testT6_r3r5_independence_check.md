# Part T6: R3/R5 Independence Check and R1 Audit

## 1. Checksum Verification
We computed MD5 hashes of all model parameters to verify the checkpoints are distinct:
- **Control:** `f152bc04d0c5460e6f099532b5ee0568`
- **R1 (Density):** `6dfb83f7c3100e6b201f11aade30fb3c`
- **R3 (Label Smooth):** `ce022e5eaef6735b081c5e17fa321c04`
- **R5 (Decoupled):** `1cd0c8738f78500c3a9f132ef9ae930c`

## 2. Degenerate Sample Overlap (Seed Vulnerability)
- **R3 Degenerate Indices (Count = 25):** [np.int64(2), np.int64(4), np.int64(5), np.int64(7), np.int64(11), np.int64(14), np.int64(16), np.int64(33), np.int64(36), np.int64(42), np.int64(49), np.int64(50), np.int64(51), np.int64(52), np.int64(53), np.int64(54), np.int64(55), np.int64(62), np.int64(63), np.int64(136), np.int64(150), np.int64(174), np.int64(187), np.int64(189), np.int64(231)]
- **R5 Degenerate Indices (Count = 25):** [np.int64(4), np.int64(5), np.int64(6), np.int64(11), np.int64(14), np.int64(24), np.int64(27), np.int64(29), np.int64(30), np.int64(33), np.int64(35), np.int64(42), np.int64(46), np.int64(50), np.int64(52), np.int64(53), np.int64(54), np.int64(55), np.int64(58), np.int64(60), np.int64(62), np.int64(63), np.int64(136), np.int64(174), np.int64(182)]

**Finding:** The degenerate samples in R3 and R5 are NOT identical (Overlap: 15 samples). The identical sum of 25 is a coincidence.

## 3. R1 (Density Reweighting) Audit
- **Control Degenerate (Count = 52)**
- **R1 Degenerate (Count = 90)**
- **New failures introduced by R1:** 52 samples
- **Control failures fixed by R1:** 14 samples

**Explanation:**
Density reweighting (oversampling early timesteps) worsened degeneracy primarily by failing on seeds that the Control model survived. Because continuous-time flow-matching requires extreme precision at the boundary (t→1) to form coherent discrete tokens, neglecting the t→1 training region in favor of the early region (t→0) leaves the model highly vulnerable to truncation and discretization errors at the boundary. The model learns poor vector fields precisely where ODE instability is highest, leading to widespread catastrophic collapse on noise vectors that a balanced model can handle.