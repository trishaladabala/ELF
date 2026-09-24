# Second Synthetic Task: Design Spec

## Goal
Reproduce the U-shaped compression curve on a task with a DIFFERENT 
generative structure than the first synthetic task.

## Candidate designs
- **Option A [PRIMARY]**: Hierarchical key-value copy. First half of sequence 
  contains keys; second half must generate values in a fixed order.
- **Option C [SECOND TASK]**: Sorted sequence generation. Insertion order matters; 
  tokens must be inserted in sorted order.
- ~~Option B: Variable-length generation. Dropped.~~

## Constraints
- Max sequence length: 256 tokens
- Vocab size: <= 1k (synthetic)
- Training budget: 50k steps (matching the first task)
- Model: 36M params (matching the first task)

## Metrics
- Gen-PPL under a small held-out scorer
- Entropy (to detect mode collapse)
- Report the U-shaped curve across: continuous, Quantized K=2/K=4, 
  Low-Rank K=2/K=4

## Success criterion
U-shape reproduces. Bonus: the optimal K shifts with task 
complexity, supporting the theory prediction C* ~ W.

