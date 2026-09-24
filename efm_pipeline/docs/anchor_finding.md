# Anchor Finding
We observe a U-shaped relationship between the complexity of the 
local-time conditioning signal and generation quality in Expanding 
Flow Maps. At 36M parameters and 50k training steps on WikiText-2, 
continuous local time (Gen-PPL 1398) performs worse than no local 
time (1102), while learned compressed representations—specifically 
Low-Rank K=4 (984) and Quantized K=2 (1040)—outperform both. This 
suggests that compression acts as a regularizer that preserves 
useful timing information while reducing the learning complexity 
for the model.

