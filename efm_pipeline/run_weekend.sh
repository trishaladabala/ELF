#!/bin/bash
# We do NOT use 'set -e' here. 
# If one condition crashes (e.g., Out Of Memory), the script will ignore the error and keep going!

echo "=================================================="
echo "Starting Weekend Run: $(date)"
echo "=================================================="

echo ""
echo "--- TASK 1: Finishing W=8 Sweep ---"
# Running each condition independently. The --skip-completed flag ensures anything already done is skipped!
for cond in global_time_only continuous lowrank_K1 lowrank_K2 lowrank_K4 lowrank_K8 lowrank_K16 quantized_K1 quantized_K2 quantized_K4 quantized_K8 quantized_K16; do
    echo "Running W=8 Sweep: $cond"
    python -m experiments.run_wk_sweep --W 8 --conditions $cond --skip-completed || echo "WARNING: $cond failed!"
done

echo ""
echo "--- TASK 2: Multi-Seed Validation (W=8) ---"
# Again, isolating each condition
for cond in global_time_only continuous lowrank_K2 lowrank_K4 lowrank_K8; do
    echo "Running Multi-Seed W=8: $cond"
    python -m experiments.run_multiseed --W 8 --conditions $cond --skip-completed || echo "WARNING: Multi-seed $cond failed!"
done

echo ""
echo "--- TASK 3: 105M Scale-Up Gen-PPL Evaluation ---"
# The scale-up script evaluates all checkpoints.
# We will just let it run. If a specific checkpoint evaluation fails, it might stop the scaleup script,
# but it's the last step anyway.
python -m experiments.evaluate_scaleup --skip-completed || echo "WARNING: Scale-up evaluation failed!"

echo ""
echo "=================================================="
echo "Weekend Run Completed: $(date)"
echo "=================================================="

