#!/bin/bash
# Consistency Flow Distillation — Full Pipeline
# Run from: /home/snu/trishal_sanjay_proj/ELF/elf-basin
set -e

PYTHON="/home/snu/miniconda3/envs/mdlm/bin/python"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo " Step 0: Verify teacher reproduction"
echo "=========================================="
# Skip if already verified (reproduce_deen.py gives 26.51 BLEU)

echo ""
echo "=========================================="
echo " Step 1: Prepare distillation data"
echo "=========================================="
# $PYTHON scripts/prepare_distill_data.py

echo ""
echo "=========================================="
echo " Step 2: Train consistency model (our method)"
echo "=========================================="
$PYTHON scripts/train_consistency.py

echo ""
echo "=========================================="
echo " Step 3: Train progressive distillation (baseline)"
echo "=========================================="
$PYTHON scripts/train_progressive.py

echo ""
echo "=========================================="
echo " Step 4: Evaluate all models"
echo "=========================================="
$PYTHON scripts/eval_distilled.py

echo ""
echo "=========================================="
echo " Step 5: Generate analysis & paper assets"
echo "=========================================="
$PYTHON scripts/analysis_plots.py

echo ""
echo "=========================================="
echo " Done! Results in runs/eval_results/"
echo "=========================================="
