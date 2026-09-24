#!/bin/bash
# Fine-tune ELF with BW-ELF (Basin-Widened ELF) objectives

cd "$(dirname "$0")/.." || exit

# Create the output directory
mkdir -p phase3_basin/results/finetune

echo "============================================================"
echo "Starting BW-ELF Fine-tuning on OpenWebText"
echo "============================================================"

# Ensure dataset path is set to the correct location for OpenWebText
# If the dataset is cached locally, update the config file appropriately.
# Here we just run the training script with our custom config.

export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    --config src/configs/bw_elf_finetune.yaml \
    --config_override "data_path=openwebtext" \
    --config_override "lambda_robust=0.03" \
    --config_override "lambda_intermediate_ce=0.0" \
    --config_override "lambda_margin=0.0"

echo "Fine-tuning run complete!"
