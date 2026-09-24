#!/bin/bash
set -e
python -m efm_pipeline.train_efm --seed 42 --steps 50000 --local-time none --exp-name global_time_only_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/global_time_only_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/global_time_only_seed42_result.json
python -m efm_pipeline.train_efm --seed 42 --steps 50000 --local-time expansion --exp-name continuous_local_time_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/continuous_local_time_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/continuous_local_time_seed42_result.json
python -m efm_pipeline.train_efm --seed 42 --steps 50000 --local-time none --exp-name trained_without_local_time_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/trained_without_local_time_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/trained_without_local_time_seed42_result.json
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/continuous_local_time_s42/final.pt --quantize-k 1 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K1_seed42_result.json
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/continuous_local_time_s42/final.pt --quantize-k 2 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K2_seed42_result.json
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/continuous_local_time_s42/final.pt --quantize-k 4 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K4_seed42_result.json
python -m efm_pipeline.train_efm --seed 42 --steps 5000 --local-time quantized --quantize-k 2 --resume checkpoints/continuous_local_time_s42/final.pt --exp-name learned_quantized_K2_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/learned_quantized_K2_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_quantized_K2_seed42_result.json
python -m efm_pipeline.train_efm --seed 42 --steps 5000 --local-time quantized --quantize-k 4 --resume checkpoints/continuous_local_time_s42/final.pt --exp-name learned_quantized_K4_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/learned_quantized_K4_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_quantized_K4_seed42_result.json
python -m efm_pipeline.train_efm --seed 42 --steps 5000 --local-time lowrank --rank 2 --resume checkpoints/continuous_local_time_s42/final.pt --exp-name learned_lowrank_K2_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/learned_lowrank_K2_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_lowrank_K2_seed42_result.json
python -m efm_pipeline.train_efm --seed 42 --steps 5000 --local-time lowrank --rank 4 --resume checkpoints/continuous_local_time_s42/final.pt --exp-name learned_lowrank_K4_s42
python -m efm_pipeline.eval_runner --seed 42 --checkpoint checkpoints/learned_lowrank_K4_s42/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_lowrank_K4_seed42_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 50000 --local-time none --exp-name global_time_only_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/global_time_only_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/global_time_only_seed137_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 50000 --local-time expansion --exp-name continuous_local_time_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/continuous_local_time_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/continuous_local_time_seed137_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 50000 --local-time none --exp-name trained_without_local_time_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/trained_without_local_time_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/trained_without_local_time_seed137_result.json
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/continuous_local_time_s137/final.pt --quantize-k 1 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K1_seed137_result.json
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/continuous_local_time_s137/final.pt --quantize-k 2 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K2_seed137_result.json
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/continuous_local_time_s137/final.pt --quantize-k 4 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K4_seed137_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 5000 --local-time quantized --quantize-k 2 --resume checkpoints/continuous_local_time_s137/final.pt --exp-name learned_quantized_K2_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/learned_quantized_K2_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_quantized_K2_seed137_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 5000 --local-time quantized --quantize-k 4 --resume checkpoints/continuous_local_time_s137/final.pt --exp-name learned_quantized_K4_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/learned_quantized_K4_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_quantized_K4_seed137_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 5000 --local-time lowrank --rank 2 --resume checkpoints/continuous_local_time_s137/final.pt --exp-name learned_lowrank_K2_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/learned_lowrank_K2_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_lowrank_K2_seed137_result.json
python -m efm_pipeline.train_efm --seed 137 --steps 5000 --local-time lowrank --rank 4 --resume checkpoints/continuous_local_time_s137/final.pt --exp-name learned_lowrank_K4_s137
python -m efm_pipeline.eval_runner --seed 137 --checkpoint checkpoints/learned_lowrank_K4_s137/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_lowrank_K4_seed137_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 50000 --local-time none --exp-name global_time_only_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/global_time_only_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/global_time_only_seed2024_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 50000 --local-time expansion --exp-name continuous_local_time_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/continuous_local_time_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/continuous_local_time_seed2024_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 50000 --local-time none --exp-name trained_without_local_time_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/trained_without_local_time_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/trained_without_local_time_seed2024_result.json
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/continuous_local_time_s2024/final.pt --quantize-k 1 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K1_seed2024_result.json
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/continuous_local_time_s2024/final.pt --quantize-k 2 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K2_seed2024_result.json
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/continuous_local_time_s2024/final.pt --quantize-k 4 --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/frozen_K4_seed2024_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 5000 --local-time quantized --quantize-k 2 --resume checkpoints/continuous_local_time_s2024/final.pt --exp-name learned_quantized_K2_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/learned_quantized_K2_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_quantized_K2_seed2024_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 5000 --local-time quantized --quantize-k 4 --resume checkpoints/continuous_local_time_s2024/final.pt --exp-name learned_quantized_K4_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/learned_quantized_K4_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_quantized_K4_seed2024_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 5000 --local-time lowrank --rank 2 --resume checkpoints/continuous_local_time_s2024/final.pt --exp-name learned_lowrank_K2_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/learned_lowrank_K2_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_lowrank_K2_seed2024_result.json
python -m efm_pipeline.train_efm --seed 2024 --steps 5000 --local-time lowrank --rank 4 --resume checkpoints/continuous_local_time_s2024/final.pt --exp-name learned_lowrank_K4_s2024
python -m efm_pipeline.eval_runner --seed 2024 --checkpoint checkpoints/learned_lowrank_K4_s2024/final.pt --out /home/snu/trishal_sanjay_proj/ELF/efm_pipeline/validation_phase/learned_lowrank_K4_seed2024_result.json
