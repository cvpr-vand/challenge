#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT=https://hf-mirror.com

# test single seed and k_shot
uv run eval --k_shots 1 --seeds 42  --dataset_path /data/public/dataset/mvtec_loco

# default --k_shots 1 2 4 8 --seeds 42 0 1234
# nohup uv run eval --dataset_path /data/public/dataset/mvtec_loco >eval.log 2>&1 &