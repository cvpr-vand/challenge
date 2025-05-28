#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# test single seed and k_shot
uv run eval --k_shots 2 4  --seeds 42  --dataset_path /data/public/dataset/mvtec_loco/

# default --k_shots 1 2 4 8 --seeds 42 0 1234
# uv run eval --dataset_path /data/public/dataset/mvtec_loco/