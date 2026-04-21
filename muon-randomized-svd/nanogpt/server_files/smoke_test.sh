#!/usr/bin/env bash
# Smoke test: verify the code runs on a single A100 without crashing.
# Shortens training to ~50 steps via NUM_ITERATIONS / NUM_EXT_ITERATIONS env vars.
# Not a performance/accuracy run — world_size=1 breaks the batch-size formula.
set -e

# --- Hardware fallbacks (A100 doesn't support H100-only kernels) ---
export DISABLE_FP8=1
# xformers varlen path uses .tolist() which breaks torch.compile fullgraph mode.
export DISABLE_COMPILE=1
# Shrink val_batch_size (default 2,097,152 tokens) so the warmup forward fits on one A100.
# Must be a multiple of world_size*grad_accum_steps (=8 at nproc=1) AND divide val_tokens below.
export SMOKE_VAL_TOKENS=128
# Also cap total val_tokens so val_steps = grad_accum * val_tokens / val_batch_size stays small.
# With gas=8, val_batch=128: 8*1024/128 = 64 val iterations per pass (~seconds instead of hours).
export SMOKE_VAL_TOKENS_TOTAL=1024

# --- Smoke-test scale: tiny training loop ---
export NUM_ITERATIONS=50
export NUM_EXT_ITERATIONS=5

# TODO: adjust cd path + conda env name to match your HPC setup.
cd /home/xcheng328/nanogpt
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon

torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 1 \
  --log-every 25 \
  --wandb False \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov True \
  --inexact-solver polar_express \
  --orth-steps 3 \
  --randomized False \
  --rank 100 \
  --oversampling 10 \
  --power-iters 0 \
  --sgd-lr 0.003 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 128 \
  "$@"
