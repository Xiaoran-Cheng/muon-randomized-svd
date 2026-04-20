#!/usr/bin/env bash
# Muon (Polyak momentum, full Newton-Schulz). Update TUNED values after the sweep finishes.
set -e

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 3 \
  --log-every 50 \
  --wandb True \
  --wandb-project muon-polyak-full-nanogpt \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov False \
  --inexact-solver polar_express \
  --orth-steps 5 \
  --randomized False \
  --rank 100 \
  --oversampling 10 \
  --power-iters 0 \
  --sgd-lr 0.003 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 2048 \
  "$@"
