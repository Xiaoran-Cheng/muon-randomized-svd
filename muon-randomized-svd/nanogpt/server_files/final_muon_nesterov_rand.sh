#!/usr/bin/env bash
# Muon (Nesterov momentum, randomized Newton-Schulz) — main method.
# Update TUNED values after the sweep finishes.
set -e

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 3 \
  --log-every 50 \
  --wandb True \
  --wandb-project muon-nanogpt \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov True \
  --inexact-solver polar_express \
  --orth-steps 5 \
  --randomized True \
  --rank 100 \
  --oversampling 10 \
  --power-iters 1 \
  --sgd-lr 0.003 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 2048 \
  "$@"
