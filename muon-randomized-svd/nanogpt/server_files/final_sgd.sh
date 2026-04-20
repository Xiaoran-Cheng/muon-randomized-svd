#!/usr/bin/env bash
# SGD + Nesterov baseline (E5 row 2). Update TUNED values after the sweep finishes.
set -e

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --optimizer-mode sgd_nesterov \
  --num-trials 3 \
  --log-every 50 \
  --wandb True \
  --wandb-project sgd-nesterov-nanogpt \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --sgd-lr 0.003 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov True \
  --adamw-lr 0.001 \
  --inexact-solver polar_express \
  --orth-steps 5 \
  --randomized False \
  --rank 100 \
  --oversampling 10 \
  --power-iters 0 \
  --seq-len 2048 \
  "$@"
