#!/usr/bin/env bash
# Ablation (rank=100), 1x A100 80GB, xformers backend.
set -e
export DISABLE_FP8=1
export DISABLE_COMPILE=1
export VAL_BATCH_SIZE=262144
export GRAD_ACCUM_STEPS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/xcheng328/nanogpt                               # TODO: adjust to your path
source /home/xcheng328/miniconda3/bin/activate muon      # TODO: adjust to your env

torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 5 \
  --log-every 50 \
  --wandb True \
  --wandb-project new_muon-ablation-rank-nanogpt \
  --wandb-group ablation_rank_100_GT \
  --muon-lr 0.032463575100400426 \
  --muon-momentum 0.9664542176931918 \
  --muon-nesterov True \
  --inexact-solver quintic_theoretical \
  --orth-steps 7 \
  --randomized True \
  --rank 100 \
  --oversampling 10 \
  --power-iters 1 \
  --sgd-lr 0.001 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 2048 \
  --log-root ablation_Logs \
  "$@"
