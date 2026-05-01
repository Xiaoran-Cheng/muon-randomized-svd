#!/usr/bin/env bash
# Ablation (seq_len=4096, double batch schedule), 1x A100 80GB, xformers backend.
set -e
export DISABLE_FP8=1
export DISABLE_COMPILE=1
export VAL_BATCH_SIZE=262144
# Larger seq_len doubles per-step batch; scale GRAD_ACCUM_STEPS up to keep
# per-GPU per-micro-batch tokens comparable to the seq_len=2048 baseline.
export GRAD_ACCUM_STEPS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/xcheng328/nanogpt                               # TODO: adjust to your path
source /home/xcheng328/miniconda3/bin/activate muon      # TODO: adjust to your env

torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 5 \
  --log-every 50 \
  --wandb True \
  --wandb-project new_muon-ablation-batch-nanogpt \
  --wandb-group ablation_batch_4096_GT \
  --muon-lr 0.032463575100400426 \
  --muon-momentum 0.9664542176931918 \
  --muon-nesterov True \
  --inexact-solver quintic_theoretical \
  --orth-steps 7 \
  --randomized True \
  --rank 200 \
  --oversampling 10 \
  --power-iters 1 \
  --sgd-lr 0.001 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 4096 \
  --log-root ablation_Logs \
  "$@"
