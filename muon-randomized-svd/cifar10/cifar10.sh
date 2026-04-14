#!/usr/bin/env bash
cd /home/xcheng328/cifar10
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon


python airbench94_muon.py \
  --optimizer-mode muon \
  --epochs 1 \
  --num-trials 1 \
  --val-every-steps 100 \
  --wandb False \
  --batch-size 2000 \
  --sgd-momentum 0.85 \
  --sgd-nesterov True \
  --muon-lr 0.24 \
  --muon-momentum 0.6 \
  --muon-nesterov True \
  --filter-sgd-lr 0.24 \
  --filter-sgd-weight-decay 0.0 \
  --adamw-beta1 0.9 \
  --adamw-beta2 0.999 \
  --adamw-eps 1e-8 \
  --filter-adamw-lr 0.24 \
  --filter-adamw-weight-decay 0.0 \
  --inexact_solver quintic_ns_empirical \
  --orth-steps 3 \
  --randomized False \
  --rank 32 \
  --oversampling 2 \
  --power-iters 0 \
  "$@"
