#!/usr/bin/env bash
cd /home/xcheng328/cifar10
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon


python airbench94_muon.py \
  --optimizer-mode adamw \
  --epochs 8 \
  --num-trials 50 \
  --val-every-steps 10 \
  --wandb True \
  --wandb-project adamw \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --batch-size 500 \
  --adamw-beta1 0.8755226098803631 \
  --adamw-beta2 0.9903971024615889 \
  --adamw-eps 1e-8 \
  --filter-adamw-lr 0.004368346963334174 \
  --filter-adamw-weight-decay 0.0 \
  --sgd-momentum 0.85 \
  --sgd-nesterov True \
  --filter-sgd-lr 0.24 \
  --filter-sgd-weight-decay 0.0 \
  --muon-lr 0.24 \
  --muon-momentum 0.6 \
  --muon-nesterov True \
  --inexact_solver cubic_ns_theoretical \
  --orth-steps 5 \
  --randomized False \
  --rank 32 \
  --oversampling 10 \
  --power-iters 0 \
  "$@"
