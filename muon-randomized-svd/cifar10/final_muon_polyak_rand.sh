#!/usr/bin/env bash
cd /home/xcheng328/cifar10
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon


python airbench94_muon.py \
  --optimizer-mode muon \
  --epochs 8 \
  --num-trials 50 \
  --val-every-steps 10 \
  --wandb True \
  --wandb-project muon-polyak-rand \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --batch-size 1000 \
  --sgd-momentum 0.85 \
  --sgd-nesterov True \
  --muon-lr 0.1410344876266027 \
  --muon-momentum 0.4199272644539725 \
  --muon-nesterov False \
  --filter-sgd-lr 0.24 \
  --filter-sgd-weight-decay 0.0 \
  --adamw-beta1 0.9 \
  --adamw-beta2 0.999 \
  --adamw-eps 1e-8 \
  --filter-adamw-lr 0.24 \
  --filter-adamw-weight-decay 0.0 \
  --inexact_solver cubic_ns_theoretical \
  --orth-steps 9 \
  --randomized True \
  --rank 256 \
  --oversampling 10 \
  --power-iters 1 \
  "$@"
