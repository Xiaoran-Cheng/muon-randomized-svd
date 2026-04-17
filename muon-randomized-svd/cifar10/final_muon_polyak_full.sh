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
  --wandb-project muon-polyak-full \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --batch-size 1000 \
  --sgd-momentum 0.85 \
  --sgd-nesterov True \
  --muon-lr 0.14924799276952783 \
  --muon-momentum 0.49283224551213034 \
  --muon-nesterov False \
  --filter-sgd-lr 0.24 \
  --filter-sgd-weight-decay 0.0 \
  --adamw-beta1 0.9 \
  --adamw-beta2 0.999 \
  --adamw-eps 1e-8 \
  --filter-adamw-lr 0.24 \
  --filter-adamw-weight-decay 0.0 \
  --inexact_solver polar_express \
  --orth-steps 7 \
  --randomized False \
  --rank 32 \
  --oversampling 10 \
  --power-iters 0 \
  "$@"
