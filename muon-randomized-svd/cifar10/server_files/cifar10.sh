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
  --wandb-project muon-nesterov-rand \
  --wandb-group best_$(date +%Y%m%d_%H%M%S) \
  --batch-size 1000 \
  --sgd-momentum 0.85 \
  --sgd-nesterov True \
  --muon-lr 0.19572308843587835 \
  --muon-momentum 0.48356183431214944 \
  --muon-nesterov True \
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
  --power-iters 2 \
  "$@"
