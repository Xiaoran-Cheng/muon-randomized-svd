#!/usr/bin/env bash
cd /home/xcheng328/cifar10
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon


python airbench94_muon_a6000.py \
  --optimizer-mode sgd \
  --epochs 8 \
  --num-trials 50 \
  --val-every-steps 10 \
  --wandb True \
  --wandb-project sgd-nesterov \
  --wandb-group final_$(date +%Y%m%d_%H%M%S) \
  --batch-size 500 \
  --sgd-momentum 0.7061487457292506 \
  --sgd-nesterov True \
  --filter-sgd-lr 0.017051314712312846 \
  --filter-sgd-weight-decay 0.0 \
  --muon-lr 0.24 \
  --muon-momentum 0.6 \
  --muon-nesterov True \
  --adamw-beta1 0.9 \
  --adamw-beta2 0.999 \
  --adamw-eps 1e-8 \
  --filter-adamw-lr 0.24 \
  --filter-adamw-weight-decay 0.0 \
  --inexact_solver cubic_ns_theoretical \
  --orth-steps 5 \
  --randomized False \
  --rank 32 \
  --oversampling 10 \
  --power-iters 0 \
  "$@"
