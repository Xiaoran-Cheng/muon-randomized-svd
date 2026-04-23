# Final run: SGD + Nesterov baseline, 1x A6000 48GB, xformers backend.
set -e
export DISABLE_FP8=1                   # A6000 is sm86, no FP8
export DISABLE_COMPILE=1               # xformers .tolist() conflicts with fullgraph compile
export VAL_BATCH_SIZE=262144           # val_batch_size: per-rank T=32,768 fits 48GB forward
# GRAD_ACCUM_STEPS scaling by GPU count (keep GAS x world_size = 16 so per-GPU per-mb tokens stay constant):
#   1x A6000: GRAD_ACCUM_STEPS=16
#   2x A6000: GRAD_ACCUM_STEPS=8
#   4x A6000: GRAD_ACCUM_STEPS=4
#   8x A6000: GRAD_ACCUM_STEPS=2
export GRAD_ACCUM_STEPS=16

# cd /home/xcheng328/nanogpt                               # TODO: adjust to your path
# source /home/xcheng328/miniconda3/bin/activate muon      # TODO: adjust to your env

NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --optimizer-mode sgd_nesterov \
  --num-trials 5 \
  --log-every 50 \
  --wandb True \
  --wandb-project sgd-nesterov-nanogpt \
  --wandb-group a6000_$(date +%Y%m%d_%H%M%S) \
  --sgd-lr 0.001 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov True \
  --adamw-lr 0.001 \
  --inexact-solver quintic_theoretical \
  --orth-steps 5 \
  --randomized False \
  --rank 100 \
  --oversampling 10 \
  --power-iters 2 \
  --seq-len 2048 \
  "$@"
