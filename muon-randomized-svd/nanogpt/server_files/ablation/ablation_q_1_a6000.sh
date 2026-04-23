# Ablation (q=1, orth_steps=1), 1x A6000 48GB, xformers backend.
set -e
export DISABLE_FP8=1
export DISABLE_COMPILE=1
export VAL_BATCH_SIZE=262144
# GRAD_ACCUM_STEPS scaling by GPU count (keep GAS x world_size = 16):
#   1x A6000: GRAD_ACCUM_STEPS=16
#   2x A6000: GRAD_ACCUM_STEPS=8
#   4x A6000: GRAD_ACCUM_STEPS=4
#   8x A6000: GRAD_ACCUM_STEPS=2
export GRAD_ACCUM_STEPS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cd /home/xcheng328/nanogpt                               # TODO: adjust to your path
# source /home/xcheng328/miniconda3/bin/activate muon      # TODO: adjust to your env

NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 5 \
  --log-every 50 \
  --wandb True \
  --wandb-project muon-ablation-q-nanogpt \
  --wandb-group a6000_q_1 \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov True \
  --inexact-solver quintic_theoretical \
  --orth-steps 1 \
  --randomized True \
  --rank 100 \
  --oversampling 10 \
  --power-iters 2 \
  --sgd-lr 0.001 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 2048 \
  "$@"
