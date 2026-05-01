# Ablation (seq_len=1024, half batch schedule), 1x A6000 48GB, xformers backend.
set -e
export DISABLE_FP8=1
export DISABLE_COMPILE=1
export VAL_BATCH_SIZE=262144
# Smaller seq_len halves per-step batch; scale GRAD_ACCUM_STEPS down to match
# per-GPU per-micro-batch tokens from the seq_len=2048 baseline.
# GRAD_ACCUM_STEPS scaling by GPU count (keep GAS x world_size = 8):
#   1x A6000: GRAD_ACCUM_STEPS=8
#   2x A6000: GRAD_ACCUM_STEPS=4
#   4x A6000: GRAD_ACCUM_STEPS=2
#   8x A6000: GRAD_ACCUM_STEPS=1
export GRAD_ACCUM_STEPS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cd /home/xcheng328/nanogpt                               # TODO: adjust to your path
# source /home/xcheng328/miniconda3/bin/activate muon      # TODO: adjust to your env

NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 5 \
  --log-every 50 \
  --wandb True \
  --wandb-project new_muon-ablation-batch-nanogpt \
  --wandb-group ablation_batch_1024_Sayantan \
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
  --seq-len 1024 \
  "$@"
