set -e
export DISABLE_FP8=1 # A6000 cannot use FP8
export DISABLE_COMPILE=1 # if there is only xformers package but not FA2, needs to set to 1；if there is FA2, set to 0

# small test
export SMOKE_VAL_TOKENS=1024
export SMOKE_VAL_TOKENS_TOTAL=8192
export NUM_ITERATIONS=50
export NUM_EXT_ITERATIONS=5


cd /home/xcheng328/nanogpt  # cd to the nanogpt folder
source /home/xcheng328/miniconda3/bin/activate muon  # activate venv


torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --optimizer-mode muon \
  --num-trials 1 \
  --log-every 25 \
  --wandb False \
  --muon-lr 0.023 \
  --muon-momentum 0.95 \
  --muon-nesterov True \
  --inexact-solver polar_express \
  --orth-steps 3 \
  --randomized False \
  --rank 100 \
  --oversampling 10 \
  --power-iters 0 \
  --sgd-lr 0.003 \
  --sgd-momentum 0.95 \
  --sgd-nesterov True \
  --adamw-lr 0.001 \
  --seq-len 1024 \
  "$@"
