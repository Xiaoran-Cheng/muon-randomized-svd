# NanoGPT

Training code for the NanoGPT experiments, built on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Single entry point: [train_gpt.py](train_gpt.py). The same script runs all baselines (SGD+Nesterov, AdamW, Muon) and Muon variants (full / randomized × Nesterov / Polyak). Behavior is controlled by command-line flags.

## Setup

```bash
conda create -n muon python=3.11 -y
conda activate muon
pip install -r requirements.txt
```

Tested on 4×H100 (Lambda). Download the FineWeb-10B tokens once before training:

```bash
python data/cached_fineweb10B.py
```

## Quick start

```bash
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  --optimizer-mode muon --num-trials 1 --log-every 50
```

H100 backend env vars (FA3 leaks memory on this stack, use the xformers fallback):

```bash
export DISABLE_FA3=1
export DISABLE_FP8=0
export DISABLE_COMPILE=1
export GRAD_ACCUM_STEPS=2
export REFERENCE_WORLD_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VAL_BATCH_SIZE=262144
```

To enable W&B logging add `--wandb True --wandb-project <name>` (and optionally `--wandb-entity`, `--wandb-group`).

### Muon + Nesterov + randomized Guassian projection

```bash
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  --optimizer-mode muon --num-trials 5 --log-every 50 \
  --muon-nesterov True --muon-lr 0.032463575100400426 --muon-momentum 0.9664542176931918 \
  --inexact-solver quintic_theoretical --orth-steps 7 \
  --randomized True --rank 200 --oversampling 10 --power-iters 1 \
  --seq-len 2048 \
  --sgd-lr 0.001 --sgd-momentum 0.95 --sgd-nesterov True --adamw-lr 0.001 \
  --wandb True --wandb-project muon-nesterov-rand-nanogpt-h100-final
```

## Key flags

| Flag | Meaning |
| ---- | ------- |
| `--optimizer-mode` | `muon`, `adamw`, or `sgd_nesterov` |
| `--inexact-solver` | `cubic`, `quintic_theoretical`, `quintic_empirical`, `polar_express` |
| `--orth-steps` | Newton–Schulz iterations `q` |
| `--randomized` | Enable randomized projection before orthogonalization |
| `--rank` / `--oversampling` / `--power-iters` | Sketch parameters `k`, `p`, `h` |
| `--projector` | Sketch family: `gaussian` (Halko) or `kaczmarz` |
| `--muon-nesterov` | `True` = Nesterov, `False` = Polyak |
| `--seq-len` | Middle factor of total batch (uniformly scales batch across stages) |
| `--num-trials` | Trials per invocation, each with a fresh init |
| `--log-every` | Train-step cadence for val_loss eval and logging |
| `--wandb` / `--wandb-project` / `--wandb-entity` / `--wandb-group` | W&B logging |

