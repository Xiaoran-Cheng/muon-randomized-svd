# CIFAR-10

Training code for the CIFAR-10 experiments, built on [airbench94](https://github.com/KellerJordan/cifar10-airbench). Single entry point: [airbench94_muon.py](airbench94_muon.py). The same script runs all baselines (SGD+Nesterov, AdamW, Muon) and Muon variants (full / randomized × Nesterov / Polyak); behavior is controlled by command-line flags.

## Setup

```bash
conda create -n muon python=3.11 -y
conda activate muon
pip install torch torchvision wandb
```

Trains on a single GPU. Each "trial" is one full 8-epoch run; results are averaged over `--num-trials` trials after a warmup trial.

## Quick start

```bash
python airbench94_muon.py --optimizer-mode muon --epochs 8 --num-trials 1
```

To enable W&B logging, add `--wandb True --wandb-project <name>`.

## Final configurations

The exact flags used to produce the headline results in the paper. Each command runs 50 trials and logs to W&B.

### Muon + Nesterov + randomized Guassian projection

```bash
python airbench94_muon.py \
  --optimizer-mode muon --epochs 8 --num-trials 50 --batch-size 500 \
  --val-every-steps 10 \
  --muon-lr 0.10471931304418906 --muon-momentum 0.7098608967968731 --muon-nesterov True \
  --sgd-momentum 0.85 --sgd-nesterov True \
  --inexact_solver quintic_ns_theoretical --orth-steps 7 \
  --randomized True --rank 128 --oversampling 10 --power-iters 2 \
  --wandb True --wandb-project muon-nesterov-rand
```

## Key flags

| Flag | Meaning |
| ---- | ------- |
| `--optimizer-mode` | `muon`, `sgd`, or `adamw` |
| `--inexact_solver` | `cubic_ns_theoretical`, `quintic_ns_theoretical`, `quintic_ns_empirical`, `polar_express` |
| `--orth-steps` | Newton–Schulz iterations `q` |
| `--randomized` | Enable randomized projection before orthogonalization |
| `--rank` / `--oversampling` / `--power-iters` | Sketch parameters `k`, `p`, `h` |
| `--muon-nesterov` | `True` = Nesterov momentum, `False` = Polyak |
| `--num-trials` | Trials averaged for reported metrics |
| `--val-every-steps` | Validation cadence (in steps) |
| `--wandb` / `--wandb-project` / `--wandb-group` | W&B logging |

Run `python airbench94_muon.py --help` for the full list.
