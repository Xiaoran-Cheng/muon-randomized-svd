# muon-randomized-svd

Code for the paper *Analysis of Muon with Nesterov Momentum*, which studies a Muon variant that combines Nesterov-accelerated momentum with an inexact orthogonalization based on randomized SVD. Before each Newton–Schulz iteration the momentum buffer is projected onto a low-rank Gaussian sketch, so the orthogonalization runs on a rank-`k` compressed matrix and is then lifted back to the original space. We analyze the sample complexity of this scheme and validate it empirically on CIFAR-10 and NanoGPT.

## Layout

- [cifar10/](cifar10/) — CIFAR-10 experiments, built on Keller Jordan's `airbench94` 94% speedrun. The main trainer is [cifar10/airbench94_muon.py](cifar10/airbench94_muon.py), which wires `randomized_project` into the Muon update and supports four Newton–Schulz solvers (`cubic_ns_theoretical`, `quintic_ns_theoretical`, `quintic_ns_empirical`, `polar_express`). Per-experiment sweep configs live alongside it (`sweep_e1_solver.yaml`, `sweep_e2_rank.yaml`, …).
- [nanogpt/](nanogpt/) — Modded-NanoGPT speedrun fork with the same randomized-SVD Muon plugged into [nanogpt/train_gpt.py](nanogpt/train_gpt.py). FineWeb-validation loss on 8×H100 is the target metric.
- [../plan/experiment_plan_final.tex](../plan/experiment_plan_final.tex) — full experiment plan and shared hyperparameter defaults.

## Experiments

The five experiments referenced in the paper:

| ID | Sweep | Purpose |
| -- | ----- | ------- |
| E1 | Newton–Schulz solver | Pick the orthogonalization polynomial |
| E2 | Rank `k ∈ {d/16, d/8, d/4, d/2}` | Sample complexity of the sketch |
| E3 | NS steps `q ∈ {1, 3, 5, 7, 9}` | Iteration vs. accuracy trade-off |
| E4 | Batch size | Interaction with stochastic gradient noise |
| E5 | Final comparison | vs. SGD+Nesterov, AdamW, vanilla Muon |

Shared defaults: Nesterov momentum, oversampling `p = 10`, power iteration `h = 1`, provisional `q = 5`, `k = d/8`, CIFAR-10 batch 2000, NanoGPT batch 1024.

## Running

See the per-subproject READMEs for setup and launch instructions:

- [cifar10/README.md](cifar10/README.md)
- [nanogpt/README.md](nanogpt/README.md)
