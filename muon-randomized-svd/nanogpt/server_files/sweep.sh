#!/usr/bin/env bash
# Launches a wandb agent that pulls hyperparameter combinations from the sweep.
# TODO: adjust cd path + conda env name to match your HPC setup.
cd /home/xcheng328/nanogpt
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon

# $1 = sweep ID (e.g. your-entity/muon-nanogpt/abc123)
# $2 = number of runs per agent (default: 10)
wandb agent --count "${2:-10}" "$1"
