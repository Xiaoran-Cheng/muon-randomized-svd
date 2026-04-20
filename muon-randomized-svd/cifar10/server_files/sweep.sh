#!/usr/bin/env bash
cd /home/xcheng328/cifar10
export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"
source /home/xcheng328/miniconda3/bin/activate muon

# $1 = sweep ID (e.g. your-entity/cifar10-muon/abc123)
# $2 = number of runs per agent (default: 10)
wandb agent --count "${2:-10}" "$1"
