#!/bin/bash
#SBATCH --job-name=miles-terminal-lego
#SBATCH --partition=gpu-nodes
#SBATCH --nodes=4
#SBATCH --nodelist=gpu-nodes-[0-3]
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/preflight/slurm-%j.out

set -eu
exec python3 /shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/code/coordinator.py
