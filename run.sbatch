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

set -eu
: "${MILES_CAMPAIGN_ROOT:?Set the unique campaign directory; pass --output to sbatch}"
exec python3 "${MILES_CAMPAIGN_ROOT}/code/coordinator.py"
