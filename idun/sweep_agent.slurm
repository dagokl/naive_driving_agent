#!/bin/bash
#SBATCH --job-name="hyperparameter-search-naive-driving-agent"
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c28
#SBATCH --time=00-04:00:00
#SBATCH --output=sweep_test_%j.out

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <sweep_id>"
    exit 1
fi

sweep_id="$1"

repo_root=$(git rev-parse --show-toplevel)

module load Python/3.10.8-GCCcore-12.2.0
source "$repo_root/venv/bin/activate"
wandb agent $sweep_id
