#!/bin/bash
#SBATCH --job-name="train-naive-driving-agent"
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c28
#SBATCH --time=01-00:00:00
#SBATCH --output=train_test_%j.out

repo_root=$(git rev-parse --show-toplevel)

module load Python/3.10.8-GCCcore-12.2.0
source "$repo_root/venv/bin/activate"
python "$repo_root/src/train.py"
