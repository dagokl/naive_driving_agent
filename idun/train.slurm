#!/bin/bash
#SBATCH --job-name="train-naive-driving-agent"
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=slurm_out/train_%j.out

repo_root=$(git rev-parse --show-toplevel)

if [ -n "$1" ]; then
    export NDA_CONFIG=$1
    echo "Config: $NDA_CONFIG"
else
    echo "No config set, using default. To set config, provide path to the config file as first argument to the script."
fi

module load Python/3.10.8-GCCcore-12.2.0
source "$repo_root/venv/bin/activate"
python "$repo_root/src/train.py"
