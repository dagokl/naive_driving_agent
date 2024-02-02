#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sweep_id> <number_of_jobs>"
    exit 1
fi

repo_root=$(git rev-parse --show-toplevel)

sweep_id="$1"
num_jobs="$2"
for ((i=1; i<=$num_jobs; i++)); do
    job_name="hyperparameter-search-naive-driving-agent-$i"

    sbatch --job-name="$job_name" "$repo_root/idun/sweep_agent.slurm" "$sweep_id"
done
