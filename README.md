# Naive driving agent

Naive Driving Agent is a project that aims to train a driving agent using imitation learning. The agent is trained and evaluated in the CARLA simulator.

## Prerequisites

- Ubuntu 22.04

## Setup

```bash
# Install CARLA and Poetry and set up the Python environment.
./scripts/setup_ubuntu22.bash

# Activate Python virtual environment
poetry shell

# Log into weights & biases for experiment tracking and hyperparameter sweeps
# Required for training unless disabled in config.yaml
wandb login
```

## Configuration

All configuration options are in config.yaml.

## Generate Dataset

The data generation script will start both CARLA and the data generation agent. It is robust to crashes and will restart both and resume where it left off in the case of a restart. 

```bash
./scripts/generate_data.bash
```

## Train Model

```bash
# Train model
./src/train.py

# Convenience script for overnight training that ensures that dataset is generated before training.
./scripts/generate_data_and_train.bash
```

## IDUN

```bash
# Sync repo to idun with rsync
# Replace ntnu_user in the script before running
./idun/sync_repo_to_idun.bash

# Go to IDUN and run setup
# This will create a Python virtual environment and log into W&B
./idun/setup.bash

# Schedule a training job on idun
sbatch idun/train.slurm

# Hyperparameter search with W&B
# Edit sweep.yaml to configure the hyperparameter search
# Start the sweep with
wandb sweep sweep.yaml
# Schedule IDUN to start n agents to do hyperparameter search
./idun/queue_sweep_agents.bash <sweep-id> n
```
