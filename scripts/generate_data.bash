#!/bin/bash

if [ -n "$1" ]; then
    export NDA_CONFIG=$1
    echo "Config: $NDA_CONFIG"
else
    echo "No config set, using default. To set config, provide path to the config file as first argument to the script."
fi

echo "Starting training dataset generation"
export ROUTES=longest6
python carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent src/datagen_agent.py \
    --routes carla_garage/leaderboard/data/$ROUTES.xml \
    --scenarios carla_garage/leaderboard/data/scenarios/no_scenarios.json

echo "Starting val dataset generation"
export ROUTES=lav
python carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent src/datagen_agent.py \
    --routes carla_garage/leaderboard/data/$ROUTES.xml \
    --scenarios carla_garage/leaderboard/data/scenarios/no_scenarios.json

