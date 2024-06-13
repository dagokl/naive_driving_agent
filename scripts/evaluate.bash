#!/bin/bash

pkill -2 CarlaUE4
sleep 4
pkill -9 CarlaUE4
sleep 2


# export ROUTES=lav
export ROUTES=lav
export BENCHMARK=lav
# export ROUTES=longest6_only_first
# export BENCHMARK=longest6

if [ -n "$1" ]; then
    export NDA_CONFIG=$1
    echo "Config: $NDA_CONFIG"
else
    echo "No config set, using default. To set config, provide path to the config file as first argument to the script."
    echo "Need config for SAVE_PATH"
    exit 1
fi

config_filename="${NDA_CONFIG##*/}" 
config_name="${config_filename%.yaml}"
export RESULT_DIR="/home/dag/src/naive_driving_agent/eval_results/$config_name/$BENCHMARK"

echo $RESULT_DIR

echo "Starting CARLA"
(
    cd ./carla_garage/carla;
    ./CarlaUE4.sh -opengl &
)
sleep 4

echo "Starting eval run"
python carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --agent src/eval_agent.py \
    --routes carla_garage/leaderboard/data/$ROUTES.xml \
    --scenarios carla_garage/leaderboard/data/scenarios/no_scenarios.json

echo "Killing CARLA"
pkill -2 CarlaUE4
sleep 4
pkill -9 CarlaUE4
sleep 2

mkdir -p $RESULT_DIR
mv ./simulation_results.json "$RESULT_DIR/simulation_results.json"

CARLA_GARAGE_DIR=./carla_garage

python $CARLA_GARAGE_DIR/tools/result_parser.py --xml ${CARLA_GARAGE_DIR}/leaderboard/data/lav.xml --results $RESULT_DIR --log_dir $RESULT_DIR
