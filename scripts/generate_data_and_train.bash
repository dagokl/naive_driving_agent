#!/bin/bash

CARLA_ROOT=$CARLA_ROOT
if [ -z "$CARLA_ROOT" ]; then
    echo "Error: Enviorment variable CARLA_ROOT not set."
    exit 1
fi

while true; do

    echo "Starting Carla."
    "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen &
    sleep 2

    echo "Starting data generation script."
    python src/generate_episodes.py


    # Check exit status of python script
    if [ $? -eq 0 ]; then
        echo "Data generation completed successfully."
        break
    else
        echo "Data generation failed. Will retry until successfull."
    fi

    # Killing carla
    pkill -2 CarlaUE4
    sleep 10
done

echo "Starting training script."
python src/train.py
