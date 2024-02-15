#!/bin/bash
source scripts/generate_data.bash

echo "Starting training script."
python src/train.py
