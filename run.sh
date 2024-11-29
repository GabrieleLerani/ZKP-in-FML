#!/bin/bash

# Define common arguments
STRATEGIES="PoCZk,ZkAvg,ContAvg,FedAvg"
PARTITIONER="iid_and_non_iid"
NUM_ROUNDS=40
NUM_NODES=10
DATASET="FMNIST"
D=5

# Array of iid_ratio values
IID_RATIOS=(0.3 0.5 0.7)

# Loop over the iid_ratio values and run the command
for IID_RATIO in "${IID_RATIOS[@]}"; do
    echo "Running with iid_ratio=${IID_RATIO}..."
    python main.py \
        --strategies "$STRATEGIES" \
        --partitioner "$PARTITIONER" \
        --num_rounds "$NUM_ROUNDS" \
        --num_nodes "$NUM_NODES" \
        --dataset "$DATASET" \
        --iid_ratio "$IID_RATIO" \
        --d "$D"
done
