#!/bin/bash

# Define the arrays for datasets and iid_ratio values
datasets=("FMNIST" "MNIST")
iid_ratios=(0.3 0.5 0.7)
#PoCZk,ZkAvg,
# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Loop through each iid_ratio
  for iid_ratio in "${iid_ratios[@]}"; do
    # Set the value of d based on the iid_ratio
    if [[ $iid_ratio == 0.3 || $iid_ratio == 0.5 ]]; then
      d_value=3
    else
      d_value=5
    fi
    
    # Set the base command
    cmd="python main.py --strategies ContAvg,FedAvg --num_rounds 200 --num_nodes 10 --dataset $dataset --iid_ratio $iid_ratio --balanced --dishonest --x 2 --d $d_value"
    
    # Append additional arguments for datasets other than CIFAR10
    if [[ $dataset == "CIFAR10" ]]; then
      cmd+=" --lr 0.25 --lr_decay 0.99 --batch_size 16"
    fi
    
    # Print and run the command
    echo "Running: $cmd"
    $cmd
  done
done
