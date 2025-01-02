datasets=("FMNIST" "MNIST" "CIFAR10")
iid_ratios=(0.3 0.5 0.7)

num_nodes=10

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  # Loop through each iid_ratio
  for iid_ratio in "${iid_ratios[@]}"; do

    if [[ $iid_ratio == 0.3 || $iid_ratio == 0.5 ]]; then
      d_value=3
    else
      d_value=5
    fi

    # Set the base command
    cmd="python main.py --strategies PoCZk,ZkAvg,ContAvg,FedAvg,CLAvg --num_rounds 100 --num_nodes $num_nodes --dataset $dataset --iid_ratio $iid_ratio --balanced --x 2 --d $d_value"

    # Append additional arguments for datasets other than CIFAR10
    if [[ $dataset == "CIFAR10" ]]; then
      cmd+=" --lr 0.25 --lr_decay 0.985 --batch_size 16"
    fi
    # Print and run the command
    echo "Running: $cmd"
    $cmd
  done
done
