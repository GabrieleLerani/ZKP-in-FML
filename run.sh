datasets=("FMNIST")
iid_ratios=(0.7)

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
    # cmd="python main.py --strategies PoCZk,ZkAvg,ContAvg,FedAvg,FedAdam,FedAvgM,FedProx --num_rounds 100 --num_nodes $num_nodes --dataset $dataset --iid_ratio $iid_ratio --balanced --x 2 --d $d_value"
    cmd="python main.py --strategies ContAvg --num_rounds 100 --num_nodes $num_nodes --dataset $dataset --iid_ratio $iid_ratio --balanced --x 2 --d $d_value"

    # Append additional arguments for datasets other than CIFAR10
    if [[ $dataset == "CIFAR10" ]]; then
      cmd+=" --lr 0.25 --lr_decay 0.985 --batch_size 16"
    fi
    # echo "Running without --dishonest: $cmd"
    # $cmd

    Run with --dishonest flag
    cmd+=" --dishonest"
    echo "Running with --dishonest: $cmd"
    $cmd
  done
done
