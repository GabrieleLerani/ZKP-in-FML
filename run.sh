# datasets=("MNIST")
# iid_ratios=(0.3 0.5 0.7)

# # Flag to track the state of the script
# resume=true

# # Loop through each dataset
# for dataset in "${datasets[@]}"; do
#   # Loop through each iid_ratio
#   for iid_ratio in "${iid_ratios[@]}"; do

#     if [[ $iid_ratio == 0.3 || $iid_ratio == 0.5 ]]; then
#       d_value=3
#     else
#       d_value=5
#     fi

#     # Set the base command
#     cmd="python main.py --strategies PoCZk,ZkAvg,ContAvg,FedAvg --num_rounds 100 --num_nodes 10 --dataset $dataset --iid_ratio $iid_ratio --balanced --x 2 --d $d_value"

#     # Append additional arguments for datasets other than CIFAR10
#     if [[ $dataset == "CIFAR10" ]]; then
#       cmd+=" --lr 0.25 --lr_decay 0.985 --batch_size 16"
#     fi

#     # # Check if this is the point to resume
#     # if $resume; then
#     #   if [[ $dataset == "CIFAR10" && $iid_ratio == 0.5 ]]; then
#     #     # Only run the remaining strategy for this point
#     #     cmd="python main.py --strategies FedAvg --num_rounds 100 --num_nodes 10 --dataset $dataset --iid_ratio $iid_ratio --balanced --x 2 --d $d_value --lr 0.25 --lr_decay 0.985 --batch_size 16"
#     #     resume=false  # Disable resume mode after this point
#     #   else
#     #     continue  # Skip previous tasks
#     #   fi
#     # fi

#     # Print and run the command
#     echo "Running: $cmd"
#     $cmd
#   done
# done

cmd="python main.py --strategies FedAvg --num_rounds 100 --num_nodes 10 --dataset FMNIST --iid_ratio 0.7 --dishonest --balanced --x 2 --d 5"
$cmd