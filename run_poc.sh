datasets=("FMNIST")
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
    
    cmd="python main.py --strategies PoCZk,PoC --num_rounds 100 --num_nodes $num_nodes --dataset $dataset --iid_ratio $iid_ratio --iid_data_fraction 0.1 --x 1 --d $d_value --smoothed_plot --diversity_thr 140"
    
    $cmd

  done
done
