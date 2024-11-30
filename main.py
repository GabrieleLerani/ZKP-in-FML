import subprocess
import argparse
from clientcontributionfl.utils import cleanup_proofs, plot_comparison_from_files, check_arguments
from flwr.common.config import get_project_config
from pathlib import Path

def list_of_strings(arg):
    return arg.split(',')


def run_simulation(args, strategy):

    command = [
        "flower-simulation",           
        "--app", ".",                  
        "--num-supernodes", str(args.num_nodes),  
        "--run-config", (              
            f'num_rounds={args.num_rounds} '
            f'partitioner="{args.partitioner}" '
            f'balanced={str(args.balanced).lower()} '
            f'strategy="{strategy}" '
            f'fraction_fit={args.fraction_fit} '
            f'iid_ratio={args.iid_ratio} '
            f'x_non_iid={args.x} '
            f'iid_data_fraction={args.iid_data_fraction} '
            f'dishonest={str(args.dishonest).lower()} '
            f'dataset_name="{args.dataset}" '
            f'd={args.d} '
            f'thr={args.thr}'
        )
    ]
    
    result = subprocess.run(command, stderr=subprocess.STDOUT, stdout=None, text=True)
    
    if result.returncode != 0:
            raise ValueError(f"Error: {result.stderr.decode()}")
    print(f"Simulation completed for strategy = {strategy}\n")

# TODO in toml file check all unused parameters.
def main():
    # 1. clean existing directory of previous simulation
    cleanup_proofs()

    partitioner_choices = ["linear","exponential","dirichlet","pathological","square","iid","iid_and_non_iid"]

    # 2. parse command line arguments
    parser = argparse.ArgumentParser(description="Run federated learning simulations.")
    parser.add_argument("--strategies", type=list_of_strings, default="FedAvg", help="Comma-separated list of strategies to simulate. Available are FedAvg, ZkAvg, ContAvg")
    parser.add_argument("--partitioner",choices=partitioner_choices, default="iid_and_non_iid", help="Type of partitioner. Check Flower docs for more details.")
    parser.add_argument("--balanced", action="store_true",default=False, help="If True, all partitions will have the same number of samples. Applicable only with iid_and_non_iid partitioner.")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of rounds for the simulation.")
    parser.add_argument("--num_nodes", type=int, default=10, help="Number of clients for the simulation.")
    parser.add_argument("--fraction_fit", type=float, default=0.3, help="Fraction of clients selected for training. Such values can be modified depending on the strategy.")
    parser.add_argument("--iid_ratio", type=float, default=0.7, help="IID ratio for the dataset, must be between 0 and 1.")
    parser.add_argument("--iid_data_fraction", type=float, default=0.5, help="Fraction of total data allocated to IID clients (relative to non-IID clients).")
    parser.add_argument("--dishonest", action="store_true",default=False, help="If true all non-IID node under iid_and_non_iid partitioner are dishonest and submit a fake score to the server.")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "FMNIST"], help="Dataset to use for the simulation.")
    parser.add_argument("--x", type=int, default=2, help="Number of labels of non iid clients.")
    parser.add_argument("--d", type=int, default=5, help="Size of the candidate set used in PoC. Must be: max(CK, 1) <= d <= K where C is the fraction_fit of clients")
    parser.add_argument("--thr", type=int, default=400, help="Num of minimum label to have positive contribution. Used only for ZkAvg, PocZk, ContAvg strategies.")
    parser.add_argument("--smoothed_plot", action="store_true", default=False, help="If true plots are computed with a moving average to highlight the trend.")
    args = parser.parse_args()

    if not check_arguments(args):
        parser.print_help()
        return

    # 3. Run simulation for each strategy
    strategies = args.strategies
    # for strategy in strategies:
    #     run_simulation(args, strategy)

    # 4. overrides configuration with current parameters 
    config = get_project_config(".")["tool"]["flwr"]["app"]["config"]
    
    config.update({
        "num_rounds": args.num_rounds,
        "iid_ratio": args.iid_ratio,
        "dataset_name": args.dataset,
        "partitioner": args.partitioner,
        "balanced": args.balanced,
        "dishonest": args.dishonest,
        "iid_data_fraction": args.iid_data_fraction,
        "smoothed_plots": args.smoothed_plot
    })


    # 5. save simulation results
    results_path = Path(config["save_path"]) 
    plot_comparison_from_files(
        save_plot_path=results_path,
        config=config,
        strategies=strategies
    )

if __name__ == "__main__":
    main()