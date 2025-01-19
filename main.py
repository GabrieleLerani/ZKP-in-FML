import subprocess
import argparse
from clientcontributionfl.utils import cleanup_proofs, plot_comparison_from_files, check_arguments, save_target_accuracy_to_csv, plot_accuracy_for_different_x, save_max_accuracy_to_csv
from flwr.common.config import get_project_config
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(description="Run federated learning simulations.")
    
    try:
        args = get_arguments(parser)
        
    except Exception:
        parser.print_help()
        return 
    
    run_simulations(args)
    
    config = update_configuration(args)

    comparison_path, max_accuracy_path = save_simulation_results(args.strategies, config)
    print(f"You can find simulation results at: {comparison_path} and {max_accuracy_path}")

def list_of_strings(arg):
    return arg.split(',')


def get_arguments(parser : argparse.ArgumentParser):
        
    args = add_parser_arguments(parser)

    if not check_arguments(args):
        raise BaseException("Invalid arguments")
        
    return args


def add_parser_arguments(parser : argparse.ArgumentParser):
    """Parse command line arguments."""
    partitioner_choices = ["linear", "exponential", "dirichlet", "pathological", "square", "iid", "iid_and_non_iid"]
    parser.add_argument("--strategies", type=list_of_strings, default="FedAvg", help="Comma-separated list of strategies to simulate.")
    parser.add_argument("--partitioner", choices=partitioner_choices, default="iid_and_non_iid", help="Type of partitioner.")
    parser.add_argument("--balanced", action="store_true", default=False, help="If True, all partitions will have the same number of samples.")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of rounds for the simulation.")
    parser.add_argument("--num_nodes", type=int, default=10, help="Number of clients for the simulation.")
    parser.add_argument("--fraction_fit", type=float, default=0.3, help="Fraction of clients selected for training.")
    parser.add_argument("--iid_ratio", type=float, default=0.7, help="IID ratio for the dataset.")
    parser.add_argument("--iid_data_fraction", type=float, default=0.5, help="Fraction of total data allocated to IID clients.")
    parser.add_argument("--dishonest", action="store_true", default=False, help="If true, all non-IID nodes are dishonest.")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "FMNIST"], help="Dataset to use for the simulation.")
    parser.add_argument("--x", type=int, default=2, help="Number of labels of non-IID clients.")
    parser.add_argument("--d", type=int, default=5, help="Size of the candidate set used in PoC.")
    parser.add_argument("--thr", type=int, default=400, help="Num of minimum label to have positive contribution.")
    parser.add_argument("--smoothed_plot", action="store_true", default=False, help="If true, plots are computed with a moving average.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate used during training.")
    parser.add_argument("--lr_decay", type=float, default=0.995, help="Learning rate decay value.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
    
    return parser.parse_args()    

def run_simulations(args):
    """Run simulation for each strategy."""
    strategies = args.strategies
    # for strategy in strategies:
    #     cleanup_proofs()
    #     run_simulation(args, strategy)

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
            f'thr={args.thr} '
            f'lr={args.lr} '
            f'decay_per_round={args.lr_decay} '
            f'batch_size={args.batch_size}'
        )
    ]
    
    result = subprocess.run(command, stderr=subprocess.STDOUT, stdout=None, text=True)
    
    if result.returncode != 0:
            raise ValueError(f"Error: {result.stderr.decode()}")
    print(f"Simulation completed for strategy = {strategy}\n")

def update_configuration(args):
    """Overrides configuration with current parameters."""
    config = get_project_config(".")["tool"]["flwr"]["app"]["config"]
    config.update({
        "num_rounds": args.num_rounds,
        "iid_ratio": args.iid_ratio,
        "dataset_name": args.dataset,
        "partitioner": args.partitioner,
        "balanced": args.balanced,
        "dishonest": args.dishonest,
        "iid_data_fraction": args.iid_data_fraction,
        "smoothed_plots": args.smoothed_plot,
        "x_non_iid": args.x,
    })
    return config

def save_simulation_results(strategies, config):
    """Save simulation results."""
    comparison_path = save_comparison(strategies, config)
    max_acc_path = save_training_rounds(strategies, config)
    
    return comparison_path, max_acc_path

def save_comparison(strategies, config):
    dataset = config["dataset_name"]
    dishonest = "dishonest" if config["dishonest"] else "honest"
    results_path = Path(config["save_path"]) / Path("simulation") / Path(dataset) / Path(dishonest) 
    

    simulation_path = plot_comparison_from_files(
        save_plot_path=results_path,
        config=config,
        strategies=strategies
    )
    return simulation_path

def save_training_rounds(strategies, config):
    dataset = config["dataset_name"]
    dishonest = "dishonest" if config["dishonest"] else "honest"
    results_path = Path(config["save_path"]) / Path("simulation") / Path(dataset) / Path(dishonest) 

    # training_rounds_paths = save_target_accuracy_to_csv(
    #     save_csv_path=results_path,
    #     config=config,
    #     strategies=strategies
    # )

    training_rounds_paths = save_max_accuracy_to_csv(
        save_csv_path=results_path,
        config=config,
        strategies=strategies
    )

    return training_rounds_paths
    

if __name__ == "__main__":
    main()

    # results_path = Path("results/simulation/PoC_worst_case")
    # plot_accuracy_for_different_x(results_path, "history_S=PoC") 