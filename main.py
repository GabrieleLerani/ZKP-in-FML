import subprocess
import argparse
from clientcontributionfl.utils import cleanup_proofs, plot_comparison_from_files, check_arguments
from flwr.common.config import get_project_config
from pathlib import Path

def list_of_strings(arg):
    return arg.split(',')


def run_simulation(strategy, num_rounds, iid_ratio, num_nodes, dishonest):

    if strategy in ["FedAvg", "PoC"]:
        fraction_fit = 0.3
    elif strategy in ["ZkAvg", "ContAvg"]:
        fraction_fit = 1.0

    command = [
        "flower-simulation",
        "--app", ".",
        f"--num-supernodes", str(num_nodes),
        "--run-config", f'num_rounds={num_rounds} strategy="{strategy}" fraction_fit={fraction_fit} iid_ratio={iid_ratio}'
    ]
    
    print(" ".join(command))
    result = subprocess.run(command, stderr=subprocess.STDOUT, stdout=None, text=True)
    
    if result.returncode != 0:
            raise ValueError(f"Error: {result.stderr.decode()}")
    print(f"Simulation completed for strategy = {strategy}\n")

# TODO in toml file check all unused parameters.
def main():
    # 1. clean existing directory of previous simulation
    cleanup_proofs()

    # 2. parse command line arguments
    parser = argparse.ArgumentParser(description="Run federated learning simulations.")
    parser.add_argument("--strategies", type=list_of_strings ,help="Comma-separated list of strategies to simulate. Available are FedAvg, ZkAvg, ContAvg")
    parser.add_argument("--num_rounds", type=int, default=10,help="Number of rounds for the simulation.")
    parser.add_argument("--num_nodes", type=int, default=10, help="Number of clients for the simulation.")
    parser.add_argument("--iid_ratio", type=float, default=0.7, help="IID ratio for the dataset, must be between 0 and 1.")
    parser.add_argument("--dishonest", type=bool, default=False, help="If true all non-IID node under iid_and_non_iid partitioner are dishonest and submit a fake score to the server.")
    args = parser.parse_args()

    if not check_arguments(args):
        parser.print_help()
        return

    strategies = args.strategies
    num_rounds = args.num_rounds
    iid_ratio = args.iid_ratio
    num_nodes = args.num_nodes
    dishonest = args.dishonest

    # 3. run simulation for different strategies
    # for s in strategies:
    #     run_simulation(s, num_rounds, iid_ratio, num_nodes, dishonest)

    # 4. overrides configuration with current parameters 
    config = get_project_config(".")["tool"]["flwr"]["app"]["config"]
    config["num_rounds"] = num_rounds
    config["iid_ratio"] = iid_ratio

    # 5. save simulation results
    results_path = Path(config["save_path"]) 
    plot_comparison_from_files(
        save_plot_path=results_path,
        config=config,
        strategies=strategies
    )

if __name__ == "__main__":
    main()