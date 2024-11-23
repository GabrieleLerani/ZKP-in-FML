import subprocess
from clientcontributionfl.utils import cleanup_proofs, plot_comparison_from_files

from flwr.common.config import get_project_config
from pathlib import Path

def run_simulation(strategy):

    if strategy == "FedAvg":
        fraction_fit = 0.3
    elif strategy == "ZkAvg" or strategy == "ContAvg":
        fraction_fit = 1.0
        
    command = [
        "flower-simulation",
        "--app", ".",
        "--num-supernodes", "10",
        "--run-config", f'strategy="{strategy}" fraction_fit={fraction_fit}'
    ]
    
    print(" ".join(command))
    result = subprocess.run(command, stderr=subprocess.STDOUT, stdout=None, text=True)
    
    if result.returncode != 0:
            raise ValueError(f"Error: {result.stderr.decode()}")
    print(f"Simulation completed for strategy = {strategy}\n")

# TODO in toml file check all unused parameters.
def main():
    # 1. clean existings directory of previous simulation
    cleanup_proofs()

    # 2. run simulation for different strategies
    strategies = ["ZkAvg","ContAvg","FedAvg"]
    for s in strategies:
        run_simulation(s)

    # 3. plot result
    config = get_project_config(".")["tool"]["flwr"]["app"]["config"]
    
    #save_path = Path("clientcontributionfl/plots/results")
    results_path = Path(config["save_path"]) 
    plot_comparison_from_files(
        results_path,
        config,
        strategies=strategies
    )

if __name__ == "__main__":
    main()