import subprocess
from clientcontributionfl.utils import cleanup_proofs, plot_comparison_from_files

from flwr.common.config import get_project_config
from pathlib import Path

def run_simulation(strategy):
    command = [
        "flower-simulation",
        "--app", ".",
        "--num-supernodes", "10",
        "--run-config", f'num_rounds=2 strategy="{strategy}"'
    ]
    
    print(" ".join(command))
    result = subprocess.run(command, stderr=subprocess.STDOUT, stdout=None, text=True)
    
    if result.returncode != 0:
            raise ValueError(f"Error: {result.stderr.decode()}")
    print(f"Simulation completed for strategy = {strategy}\n")


def main():
    # 1. clean existings directory of previous simulation
    cleanup_proofs()

    # 2. run simulation for different strategies
    strategies = ["ZkAvg", "FedAvg"]
    for s in strategies:
        run_simulation(s)

    # 3. plot result
    config = get_project_config(".")["tool"]["flwr"]["app"]["config"]
    # check not required
    file_suffix = (
        f"_S={config['strategy']}"
        f"_R={config['num_rounds']}"
        f"_D={config['distribution']}"
        f"_SecAgg={'On' if config['secaggplus'] else 'Off'}"
        f"_alpha={config['alpha']}"
    )

    # TODO check str and path already used in plot_util
    save_path = str(Path("clientcontributionfl/plots/results"))
    plot_comparison_from_files(
        save_path,
        config['num_rounds'],
        config['distribution'],
        config['secaggplus'],
        config['alpha'],
        strategies=strategies
    )

if __name__ == "__main__":
    main()