import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from flwr.server.history import History
from typing import Optional, List
from flwr.common.config import get_project_config

def load_history(file_path: str):
    loaded_array = np.load(file_path, allow_pickle=True).item()
    return loaded_array

def plot_for_varying_alphas(save_plot_path: Path, num_rounds: int, dataset_distribution: str, secaggplus: bool):
    """
    Read numpy files for FedAvg strategy with different alpha values and plot their centralized accuracy.

    Parameters
    ----------
    save_plot_path : Path
        Folder to save the plot to.
    num_rounds : int
        Number of rounds in the simulation.
    dataset_distribution : str
        Distribution of the dataset used.
    secaggplus : bool
        Whether SecAgg+ was used or not.
    """
    alpha_values = [0.03, 0.1, 0.5, 1.0, 2.0]
    plt.figure(figsize=(10, 6))

    for alpha in alpha_values:
        file_suffix = f"_S=FedAvg_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}_alpha={alpha}"
        file_path = Path(save_plot_path) / f"history{file_suffix}.npy"
        
        history = np.load(file_path, allow_pickle=True).item()

        # Plot centralized accuracy
        rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
        plt.plot(np.asarray(rounds_acc), np.asarray(values_acc), label=f"α = {alpha}")

    plt.title(f"Centralized Validation Accuracy - MNIST (FedAvg, {dataset_distribution})")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.2, 1])

    plt.tight_layout()
    plt.savefig(Path(save_plot_path) / Path(f"alpha_comparison_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}.png"))
    plt.close()


def plot_comparison_from_files(save_plot_path: Path, num_rounds: int, dataset_distribution: str, secaggplus: bool, alpha: float, strategies: List[str]):
    """
    Read numpy files for FedAvg and ContFedAvg strategies and plot their accuracy and loss.

    Parameters
    ----------
    save_plot_path : Path
        Folder to save the plot to.
    num_rounds : int
        Number of rounds in the simulation.
    dataset_distribution : str
        Distribution of the dataset used.
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for strategy in strategies:
        file_suffix = f"_S={strategy}_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}_alpha={alpha}"
        file_path = Path(save_plot_path) / f"history{file_suffix}.npy"
        
        history = np.load(file_path, allow_pickle=True).item()

        # Plot accuracy
        rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
        ax1.plot(np.asarray(rounds_acc), np.asarray(values_acc), label=f"{strategy} - Accuracy")

        # Plot loss
        rounds_loss, values_loss = zip(*history.losses_distributed)
        ax2.plot(np.asarray(rounds_loss), np.asarray(values_loss), label=f"{strategy} - Loss")

    ax1.set_title("Centralized Validation Accuracy - MNIST")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.set_ylim([0.4, 1])

    ax2.set_title("Distributed Training Loss - MNIST")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(Path(save_plot_path) / Path(f"strategy_comparison_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}_alpha={alpha}.png"))
    plt.close()

def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    strategy: str,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    strategy : str
        Name of the strategy used.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    metric: str
        Metric to plot. Can be "accuracy" or "loss".
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot centralized accuracy
    rounds_acc, values_acc = zip(*hist.metrics_centralized["accuracy"])
    ax1.plot(np.asarray(rounds_acc), np.asarray(values_acc), label=f"{strategy}")
    ax1.set_ylim([0.2, 1])
    ax1.set_title("Centralized Validation Accuracy - MNIST")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")

    # Plot distributed loss
    rounds_loss, values_loss = zip(*hist.losses_distributed)
    ax2.plot(np.asarray(rounds_loss), np.asarray(values_loss), label=f"{strategy}", color='red')
    ax2.set_title("Distributed Training Loss - MNIST")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(Path(save_plot_path) / Path(f"combined_metrics{suffix}.png"))
    plt.close()

# TODO check if this should be removed
def read_scores(plots_folder='plots/scores'):
    scores = {}
    for file_name in os.listdir(plots_folder):
        if file_name.endswith('.npy'):
            distribution_type = file_name.replace('.npy', '')
            scores[distribution_type] = np.load(os.path.join(plots_folder, file_name), allow_pickle=True).item()
    return scores



if __name__ == "__main__":
    config = get_project_config(".")["tool"]["flwr"]["app"]["config"]
    file_suffix = (
        f"_S={config['strategy']}"
        f"_R={config['num_rounds']}"
        f"_D={config['distribution']}"
        f"_SecAgg={'On' if config['secaggplus'] else 'Off'}"
        f"_alpha={config['alpha']}"
    )

    save_path = str(Path("clientcontributionfl/plots/results"))
    # load_path = str(Path("clientcontributionfl/plots/results") / Path(f"history{file_suffix}.npy"))
    # loaded_history = load_history(load_path)
    # plot_metric_from_history(
    #     loaded_history,
    #     save_path,
    #     config['strategy'],
    #     file_suffix,
    # )

    # plot_comparison_from_files(
    #     save_path,
    #     config['num_rounds'],
    #     config['distribution'],
    #     config['secaggplus'],
    #     config['alpha']
    # )

    plot_for_varying_alphas(
        save_path,
        config['num_rounds'],
        config['distribution'],
        config['secaggplus'],
        
    )