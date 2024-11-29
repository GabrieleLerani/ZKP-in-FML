import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from flwr.server.history import History
from typing import Optional, List
from flwr.common.config import get_project_config
from flwr.common import (NDArrays)
from functools import reduce
from enum import Enum, auto

def get_model_class(models, dataset_name):
    if dataset_name in ["MNIST", "FMNIST"]:
        return getattr(models, "NetMnist")
    elif dataset_name == "CIFAR10":
        return getattr(models, "NetCifar10")

class SelectionPhase(Enum):
    """Enum to track the current phase of the client selection process"""
    TRAIN_ACTIVE_SET = auto()
    STORE_LOSSES = auto()
    AGGREGATE_FROM_ACTIVE_SET = auto()        
    CANDIDATE_SELECTION = auto()
    SCORE_AGGREGATION = auto() # TODO   

def string_to_enum(enum_class: SelectionPhase, enum_str: str) -> SelectionPhase:
    """Convert a given string the intended enum class, used only in PoC"""
    try:
        if "." in enum_str:
            _, member_name = enum_str.split(".")
            return getattr(enum_class, member_name)
        else:
            raise ValueError("Enum string must contain a dot separating the class and member name.")
    except AttributeError:
        raise ValueError(f"{member_name} is not a valid member of {enum_class}.")

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
        
        file_suffix = f"_S=FedAvg_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}" + (f"_alpha={alpha}" if dataset_distribution == "dirichlet" else "")
        file_path = Path(save_plot_path) / f"history{file_suffix}.npy"
        
        history = np.load(file_path, allow_pickle=True).item()

        # Plot centralized accuracy
        rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
        plt.plot(np.asarray(rounds_acc), np.asarray(values_acc), label=f"α = {alpha}")

    plt.title(f"Centralized Validation Accuracy - MNIST (FedAvg, {dataset_distribution})")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.1, 1])

    plt.tight_layout()
    plt.savefig(Path(save_plot_path) / Path(f"alpha_comparison_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}.png"))
    plt.close()



def plot_comparison_from_files(save_plot_path: Path, config: dict[str, any], strategies: List[str]):
    """
    Read numpy files for strategies and plot their accuracy and loss.

    Parameters
    ----------
    save_plot_path : Path
        Directory to save the plot to.
    config : dict[str, any]
        Configuration dictionary containing simulation parameters
    strategies : List[str]
        List of strategies to compare
    """

    num_rounds=config['num_rounds']
    partitioner=config['partitioner']
    secaggplus=config['secaggplus']
    alpha=config['alpha']
    x_non_iid = config["x_non_iid"]
    iid_ratio = config["iid_ratio"]
    dishonest = config["dishonest"]
    dataset = config["dataset_name"]

    include_alpha = (f"_alpha={alpha}" if partitioner == "dirichlet" else "")
    include_x = (f"_x={x_non_iid}" if partitioner == "iid_and_non_iid" else "")
    include_iid_ratio = (f"_iid_ratio={iid_ratio}" if partitioner == "iid_and_non_iid" else "")
    include_dishonest = (f"_dishonest" if dishonest else "")
    include_sec_agg = ("SecAgg" if secaggplus else "")

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    file_suffix = (
        f"R={num_rounds}"
        f"_P={partitioner}"
        f"_D={dataset}"
        + include_sec_agg
        + include_alpha 
        + include_x 
        + include_iid_ratio 
        + include_dishonest
    )

    result_path = save_plot_path / Path("simulation") / Path(file_suffix.lstrip('_'))

    for strategy in strategies:
        
        file_path = result_path / f"history_S={strategy}.npy"
        
        history = np.load(file_path, allow_pickle=True).item()
        
        # Plot centralized accuracy
        rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
        ax1.plot(np.asarray(rounds_acc), np.asarray(values_acc), label=strategy)

        # Plot centralized loss
        rounds_loss, values_loss = zip(*history.losses_centralized)
        
        ax2.plot(np.asarray(rounds_loss), np.asarray(values_loss), label=strategy)

    ax1.set_title(f"Centralized Validation Accuracy - {dataset}")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.set_ylim([0.1, 1])

    ax2.set_title(f"Centralized Training Loss - {dataset}")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.set_ylim([0, 5])

    plt.tight_layout()
    
    plt.savefig(result_path / "comparison.png")
    plt.close()


# TODO check if this should be removed
def read_scores(plots_folder='plots/scores'):
    scores = {}
    for file_name in os.listdir(plots_folder):
        if file_name.endswith('.npy'):
            distribution_type = file_name.replace('.npy', '')
            scores[distribution_type] = np.load(os.path.join(plots_folder, file_name), allow_pickle=True).item()
    return scores


def aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime



