import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

from flwr.common import NDArrays
from functools import reduce
from enum import Enum, auto
from .file_utils import generate_file_suffix
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets.partitioner import Partitioner
import psutil
import time
from typing import Optional
import csv
from functools import wraps
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ClientData:
    client_files_path: str = ""
    contribution_score: float = 1.0
    proof_valid: bool = False
    contract_address: Optional[str] = None  
    abi: Optional[str] = None  

class SelectionPhase(Enum):
    """Enum to track the current phase of the client selection process"""
    TRAIN_ACTIVE_SET = auto()
    STORE_LOSSES = auto()
    AGGREGATE_FROM_ACTIVE_SET = auto()        
    CANDIDATE_SELECTION = auto()
    SCORE_AGGREGATION = auto() 
    DATASET_SIZE_AGGREGATION = auto() # used only in PoC


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def get_model_class(models, dataset_name):
    if dataset_name in ["MNIST", "FMNIST"]:
        return getattr(models, "NetMnist")
    elif dataset_name == "CIFAR10":
        return getattr(models, "NetCifar10")

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
    """
    alpha_values = [0.03, 0.1, 0.5, 1.0, 2.0]
    plt.figure(figsize=(10, 6))

    for alpha in alpha_values:
        
        file_suffix = f"_S=FedAvg_R={num_rounds}_D={dataset_distribution}_SecAgg={'On' if secaggplus else 'Off'}" + (f"_alpha={alpha}" if dataset_distribution == "dirichlet" else "")
        file_path = Path(save_plot_path) / f"history{file_suffix}.npy"
        
        history = np.load(file_path, allow_pickle=True).item()

        
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
    """
    dataset = config["dataset_name"]
    smoothed_plot = config["smoothed_plots"]
    file_suffix = generate_file_suffix(config)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
    
    result_path = save_plot_path / Path(file_suffix.lstrip('_'))

    for strategy in strategies:
        
        file_path = result_path / f"history_S={strategy}.npy"
        
        history = np.load(file_path, allow_pickle=True).item()
        
        rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
        rounds_loss, values_loss = zip(*history.losses_centralized)
        
        rounds_acc, values_acc = np.asarray(rounds_acc), np.asarray(values_acc)
        rounds_loss, values_loss = np.asarray(rounds_loss), np.asarray(values_loss)
        
        if smoothed_plot:
            window_size = 3
            values_acc = moving_average(values_acc, window_size)
            values_loss = moving_average(values_loss, window_size)
            rounds_acc = rounds_acc[:len(values_acc)]
            rounds_loss = rounds_loss[:len(values_loss)]
        
        ax1.plot(rounds_acc, values_acc, label=strategy)
        ax2.plot(rounds_loss, values_loss, label=strategy)

    ax1.set_title(f"Centralized Validation Accuracy - {dataset}")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    

    ax1.set_ylim([0.1, 1])
    ax2.set_ylim([0, 9])

    if dataset == "MNIST" and "honest" in str(result_path):
        ax1.set_ylim([0.7, 1])
        ax2.set_ylim([0, 4])

    ax2.set_title(f"Centralized Training Loss - {dataset}")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    
    
    plt.tight_layout()
    
    plt.savefig(result_path / "comparison.png")
    plt.close()
    return result_path

def plot_accuracy_for_different_x(save_plot_path: Path, filename: str):
    """
    Plots accuracy from specified paths for different values of X.

    Args:
        save_plot_path: Path to the directory containing the accuracy files.
        filename: The base filename to load the accuracy data from.
    """
    x_values = [0.1, 0.3, 0.5]
    iid_ratio = [0.3, 0.5, 0.7]
    plt.figure(figsize=(8, 10))
    
    
    for x, iid_ratio in zip(x_values, iid_ratio):
        file_path = save_plot_path / f"R=40_P=iid_and_non_iid_D=FMNIST_x=1_iid_ratio={iid_ratio}_bal=False_iid_df={x}/{filename}.npy"
        history = np.load(file_path, allow_pickle=True).item()
        rounds_acc, values_acc = zip(*history.metrics_centralized["accuracy"])
        
        # Apply moving average
        window_size = 4  # You can adjust the window size as needed
        values_acc = moving_average(np.asarray(values_acc), window_size)
        rounds_acc = rounds_acc[:len(values_acc)]  
        
        # Plot the accuracy
        plt.plot(np.asarray(rounds_acc), values_acc, label=f"iid_ratio={iid_ratio}")

    plt.title("PoC centralized accuracy - FMNIST")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.1, 1])
    plt.tight_layout()
    
    plt.savefig(save_plot_path / "accuracy_comparison.png")
    plt.close()


def plot_label_partitioning(partitioner: Partitioner, config: Dict[str, any], num_partitions: int):
    """Plot the label distribution."""
    plot_label_distributions(partitioner, label_name="label", verbose_labels=True, legend=True)
    label_dist_path = os.path.join(config["save_path"], "label_partitioner")
    if not os.path.exists(label_dist_path):
        os.makedirs(label_dist_path)
    plt.savefig(f"{label_dist_path}/{config['partitioner']}_P={num_partitions}.png")



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

def aggregate_between_clusters(results: list[tuple[NDArrays, int, str]], clusters: dict[str, int]) -> NDArrays:
    """Compute weighted average within clusters and then average between clusters."""
    
    # Group results by cluster
    cluster_results = {}
    for weights, num_examples, client_id in results:
        cluster_id = clusters[client_id]
        if cluster_id not in cluster_results:
            cluster_results[cluster_id] = []
        cluster_results[cluster_id].append((weights, num_examples))
    
    # Compute weighted average within each cluster
    cluster_averages = {}
    for cluster_id, cluster_data in cluster_results.items():
        num_examples_total = sum(num_examples for _, num_examples in cluster_data)
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in cluster_data
        ]
        cluster_averages[cluster_id] = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
    
    # Compute the average between clusters
    total_clients = len(results)
    weights_prime: NDArrays = [
        sum(cluster_averages[cluster_id][i] * len(cluster_results[cluster_id]) / total_clients
            for cluster_id in cluster_averages)
        for i in range(len(cluster_averages[next(iter(cluster_averages))]))
    ]
    
    return weights_prime


def generate_zok_client_score_template(classes: int):
    """
    Generate a .zok file with a dynamic classes number.
    """
    template = f"""
const u32 CLASSES = {classes};


// Function to calculate variance of the elements in an array
def variance<N>(field[N] arr, field mean) -> field {{
    field mut var = 0;
    //field mut diff = 0;
    for u32 i in 0..N {{
        // diff = arr[i] - mean; // test it could break prime fields
        var = var + (arr[i] - mean) * (arr[i] - mean);
    }}
    return var;
}}
    

// Function to calculate label diversity (number of different labels)
def diversity<N>(field[N] arr, field thr) -> field {{
    field mut div = 1;
    for u32 i in 0..N {{
        
        //assigns a +1 if client has at least thr elements 
        div = if arr[i] >= thr {{div + 1}} else {{div}};

    }}
    return div;
}}
    

// Main function to compute dataset score
// Inputs: label counts, scale factor, beta (weight)
def main(private field[CLASSES] counts, private field scale, private field beta, private field mean_val, private field thr, private field pre_computed_score) -> field {{
    
    field var = variance(counts, mean_val);
    field div = diversity(counts, thr);
    
    // Calculate the final score (scaled)
    field score = (beta * var) + (div * scale);
    assert(score == pre_computed_score);
    
    return score;
}}
"""
    return template


def generate_zok_merkle_tree_template(tree_depth):
    """
    Generate a .zok file with a dynamic tree depth.
    """
    template = f"""
import "hashes/poseidon/poseidon" as poseidon;

// Define tree depth
const u32 TREE_DEPTH = {tree_depth};

def select(bool condition, field left, field right) -> (field, field) {{
    return (condition ? right : left, condition ? left : right);
}}

def merkleTreeProof<DEPTH>(
    field root, 
    field leaf, 
    bool[DEPTH] directionSelector, 
    field[DEPTH] path
) -> bool {{
    // Start from the leaf
    field mut digest = leaf;

    // Loop up the tree
    for u32 i in 0..DEPTH {{
        (field, field) s = select(directionSelector[i], digest, path[i]);
        digest = poseidon([s.0, s.1]);
    }}

    return digest == root;
}}

// Main function
def main(
    field treeRoot, 
    field leaf, 
    private bool[TREE_DEPTH] directionSelector, 
    private field[TREE_DEPTH] path
) {{
    assert(merkleTreeProof(treeRoot, leaf, directionSelector, path));
}}
"""
    return template


def write_zok_file(filename: str, template, directory: str = None) -> str:
    
    if not directory:
        directory = ""

    #os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    with open(file_path, "w") as f:
        f.write(template)
    return file_path



def measure_cpu_and_time(csv_file="client_metrics.csv"):
    """
    Decorator to measure CPU utilization and execution time (in microseconds) of a function
    and log the results into a CSV file.

    Args:
        csv_file (str): The path to the CSV file where metrics will be saved.
    
    Returns:
        Callable: A wrapped function that logs CPU and time metrics.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure the directory for the CSV file exists
            csv_path = Path(csv_file)
            if not csv_path.parent.exists():
                csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Record initial CPU utilization and time
            start_cpu = psutil.cpu_percent(interval=None)
            start_time = time.time()
            
            # Execute the wrapped function
            result = func(*args, **kwargs)
            
            # Record final CPU utilization and time
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            
            # Calculate the total time in microseconds and average CPU usage
            time_taken_us = (end_time - start_time) * 1_000_000  # Convert to microseconds
            avg_cpu_utilization = (start_cpu + end_cpu) / 2  # Approximation
            
            
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                # Write header if the file is new
                if csv_path.stat().st_size == 0:
                    writer.writerow(["Execution Time (µs)", "Avg CPU Utilization (%)"])
                
                writer.writerow([f"{time_taken_us:.0f}", f"{avg_cpu_utilization:.2f}"])
            
            print(f"Function '{func.__name__}' executed in {time_taken_us:.0f} µs")
            print(f"Average CPU utilization: {avg_cpu_utilization:.2f}%")
            
            return result
        
        return wrapper
    return decorator



def store_deployment_metrics(actor_id: int, cost_wei: int, execution_time: float):

    filepath = Path("deployment_metrics.csv")
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['actor_id', 'cost_wei', 'execution_time'])
        writer.writerow([actor_id, cost_wei, execution_time])

    
def store_contract_function_metrics(cost_wei: int, execution_time: float):

    filepath = Path("contract_function_metrics.csv")
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['cost_wei', 'execution_time'])
        writer.writerow([cost_wei, execution_time])