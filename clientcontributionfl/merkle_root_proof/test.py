import time
import sys
import csv
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from clientcontributionfl.merkle_root_proof import *
from clientcontributionfl import Zokrates
from clientcontributionfl.utils import generate_zok_merkle_tree_template
from clientcontributionfl.utils import write_zok_file
from tqdm import tqdm
from pprint import PrettyPrinter


MNIST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

CIFAR_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def compute_merkle_tree_build_time(proofs_path : str):
    
    datasets_info = {
        "MNIST": datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORMS),
        "FMNIST": datasets.FashionMNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORMS),
        "CIFAR10": datasets.CIFAR10(root="./data", train=True, download=True, transform=CIFAR_TRANSFORMS)
    }
    
    batch_sizes = [8, 16, 32, 64, 128]
    
    results = {}
    
    zk = Zokrates(working_dir=proofs_path)

    for dataset_name, dataset in tqdm(datasets_info.items(), desc="Datasets"):
        print(f"Analyzing dataset: {dataset_name}")
        for batch_size in tqdm(batch_sizes, desc="Batch Sizes", leave=False):
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Measure time to build the Merkle tree
            merkle_tree, build_tree_elapsed_time = measure_execution_time(compute_merkle_tree, trainloader)
            
            leaf_index=0
            # Measure time to compute the Merkle proof
            res, proof_time = measure_execution_time(compute_merkle_proof, merkle_tree, leaf_index)
            proof, direction_selector, leaf = res
            

            tree_depth = len(proof)
            template = generate_zok_merkle_tree_template(tree_depth)
            
            filename = f"merkle_proof_{tree_depth}_{dataset_name}.zok"
            
            write_zok_file(
                directory=temp_proof_path, 
                filename=filename,
                template=template
            )
            
            arguments = format_proof_arguments(merkle_tree, proof, direction_selector, leaf)
            
            # Measure time to setup and generate the proof
            start_setup_time = time.time()
            zk.setup(zok_file_path=filename)
            zk.generate_proof(arguments)
            setup_time = time.time() - start_setup_time
            

            # Get tree depth and total number of nodes
            tree_depth = len(merkle_tree)
            total_nodes = sum(len(level) for level in merkle_tree)
            occupancy = calculate_memory(merkle_tree)
            
            results[(dataset_name, batch_size)] = {
                "build_tree_time": build_tree_elapsed_time,
                "compute_proof_time": proof_time * 10**6,  
                "zokrates_compile_time": setup_time,    
                "tree_depth": tree_depth,
                "total_nodes": total_nodes,
                "occupancy": occupancy
            }

    return results

def save_results_to_csv(results, file_path : str, filename: str= 'merkle_tree_experiments.csv'):
    file_path = os.path.join(file_path, filename)
    with open(file_path, mode='w', newline='') as csvfile:
        fieldnames = ['dataset_name', 'batch_size', 'build_tree_time', 'compute_proof_time', 'zokrates_compile_time', 'tree_depth', 'total_nodes', 'occupancy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()

        for (dataset_name, batch_size), metrics in tqdm(results.items(), desc="Saving Results"):
            
            writer.writerow({
                'dataset_name': dataset_name,
                'batch_size': batch_size,
                'build_tree_time': metrics['build_tree_time'],
                'compute_proof_time': metrics['compute_proof_time'],
                'zokrates_compile_time': metrics['zokrates_compile_time'],
                'tree_depth': metrics['tree_depth'],
                'total_nodes': metrics['total_nodes'],
                'occupancy': metrics['occupancy']
            })



def calculate_memory(obj):
    """
    Recursively calculate the memory usage of an object, including nested objects.

    Args:
        obj: The object to calculate memory usage for.

    Returns:
        int: Total memory usage in bytes.
    """
    seen_ids = set()  

    def inner(o):
        if id(o) in seen_ids:  
            return 0
        seen_ids.add(id(o))
        
        size = sys.getsizeof(o)
        
        if isinstance(o, dict):
            size += sum(inner(k) + inner(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set)):
            size += sum(inner(i) for i in o)
        return size

    return inner(obj)


def measure_execution_time(func, *args, **kwargs):
    """
    Measure the execution time of a given function.

    Args:
        func: The function to measure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: A tuple containing the result of the function and the elapsed time in seconds.
    """
    start_time = time.time()  
    result = func(*args, **kwargs)  
    elapsed_time = time.time() - start_time
    return result, elapsed_time


if __name__ == "__main__":

    temp_proof_path = "clientcontributionfl/merkle_root_proof/temp_proofs"
    os.makedirs(temp_proof_path, exist_ok=True)

    results = compute_merkle_tree_build_time(temp_proof_path)
    PrettyPrinter(indent=4).pprint(results)

    result_path = "results/merkle_tree_proofs"
    os.makedirs(result_path, exist_ok=True)

    save_results_to_csv(results, result_path)

    # Remove the temporary directory
    shutil.rmtree(temp_proof_path)