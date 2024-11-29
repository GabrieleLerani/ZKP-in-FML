from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, LinearPartitioner, ExponentialPartitioner, PathologicalPartitioner, SquarePartitioner
from .custom_partitioner import LabelBasedPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr.common import log
from logging import INFO
from collections import defaultdict, Counter


import hashlib
import pandas as pd
import numpy as np
from .utils.score import entropy_score

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


fds = None # cache FederatedDataset
node_partitions = None # cache node partitions

global_scores = None # cache global scores
partition_class_counts = {} # cache number of partitions

def flwr_dataset_name(dataset: str):
    if dataset == "MNIST":
        return "ylecun/mnist"
    elif dataset == "CIFAR10":
        return "uoft-cs/cifar10"
    elif dataset == "FMNIST":
        return "zalando-datasets/fashion_mnist"

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    feature = list(batch.keys())[0] # take the column name of image
    batch[feature] = [TRAIN_TRANSFORMS(img) for img in batch[feature]]
    return batch



def load_data(config: Dict[str, any], partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader, int]:
    """Load client dataset using flower FederatedDataset and partitioners."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        log(INFO,"Initializing FederatedDataset")
        partitioner = get_partitioner(config, num_partitions)
        fds = FederatedDataset(
            dataset=flwr_dataset_name(config["dataset_name"]),
            partitioners={"train": partitioner},
        )
        if config["plot_label_distribution"]:
            partitioner = fds.partitioners["train"]
            plot_label_distributions(partitioner ,label_name=f"label",verbose_labels=True, legend=True)
            label_dist_path = os.path.join(config["save_path"], "label_dist")
            if not os.path.exists(label_dist_path):
                os.makedirs(label_dist_path)
            plt.savefig(f"{label_dist_path}/{config['partitioner']}_P={num_partitions}.png")

    partition = fds.load_partition(partition_id, "train")

    partition = partition.with_transform(apply_transforms)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # TODO check if with_transform is really necessary again since it's already applied above
    train_partition = partition_train_test["train"].with_transform(apply_transforms)
    test_partition = partition_train_test["test"].with_transform(apply_transforms)
    
    batch_size = config["batch_size"]

    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, num_workers=7)
    testloader = DataLoader(test_partition, batch_size=batch_size, num_workers=7)
    return trainloader, testloader, get_num_classes(flwr_dataset_name(config["dataset_name"]))


def load_centralized_dataset(config: Dict[str, any]) -> tuple[DataLoader, int]:
    dataset_name = flwr_dataset_name(config["dataset_name"])
    dataset = load_dataset(dataset_name)["test"]
    centralized_test_loader = DataLoader(
        dataset.with_transform(apply_transforms), 
        batch_size=config["batch_size"], 
        num_workers=7
    )
    return centralized_test_loader, get_num_classes(flwr_dataset_name(config["dataset_name"]))


def get_partitioner(cfg: Dict[str, any], num_partitions: int):
    partitioner = cfg["partitioner"]
    
    if partitioner == "linear":
        partitioner = LinearPartitioner(num_partitions=num_partitions)
    elif partitioner == "exponential":
        partitioner = ExponentialPartitioner(num_partitions=num_partitions)
    elif partitioner == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha= cfg["alpha"], partition_by="label")
    elif partitioner == "pathological":
        partitioner = PathologicalPartitioner(num_partitions=num_partitions, partition_by="label", num_classes_per_partition=cfg["num_classes_per_partition"], class_assignment_mode="deterministic")
    elif partitioner == "square":
        partitioner = SquarePartitioner(num_partitions=num_partitions)
    elif partitioner == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    elif partitioner == "iid_and_non_iid":
        partitioner = LabelBasedPartitioner(
            num_partitions=num_partitions,
            iid_ratio = cfg["iid_ratio"],
            x = cfg["x_non_iid"],
            balance_partitions=cfg["balanced"],
            iid_fraction = cfg["iid_data_fraction"]#0.3 # TODO pass as argument.
        )
    else:
        raise ValueError(f"Partitioner {partitioner} not supported") 
            
    return partitioner

def get_num_classes(name: str) -> int:
    if "mnist" in name or "cifar10" in name:
        return 10
    elif "cifar100" in name:
        return 100
    else:
        raise ValueError(f"Dataset {name} not supported")
    


def compute_partition_score(
        num_partitions : int, 
        num_classes : int, 
        distribution : str, 
        save_path : str
    ) -> dict[int, float]:
    """
    Compute a score for each partition based on class distribution and entropy.

    This function calculates a score for each partition in the federated dataset. The score
    is based on two factors:
    1. The relative size of the partition compared to the largest partition.
    2. The entropy of the class distribution within the partition.

    The final score is the average of these two factors, resulting in a value between 0 and 1.
    A higher score indicates a more balanced and diverse partition.

    Returns:
        dict[int, float]: A dictionary mapping partition IDs to their computed scores.
    """
    global global_scores

    if global_scores is None and fds is not None:
        # Count the number of samples for each class in each partition
        partition_class_counts = {
            partition_id: fds.load_partition(partition_id, "train")
                .select_columns(['label'])
                .to_pandas()['label']
                .value_counts()
                .to_dict()
            for partition_id in range(num_partitions)
        }
        
        # Create a DataFrame from the class counts and fill missing values with 0
        df = pd.DataFrame(partition_class_counts).T.fillna(0)
        max_samples_per_partition = df.sum(axis=1).max()
        
        # Compute scores for each partition
        global_scores = {
            partition_id: (row.sum() / max_samples_per_partition + entropy_score(row, num_classes)) / 2
            for partition_id, row in df.iterrows()
        }

        # Save scores to a file
        label_dist_path = os.path.join(save_path, "scores")
        if not os.path.exists(label_dist_path):
            os.makedirs(label_dist_path)
        
        file_path = os.path.join(label_dist_path, f"{distribution}.npy")
        np.save(file_path, global_scores)

    return global_scores



def compute_partition_counts(
        data_loader: DataLoader, 
        partition_id: int,
        num_classes: int
):
    global partition_class_counts
    if partition_id not in partition_class_counts:

        # Initialize a counter for labels
        label_counter = Counter()

        # Iterate through the DataLoader to count labels
        for batch in data_loader:
            # when dataset is from FlowerDataset
            labels = batch["label"]
            label_counter.update(labels.tolist())

        
        # Create a sorted list of counts, one for each class
        partition_class_counts[partition_id] = [0] * num_classes
        
        for i in range(num_classes):
            partition_class_counts[partition_id][i] = label_counter[i]

    return partition_class_counts[partition_id]



def hash(data: bytes) -> bytes:
    """
    Computes the SHA-256 hash of the given data.
    """
    return hashlib.sha256(data).digest()

def hash_to_u32(val: bytes) -> str:
    """Converts a SHA-256 hash into a sequence of 32-bit integers."""
    M0 = val.hex()[:128]
    b0 = [str(int(M0[i:i+8], 16)) for i in range(0, len(M0), 8)]
    return " ".join(b0)

def build_merkle_tree(leaves: list[bytes]) -> list[list[bytes]]:
    """Constructs a Merkle tree and returns all levels."""
    tree = [leaves]  # Start with the leaf nodes
    current_level = leaves

    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            # Combine pairs of hashes to form the next level
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                # Handle odd number of nodes (duplicate last node)
                combined = current_level[i] + current_level[i]            

            next_level.append(hash(combined))
        tree.append(next_level)
        current_level = next_level

    return tree


def get_merkle_proof(tree: list[list[bytes]], leaf_index: int) -> Tuple[list[bytes], list[int]]:
    """Generates the Merkle proof for a given leaf index."""
    proof = []
    directions = []
    for level in range(len(tree) - 1):
        sibling_index = leaf_index ^ 1  # XOR with 1 to get the sibling index
        directions.append(leaf_index % 2)  # 0 if left, 1 if right
        proof.append([tree[level][sibling_index],(level,sibling_index)])
        leaf_index //= 2
    return proof, directions

def compute_merkle_root(dataloader: DataLoader) -> str:
    """
    Computes the Merkle root of all the dataset images in a PyTorch DataLoader.

    Args:
        dataloader: PyTorch DataLoader containing the dataset.
    
    Returns:
        The Merkle root as a hex string.
    """
    # Step 1: Hash each image in the dataset
    leaf_hashes = []
    for batch in dataloader:
        feature, label = list(batch.keys())
        images, labels = batch[feature], batch[label]

        # Flatten images and convert to bytes
        for image, label in zip(images, labels):
            image_bytes = image.numpy().tobytes()
            label_bytes = label.numpy().tobytes()
            combined_data = image_bytes + label_bytes
            #leaf_hashes.append(hash(int.to_bytes(combined_data, 64, "big")))
            leaf_hashes.append(hash(combined_data))
            
    
    # Step 2: Build the Merkle tree
    merkle_tree = build_merkle_tree(dataloader)

    root = merkle_tree[-1][0]
    
    # Step 3: Return the root hash
    return root