from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, LinearPartitioner, ExponentialPartitioner, PathologicalPartitioner, SquarePartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr.common import log
from logging import INFO

import pandas as pd
import numpy as np
from .utils.score import entropy_score

import os
from typing import Dict


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


fds = None # cache FederatedDataset
global_scores = None # cache global scores

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
    return batch


def load_data(config: Dict[str, any], partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader, int]:
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        log(INFO,"Initializing FederatedDataset")
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": get_partitioner(config, num_partitions)},
        )
        
    partition = fds.load_partition(partition_id, "train")

    partition = partition.with_transform(apply_transforms)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_test["train"].with_transform(apply_transforms)
    test_partition = partition_train_test["test"].with_transform(apply_transforms)
    
    batch_size = config["batch_size"]

    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, num_workers=7)
    testloader = DataLoader(test_partition, batch_size=batch_size, num_workers=7)
    return trainloader, testloader, get_num_classes(config["dataset_name"])


def load_centralized_dataset(config: Dict[str, any]) -> tuple[DataLoader, int]:
    dataset = load_dataset(config["dataset_name"])["test"]
    centralized_test_loader = DataLoader(
        dataset.with_transform(apply_transforms), 
        batch_size=config["batch_size"], 
        num_workers=7
    )
    return centralized_test_loader, get_num_classes(config["dataset_name"])


def get_partitioner(cfg: Dict[str, any], num_partitions: int):
    distribution = cfg["distribution"]
    num_classes_per_partition = cfg["num_classes_per_partition"]
    alpha = cfg["alpha"]
    
    if distribution == "linear":
        partitioner = LinearPartitioner(num_partitions=num_partitions)
    elif distribution == "exponential":
        partitioner = ExponentialPartitioner(num_partitions=num_partitions)
    elif distribution == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=alpha, partition_by="label")
    elif distribution == "pathological":
        partitioner = PathologicalPartitioner(num_partitions=num_partitions, partition_by="label", num_classes_per_partition=num_classes_per_partition, class_assignment_mode="deterministic")
    elif distribution == "square":
        partitioner = SquarePartitioner(num_partitions=num_partitions)
    elif distribution == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    else:
        raise ValueError(f"Distribution {distribution} not supported") 
            
    return partitioner

def get_num_classes(name: str) -> int:
    if "mnist" in name:
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
