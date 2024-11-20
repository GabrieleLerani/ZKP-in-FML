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

import random
import pandas as pd
import numpy as np
from .utils.score import entropy_score

import os
from typing import Dict
import matplotlib.pyplot as plt

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


fds = None # cache FederatedDataset
node_partitions = None # cache node partitions

global_scores = None # cache global scores
partition_class_counts = {} # cache number of partitions


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
    return batch


def partition_dataset_mixed(dataset, num_clients=10, samples_per_client=600, iid_clients=8, x_non_iid=2, batch_size=32, test_split=0.2):
    """
    Partition a dataset with mixed IID and non-IID clients, similar 
    to Wu et al. https://arxiv.org/abs/2012.00661
    Args:
        dataset: A PyTorch Dataset object with a `targets` attribute.
        num_clients: Total number of clients.
        samples_per_client: Number of samples per client.
        iid_clients: Number of clients using IID partitioning.
        x_non_iid: Number of classes for non-IID clients.
        batch_size: Batch size for the DataLoader.
        test_split: Proportion of samples to reserve for testing.

    Returns:
        A list of tuples (train_loader, test_loader, classes) for each client.
    """
    

    global node_partitions
    # Nodes partition
    if node_partitions is None: 

        # Group indices of dataset by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(dataset.targets):
            class_indices[int(label)].append(int(idx))

        node_partitions = []
        num_classes = len(class_indices)
        assert 0 <= x_non_iid <= num_classes, "x labels must be less than num classes."
        if x_non_iid == 0:
            iid_clients = num_clients

        for i in range(num_clients):
            if i < iid_clients:
                # IID client: Randomly sample from the entire dataset
                all_indices = list(range(len(dataset)))
                selected_indices = random.sample(all_indices, samples_per_client)
            else:
                # Non-IID client: Select x classes and sample from them
                selected_classes = random.sample(range(num_classes), x_non_iid)
                selected_indices = []
                for cls in selected_classes:
                    
                    selected_indices += random.sample(class_indices[cls], samples_per_client // x_non_iid)

                # Adjust for uneven split
                remaining = samples_per_client - len(selected_indices)
                if remaining > 0:
                    additional_indices = random.sample(
                        [i for cls in selected_classes for i in class_indices[cls]],
                        remaining
                    )
                    selected_indices += additional_indices

            
            # Shuffle selected indices to add randomness
            random.shuffle(selected_indices)

            # Get unique classes for this client
            selected_labels = [int(dataset.targets[idx]) for idx in selected_indices]
            unique_classes = set(selected_labels)

            # Split into train and test sets
            split = int((1 - test_split) * len(selected_indices))
            train_indices = selected_indices[:split]
            test_indices = selected_indices[split:]

            # Create DataLoaders
            train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)

            node_partitions.append((train_loader, test_loader, 10)) # mnist dataset returns 10 labels

    return node_partitions

def load_data_mixed(config: Dict[str, any], partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader, int]:
    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=TRAIN_TRANSFORMS)

    batch_size = config["batch_size"]
    iid_ratio = config["iid_ratio"]
    x_non_iid = config["x_non_iid"]
    # Partition MNIST dataset   
    node_data = partition_dataset_mixed(
        dataset=mnist_dataset,
        num_clients=num_partitions,
        samples_per_client=6000, # TODO set as hyper params
        iid_clients=int(num_partitions * iid_ratio),  # 8 clients are IID
        x_non_iid=x_non_iid,    # Non-IID clients use x=2
        batch_size=batch_size 
    )[partition_id]
    return node_data


def load_data(config: Dict[str, any], partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader, int]:
    """Load client dataset using flower FederatedDataset and partitioners."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        log(INFO,"Initializing FederatedDataset")
        partitioner = get_partitioner(config, num_partitions)
        fds = FederatedDataset(
            dataset=config["dataset_name"],
            partitioners={"train": partitioner},
        )
        if config["plot_label_distribution"]:
            partitioner = fds.partitioners["train"]
            plot_label_distributions(partitioner, label_name=f"label", verbose_labels=True)
            label_dist_path = os.path.join(config["save_path"], "label_dist")
            if not os.path.exists(label_dist_path):
                os.makedirs(label_dist_path)
            plt.savefig(f"{label_dist_path}/{config['distribution']}_P={num_partitions}.png")

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
    
    if distribution == "linear":
        partitioner = LinearPartitioner(num_partitions=num_partitions)
    elif distribution == "exponential":
        partitioner = ExponentialPartitioner(num_partitions=num_partitions)
    elif distribution == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha= cfg["alpha"], partition_by="label")
    elif distribution == "pathological":
        partitioner = PathologicalPartitioner(num_partitions=num_partitions, partition_by="label", num_classes_per_partition=cfg["num_classes_per_partition"], class_assignment_mode="deterministic")
    elif distribution == "square":
        partitioner = SquarePartitioner(num_partitions=num_partitions)
    elif distribution == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    elif distribution == "iid_and_non_iid":
        partitioner = LabelBasedPartitioner(
            num_partitions=num_partitions,
            iid_ratio = cfg["iid_ratio"],
            x = cfg["x_non_iid"]
        )
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