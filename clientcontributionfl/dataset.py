from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner, 
    DirichletPartitioner, 
    LinearPartitioner, 
    ExponentialPartitioner, 
    PathologicalPartitioner, 
    SquarePartitioner
)
from .custom_partitioner import LabelBasedPartitioner
from clientcontributionfl.utils.train_utils import plot_label_partitioning
from flwr.common import log
from logging import INFO
from collections import Counter
from typing import Dict



fds = None # cache FederatedDataset
node_dataloader = {} # cache node partitions
partition_class_counts = {} # cache number of partitions


MNIST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

CIFAR_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def get_transform_fn(dataset_name : str):
    
    def apply_transforms_cifar(batch):
        """Apply transforms to the partition from FederatedDataset."""
        feature = list(batch.keys())[0] # take the column name of image
        batch[feature] = [CIFAR_TRANSFORMS(img) for img in batch[feature]]
        return batch

    def apply_transforms_mnist(batch):
        """Apply transforms to the partition from FederatedDataset."""
        feature = list(batch.keys())[0] # take the column name of image
        batch[feature] = [MNIST_TRANSFORMS(img) for img in batch[feature]]
        return batch

    if dataset_name in ["MNIST", "FMNIST"]:
        return apply_transforms_mnist
    elif dataset_name == "CIFAR10":
        return apply_transforms_cifar


def flwr_dataset_name(dataset: str):
    if dataset == "MNIST":
        return "ylecun/mnist"
    elif dataset == "CIFAR10":
        return "uoft-cs/cifar10"
    elif dataset == "FMNIST":
        return "zalando-datasets/fashion_mnist"


def load_data(config: Dict[str, any], partition_id: int, num_partitions: int) -> tuple[DataLoader, DataLoader, int]:
    """Load client dataset using flower FederatedDataset and partitioners."""
    global fds
    if fds is None:
        initialize_federated_dataset(config, num_partitions)

    batch_size = config["batch_size"]
    trainloader, testloader = get_data_loaders(fds, partition_id, batch_size, config["dataset_name"])

    return trainloader, testloader, get_num_classes(config["dataset_name"])


def initialize_federated_dataset(config: Dict[str, any], num_partitions: int):
    """Initialize the FederatedDataset."""
    global fds
    log(INFO, "Initializing FederatedDataset")
    partitioner = get_partitioner(config, num_partitions)
    fds = FederatedDataset(
        dataset=flwr_dataset_name(config["dataset_name"]),
        partitioners={"train": partitioner},
    )
    if config["plot_label_distribution"]:
        plot_label_partitioning(fds.partitioners["train"], config, num_partitions)

def get_data_loaders(fds: FederatedDataset, partition_id: int, batch_size: int, dataset_name: str) -> tuple:
    """Split the partition into train and test sets and return the corresponding dataloaders."""
    global node_dataloader
    if partition_id not in node_dataloader:
        transform_fn = get_transform_fn(dataset_name)
        partition = fds.load_partition(partition_id, "train").with_transform(transform_fn)
        train_test_split = partition.train_test_split(test_size=0.2, seed=42)
        trainloader = DataLoader(train_test_split["train"], batch_size=batch_size, shuffle=True, num_workers=7)
        testloader = DataLoader(train_test_split["test"], batch_size=batch_size, shuffle=False, num_workers=7)
        node_dataloader[partition_id] = trainloader, testloader
    
    return node_dataloader[partition_id]

def load_centralized_dataset(config: Dict[str, any]) -> tuple[DataLoader, int]:
    dataset_name = flwr_dataset_name(config["dataset_name"])
    dataset = load_dataset(dataset_name)["test"]
    transform_fn = get_transform_fn(config["dataset_name"])
    centralized_test_loader = DataLoader(
        dataset.with_transform(transform_fn), 
        batch_size=config["batch_size"], 
        num_workers=7
    )
    return centralized_test_loader, get_num_classes(config["dataset_name"])


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
            iid_fraction = cfg["iid_data_fraction"]
        )
    else:
        raise ValueError(f"Partitioner {partitioner} not supported") 
            
    return partitioner

def get_num_classes(name: str) -> int:
    name = flwr_dataset_name(name)
    if "mnist" in name or "cifar10" in name:
        return 10
    elif "cifar100" in name:
        return 100
    else:
        raise ValueError(f"Dataset {name} not supported")
    

def compute_partition_counts(
        data_loader: DataLoader, 
        partition_id: int,
        num_classes: int
):
    global partition_class_counts
    if partition_id not in partition_class_counts:

        label_counter = count_labels(data_loader)
        partition_class_counts[partition_id] = create_sorted_counts(label_counter, num_classes)

    return partition_class_counts[partition_id]


def count_labels(data_loader: DataLoader) -> Counter:
    """Count labels in the DataLoader."""
    label_counter = Counter()
    for batch in data_loader:
        # when dataset is from FlowerDataset
        labels = batch["label"]
        label_counter.update(labels.tolist())
    return label_counter


def create_sorted_counts(label_counter: Counter, num_classes: int) -> list[int]:
    """Create a sorted list of counts for each class."""
    sorted_counts = [0] * num_classes
    for i in range(num_classes):
        sorted_counts[i] = label_counter[i]
    return sorted_counts