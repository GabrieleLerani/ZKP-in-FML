from torchvision import transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, LinearPartitioner, ExponentialPartitioner, PathologicalPartitioner, SquarePartitioner
from flwr_datasets.visualization import plot_label_distributions
from matplotlib import pyplot as plt
from tqdm import tqdm

class DatasetLoader:
    def __init__(self, num_partitions, dist="linear", plot_label_distribution=True, alpha=1, batch_size=32):
        self.num_partitions = num_partitions
        self.distribution = dist
        self.plot_label_distribution = plot_label_distribution
        self.alpha = alpha
        self.batch_size = batch_size
        self.fds = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self._initialize_federated_dataset()

    def _apply_transforms(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [self.transform(img) for img in batch["image"]]
        return batch

    def _initialize_federated_dataset(self):
        if self.fds is None:
            
            if self.distribution == "linear":
                partitioner = LinearPartitioner(num_partitions=self.num_partitions)
            elif self.distribution == "exponential":
                partitioner = ExponentialPartitioner(num_partitions=self.num_partitions)
            elif self.distribution == "dirichlet":
                partitioner = DirichletPartitioner(num_partitions=self.num_partitions, alpha=self.alpha, partition_by="label")
            elif self.distribution == "pathological":
                partitioner = PathologicalPartitioner(num_partitions=self.num_partitions, partition_by="label", num_classes_per_partition=2, class_assignment_mode="deterministic")
            elif self.distribution == "square":
                partitioner = SquarePartitioner(num_partitions=self.num_partitions)
            else:
                partitioner = IidPartitioner(num_partitions=self.num_partitions)
            
            self.fds = FederatedDataset(
                dataset="ylecun/mnist",
                partitioners={"train": partitioner},
            )
            partitioner = self.fds.partitioners["train"]
            
            if self.plot_label_distribution:
                plot_label_distributions(partitioner=partitioner, label_name=f"label {self.distribution}", verbose_labels=True)        
                plt.savefig(f"plots/label_dist_{self.distribution}.png")
            

    def load_data(self) -> tuple[list[DataLoader], list[DataLoader], list[DataLoader]]:
        

        train_loaders = []
        val_loaders = []
        test_loader = []


        for partition_id in tqdm(range(self.num_partitions), desc="Loading partitions"):
            partition = self.fds.load_partition(partition_id, "train")
            partition = partition.with_transform(self._apply_transforms)

            # Split data: 20% for federated evaluation, 60% for federated train, 20% for federated validation
            partition_full = partition.train_test_split(test_size=0.2, seed=42)
            partition_train_valid = partition_full["train"].train_test_split(train_size=0.75, seed=42)

            trainloader = DataLoader(
                partition_train_valid["train"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
            )
            valloader = DataLoader(
                partition_train_valid["test"],
                batch_size=self.batch_size,
                num_workers=2,
            )
            testloader = DataLoader(partition_full["test"], batch_size=self.batch_size, num_workers=1)

            train_loaders.append(trainloader)
            val_loaders.append(valloader)
            test_loader.append(testloader)

        return train_loaders, val_loaders, test_loader