from torchvision import transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, LinearPartitioner, ExponentialPartitioner, PathologicalPartitioner, SquarePartitioner
from flwr_datasets.visualization import plot_label_distributions
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import entropy_score
from matplotlib import pyplot as plt
import os

class DatasetLoader:
    def __init__(
            self, 
            num_partitions, 
            dist="linear", 
            plot_label_distribution=True, 
            alpha=1, 
            batch_size=32,
            plots_folder="plots"
        ):
        self.num_partitions = num_partitions
        self.distribution = dist
        self.plot_label_distribution = plot_label_distribution
        self.alpha = alpha
        self.batch_size = batch_size
        self.fds = None
        self.plots_folder = plots_folder
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
            elif self.distribution == "iid":
                partitioner = IidPartitioner(num_partitions=self.num_partitions)
            else:
                raise ValueError(f"Distribution {self.distribution} not supported") 
            
            self.fds = FederatedDataset(
                dataset="ylecun/mnist",
                partitioners={"train": partitioner},
            )


            if self.plot_label_distribution:
                partitioner = self.fds.partitioners["train"]
                plot_label_distributions(partitioner=partitioner, label_name=f"label", verbose_labels=True)        
                
                label_dist_path = os.path.join(self.plots_folder, "label_dist")
                if not os.path.exists(label_dist_path):
                    os.makedirs(label_dist_path)
                plt.savefig(f"{label_dist_path}/{self.distribution}.png")
                

    def compute_partition_score(self) -> dict[int, float]:
        """
        Compute a score for each partition based on class distribution and entropy.

        This method calculates a score for each partition in the federated dataset. The score
        is based on two factors:
        1. The relative size of the partition compared to the largest partition.
        2. The entropy of the class distribution within the partition.

        The final score is the average of these two factors, resulting in a value between 0 and 1.
        A higher score indicates a more balanced and diverse partition.

        Returns:
            dict[int, float]: A dictionary mapping partition IDs to their computed scores.
        """
        # Count the number of samples for each class in each partition
        partition_class_counts = {
            partition_id: self.fds.load_partition(partition_id, "train")
                .select_columns(['label'])
                .to_pandas()['label']
                .value_counts()
                .to_dict()
            for partition_id in range(self.num_partitions)
        }
        
        # Create a DataFrame from the class counts and fill missing values with 0
        df = pd.DataFrame(partition_class_counts).T.fillna(0)
        num_classes = df.shape[1]
        max_samples_per_partition = df.sum(axis=1).max()
        
        # Compute scores for each partition
        scores = {
            partition_id: (row.sum() / max_samples_per_partition + entropy_score(row, num_classes)) / 2
            for partition_id, row in df.iterrows()
        }

        # Save scores to a file
        label_dist_path = os.path.join(self.plots_folder, "scores")
        if not os.path.exists(label_dist_path):
            os.makedirs(label_dist_path)
        
        file_path = os.path.join(label_dist_path, f"{self.distribution}.npy")
        np.save(file_path, scores)

        return scores


    
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