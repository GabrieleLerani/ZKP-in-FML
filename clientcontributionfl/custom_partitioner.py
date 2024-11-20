from typing import List, Dict, Union
from datasets import Dataset
from collections import defaultdict
from flwr_datasets.partitioner.partitioner import Partitioner
import random

class LabelBasedPartitioner(Partitioner):
    """
    Partition a dataset with mixed IID and non-IID clients, similar 
    to Wu et al. https://arxiv.org/abs/2012.00661
    """
    def __init__(
            self, 
            num_partitions: int, 
            iid_ratio: int, 
            x: int = 2
    ):
        """
        Initialize the partitioner.

        Parameters
        ----------
        num_partitions : int
            Total number of partitions (clients).
        iid_ratio : int
            Ratio of clients with IID data.
        x : int
            Number of classes per non-IID client.
        """
        super().__init__()
        self._num_partitions = num_partitions
        self.iid_ratio = self._init_iid_ratio(iid_ratio)
        self.x = x
        self._precomputed_partitions: Dict[int, List[int]] = {}

    @property
    def num_partitions(self) -> int:
        """Return the total number of partitions."""
        return self._num_partitions

    def _init_iid_ratio(self, iid_ratio : Union[float, int]) -> int:
        if not 0.0 <= iid_ratio <= 1.0:
            raise ValueError("iid ratio must be between 0 and 1")
        return iid_ratio

    def _get_class_indices(self, dataset: Dataset) -> Dict[int, List[int]]:
        """Get a mapping of class labels to their sample indices."""
        class_indices = defaultdict(list)
        for idx, label in enumerate(dataset["label"]):
            class_indices[label].append(idx)
        return class_indices

    def _precompute_partitions(self) -> None:
        """Precompute partitions and store them in a dictionary."""
        class_indices = self._get_class_indices(self.dataset)
        all_indices = list(range(len(self.dataset["image"])))
        num_samples = len(self.dataset) // self._num_partitions
        iid_clients = self._num_partitions * self.iid_ratio
        for partition_id in range(self.num_partitions):
            if partition_id < iid_clients:
                # IID setting
                selected_indices = random.sample(all_indices, num_samples)
            else:
                # Non-IID setting
                num_classes = len(class_indices)
                self.x = min(self.x, num_classes)
                selected_classes = random.sample(range(num_classes), self.x)
                selected_indices = []
                for cls in selected_classes:
                    selected_indices += random.sample(
                        class_indices[cls], min(num_samples // self.x, len(class_indices[cls]))
                    )
                # Shuffle to avoid order bias
                selected_indices = random.sample(selected_indices, num_samples)

            self._precomputed_partitions[partition_id] = selected_indices

    def load_partition(self, partition_id: int) -> Dataset:
        """
        Load a single partition based on the partition ID.

        Parameters
        ----------
        partition_id : int
            The ID of the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            The dataset partition corresponding to the given ID.
        """
        if not self.is_dataset_assigned():
            raise AttributeError("Dataset must be assigned before partitions can be loaded.")

        if not self._precomputed_partitions:
            self._precompute_partitions()

        selected_indices = self._precomputed_partitions[partition_id]

        return self.dataset.select(selected_indices)