from typing import List, Dict, Union, Optional
from datasets import Dataset
from collections import defaultdict
from flwr_datasets.partitioner.partitioner import Partitioner
import numpy as np


class LabelBasedPartitioner(Partitioner):
    """
    Partition a dataset with mixed IID and non-IID clients, similar 
    to Wu et al. https://arxiv.org/abs/2012.00661
    """
    def __init__(
            self, 
            num_partitions: int, 
            iid_ratio: float, 
            iid_fraction: float = 1.0,
            balance_partitions: bool = True,
            x: int = 2,
            seed: Optional[int] = 42,
    ):
        """
        Initialize the partitioner.

        Parameters
        ----------
        num_partitions : int
            Total number of partitions (clients).
        iid_ratio : float
            Ratio of clients with IID data.
        iid_fraction : float
            Fraction of total data allocated to IID clients (relative to non-IID clients).
        balance_partitions : bool
            If True, all partitions will have the same number of samples.
        x : int
            Number of classes per non-IID client.
        seed: int
            Seed used for dataset shuffling.
        """
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._iid_ratio = self._init_iid_ratio(iid_ratio)
        self._iid_data_fraction = self._init_iid_fraction(iid_fraction)
        self._balance_partitions = balance_partitions
        self._x = x
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._precomputed_partitions: Dict[int, List[int]] = {}

    @property
    def num_partitions(self) -> int:
        """Return the total number of partitions."""
        return self._num_partitions

    def _init_iid_ratio(self, iid_ratio: float) -> float:
        if not 0.0 <= iid_ratio <= 1.0:
            raise ValueError("IID ratio must be between 0 and 1.")
        return iid_ratio

    def _init_iid_fraction(self, iid_fraction: float) -> float:
        if not 0.0 < iid_fraction <= 1.0:
            raise ValueError("IID fraction must be between 0 (exclusive) and 1 (inclusive).")
        return iid_fraction

    def _get_class_indices(self, dataset: Dataset) -> Dict[int, List[int]]:
        """Get a mapping of class labels to their sample indices."""
        class_indices = defaultdict(list)
        for idx, label in enumerate(dataset["label"]):
            class_indices[label].append(idx)
        return class_indices

    def _precompute_partitions(self) -> None:
        """Precompute partitions and store them in a dictionary."""
        class_indices = self._get_class_indices(self.dataset)
        feature_name = self.dataset.column_names[0]
        all_indices = list(range(len(self.dataset[feature_name])))
        
        # Total number of samples in dataset
        total_samples = len(self.dataset)
        
        # Determine the number of IID and non-IID clients
        num_iid_clients = int(self._num_partitions * self._iid_ratio)
        num_non_iid_clients = self._num_partitions - num_iid_clients
        
        # Calculate the number of samples for IID and non-IID clients
        if self._balance_partitions:
            samples_per_client = total_samples // self._num_partitions
            iid_samples_per_client = samples_per_client
            non_iid_samples_per_client = samples_per_client
        else:
            iid_samples_per_client = int((total_samples * self._iid_data_fraction) / num_iid_clients)
            non_iid_samples_per_client = int((total_samples * (1 - self._iid_data_fraction)) / num_non_iid_clients)

        for partition_id in range(self.num_partitions):
            if partition_id < num_iid_clients:
                # IID setting
                selected_indices = self._rng.choice(
                    all_indices, size=iid_samples_per_client, replace=False
                ).tolist()
            else:
                # Non-IID setting
                num_classes = len(class_indices)
                self._x = min(self._x, num_classes)  # Ensure `x` does not exceed available classes
                selected_classes = self._rng.choice(range(num_classes), size=self._x, replace=False)
                
                selected_indices = []
                for cls in selected_classes:
                    selected_indices += self._rng.choice(
                        class_indices[cls], 
                        size=min(non_iid_samples_per_client // self._x, len(class_indices[cls])),
                        replace=False
                    ).tolist()
                
                # Shuffle to avoid order bias
                selected_indices = self._rng.choice(
                    selected_indices, size=min(non_iid_samples_per_client, len(selected_indices)), replace=False
                ).tolist()

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
    
    def _check_num_partitions_greater_than_zero(self) -> None:
        """Ensure num_partitions is greater than zero."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions must be greater than zero.")