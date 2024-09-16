import numpy as np
import pandas as pd

def entropy_score(row: pd.Series, num_classes: int) -> float:
    """
    Calculate the entropy score for a given distribution of classes.

    This method computes the normalized entropy of the class distribution
    in a partition. The entropy is normalized by the maximum possible
    entropy (uniform distribution) to get a score between 0 and 1.

    Args:
        row (pd.Series): A pandas Series containing the count of samples
                         for each class in a partition.
        num_classes (int): The total number of classes in the dataset.

    Returns:
        float: The normalized entropy score, ranging from 0 (completely
                   imbalanced) to 1 (perfectly balanced).

        Note:
            A small value (1e-10) is added to avoid log(0) when calculating
            entropy.
    """
    max_entropy = np.log(num_classes)
    class_proportions = row / row.sum()
    entropy = -np.sum(class_proportions * np.log(class_proportions + 1e-10))  # add small value to avoid log(0)
    entropy_score = entropy / max_entropy
    return entropy_score