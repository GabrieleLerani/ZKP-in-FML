import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def compute_score(counts: List[int], scale: int, beta: int, thr: int) -> int:
    """
    Compute the score associated with a partition, which will be used as a witness for a Zero-Knowledge (ZK) proof.

    This function calculates a score based on the variance of the counts, the diversity of the counts above a threshold,
    and a scaling factor. The score is intended to be used as a witness in a ZK proof to verify the contribution of a client
    without revealing the actual data.

    Args:
        counts (List[int]): A list of integers representing the counts of contributions from a client.
        scale (int): A scaling factor used to weight the diversity.
        beta (int): A weight factor that determines the importance of variance in the score calculation.
        thr (int): A threshold value used to determine the diversity of the counts.

    Returns:
        int: The computed score as an integer.

    Example:
        >>> counts = [300, 100, 6000, 1000, 4000]
        >>> scale = 1000
        >>> beta = 1
        >>> thr = 100
        >>> score = compute_zk_score(counts, scale, beta, thr)
        >>> print(score)
        16000000

    The score is calculated as follows:
        1. Compute the total sum of the counts.
        2. Calculate the mean value of the counts.
        3. Compute the variance of the counts from the mean value.
        4. Calculate the diversity as the number of counts greater than or equal to the threshold plus one.
        5. Compute the final score as the sum of the weighted variance and the scaled diversity.

    Formula:
        score = (beta * variance) + (diversity * scale)

    Note:
        If the label counts are uniformly distributed in the list, the score is very low, indicating identically and independently
        distributed data. Conversely, when there are few labels, the variance is very high, resulting in a higher score.
    """
    total = sum(counts)
    mean_val = total // len(counts)
    variance = sum((count - mean_val) ** 2 for count in counts)
    diversity = sum(1 for count in counts if count >= thr) + 1
    score = (beta * variance) + (diversity * scale)
    return int(score)

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
    return entropy_score if entropy_score > 0 else -entropy_score

def compute_contribution(loss: float, dataset_score: float, gamma: float = 0.5) -> float:
    """
    Compute the contribution level of a client based on its evaluation loss and dataset score.
    A higher gamma value assigns greater importance to the loss, while a lower gamma value places more emphasis on the dataset distribution.

    Args:
        loss (float): The evaluation loss of the client.
        dataset_score (float): The dataset score of the client, in the range [0, 1].
        alpha (float): Weighting factor for the loss and dataset score. Default is 0.5.

    Returns:
        float: The computed contribution level of the client.
    """
    
    # TODO
    # Compute the inverse of the loss (higher loss should result in lower contribution)
    inverse_loss = 1 / (loss + 1e-8)  # Add a small epsilon to avoid division by zero

    return (1 - gamma) * dataset_score
    #return gamma * inverse_loss + (1 - gamma) * dataset_score
