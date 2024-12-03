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

