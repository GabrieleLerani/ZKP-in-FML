import numpy as np
import matplotlib.pyplot as plt
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
    

    # Compute the inverse of the loss (higher loss should result in lower contribution)
    inverse_loss = 1 / (loss + 1e-8)  # Add a small epsilon to avoid division by zero

    return gamma * inverse_loss + (1 - gamma) * dataset_score




def plot_contribution_function(L_range, d_fixed, gammas):
    plt.figure(figsize=(10, 6))

    for gamma in gammas:
        contributions = [compute_contribution(L, d_fixed, gamma) for L in L_range]
        plt.plot(L_range, contributions, label=f'γ = {gamma}')
        

    plt.xlabel('Loss (L)')
    plt.ylabel('Contribution')
    plt.title(f'Contribution vs Loss (Fixed Dataset Score d = {d_fixed})')
    plt.legend()
    plt.grid(True)
    plt.xlim(0.001)  # Set x-axis limit from 0 to 3 to show full range of loss
    plt.ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 to show full range of contribution

    

    plt.tight_layout()
    plt.show()

    

# # Parameters
# L_values = np.linspace(0.001, 20, 1000)
# d_fixed = 0.75  # Fixed dataset score
# gammas = [0.1 ,0.5, 0.9]

# # Plotting
# plot_contribution_function(L_values, d_fixed, gammas)