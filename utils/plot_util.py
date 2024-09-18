import numpy as np
import os
import matplotlib.pyplot as plt

DISTRIBUTIONS = ['linear', 'exponential', 'dirichlet', 'square', 'pathological', 'iid']

def read_scores(plots_folder='plots/scores'):
    scores = {}
    for file_name in os.listdir(plots_folder):
        if file_name.endswith('.npy'):
            distribution_type = file_name.replace('.npy', '')
            scores[distribution_type] = np.load(os.path.join(plots_folder, file_name), allow_pickle=True).item()
    return scores

def plot_scores(scores):
    fig, axes = plt.subplots(len(scores), 2, figsize=(15, 5*len(scores)))
    if len(scores) == 1:
        axes = [axes]
    
    fig.suptitle('Scores and Label Distributions for Different Partitioning Strategies', y=1.02)

    for distribution in DISTRIBUTIONS:
        if distribution in scores:
            i = DISTRIBUTIONS.index(distribution)
            score_dict = scores[distribution]
            client_ids = list(score_dict.keys())
            score_values = list(score_dict.values())

            # Plot scores
            axes[i][0].bar(client_ids, score_values)
            axes[i][0].set_xlabel('Client ID', labelpad=10)
            axes[i][0].set_ylabel('Score')
            axes[i][0].set_title(f'{distribution} - Scores', pad=20)

            # Plot label distribution
            label_dist_path = os.path.join('plots', 'label_dist', f'{distribution}.png')
            if os.path.exists(label_dist_path):
                img = plt.imread(label_dist_path)
                axes[i][1].imshow(img)
                axes[i][1].axis('off')
                axes[i][1].set_title(f'{distribution} - Label Distribution', pad=20)
            else:
                axes[i][1].text(0.5, 0.5, 'Label distribution image not found', 
                                ha='center', va='center')
                axes[i][1].axis('off')

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


