from collections import OrderedDict
from typing import List
from torch.utils.data import DataLoader
import torch
import os


from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar


from torchmetrics import Accuracy

from clientcontributionfl.models import train, test
from clientcontributionfl.merkle_root_proof import compute_merkle_tree, compute_merkle_proof
from .fedavg_client import FedAvgClient
from clientcontributionfl.zokrates_proof import Zokrates

class MerkleProofClient(FedAvgClient):
    """
    Client that compute a merkle tree of its dataset. Samples from the dataset can be extracted
    randomly and sent to the server along with a merkle proof which is composed by a root, leaf
    and a path. Server can check that the received samples really belongs to the original dataset,
    thus guaranteeing the integrity and authenticity of the data without needing to possess the entire 
    dataset. 
    """
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        """Initialize the client with contribution scoring parameters.

        Args:
            partition_label_counts: List containing label distribution
            dishonest: Bool parameter, if true client send 
            *args, **kwargs: Arguments passed to parent FedAvgClient
        """
        super().__init__(*args, **kwargs)
        # precompute merkle_tree
        self.merkle_tree = compute_merkle_tree(self.trainloader) # TODO check if shuffled
        
        self.path_proof_dir = os.path.join("proofs", f"client_{self.node_id}")
        self.zk = Zokrates(self.path_proof_dir)


    def _pick_random_batch(self):
        
        batch_index = torch.randint(0, len(self.trainloader), (1,)).item()
        
        # Retrieve the batch at the sampled index
        for i, batch in enumerate(self.trainloader):
            if i == batch_index:
                return batch, batch_index
            
        raise IndexError(f"{batch_index} not in dataloader.")

    def _pick_random_sample(self):
        # Sample a random batch and then a random image from that batch
        (batch_data, batch_labels), batch_index = self._pick_random_batch()
        image_index = torch.randint(0, batch_data.size(0), (1,)).item()  # Random image index in the batch
        return batch_data[image_index], batch_labels[image_index], batch_index


    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.
        Skip training in first round and only return contribution score.
        """
        self.set_parameters(parameters)
        params = {}

        # compute merkle proof 
        if config["server_round"] == 1:

            batch, batch_index = self._pick_random_batch()

            path, direction_selector, leaf = compute_merkle_proof(self.merkle_tree, batch_index)
            
            
            
            #params[f"score_{self.node_id}"] = score
        
        # train the model in any other rounds
        else:
            # learning rate decay
            optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
            train(
                self.model, 
                self.trainloader, 
                self.config['num_epochs'], 
                self.config['device'], 
                optimizer,
                self.criterion,
                self.accuracy_metric
            )

        return self.get_parameters({}), len(self.trainloader), params
