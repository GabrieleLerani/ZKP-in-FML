from typing import List
import torch
from clientcontributionfl.models import train
from clientcontributionfl.utils import compute_score
from clientcontributionfl.client_strategy import FedAvgClient
from logging import INFO, DEBUG
from flwr.common.logger import log


class ContributionClient(FedAvgClient):
    """
    Clients that compute its contribution score and send to the server.
    When client is dishonest can commit a fake contribution score and being 
    considered for the next round of training.
    """
    def __init__(
        self,
        partition_label_counts: List,
        dishonest: bool,
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
        self.partition_label_counts = partition_label_counts
        self.dishonest = dishonest
        self.scale = self.config["scale"]
        self.beta = self.config["beta"]
        self.thr = self.config["thr"]


    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.
        Skip training in first round and only return contribution score.
        """
        self.set_parameters(parameters)
        params = {}

        # compute dataset score on the first round
        if config["server_round"] == 1:
            # compute dataset score
            score = compute_score(
                counts=self.partition_label_counts, 
                scale=self.scale, 
                beta=self.beta, 
                thr=self.thr
            )

            # add additional value
            if self.dishonest:
                score = 0 #TODO score + self.config["dishonest_contribution"]
                

            params[f"score_{self.node_id}"] = score
        
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
