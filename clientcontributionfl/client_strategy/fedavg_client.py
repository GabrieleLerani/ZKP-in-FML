from collections import OrderedDict
from typing import Dict
from torch.utils.data import DataLoader
import torch
import os

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from logging import INFO, DEBUG
from flwr.common.logger import log
from torchmetrics import Accuracy

from clientcontributionfl.models import Net, train, test

# relative imports
from clientcontributionfl.utils import compute_contribution, compute_zk_score
from clientcontributionfl import load_data, compute_partition_counts, Zokrates


class FedAvgClient(NumPyClient):
    """Define a Flower Client."""

    def __init__(
        self, 
        node_id: str,
        partition_id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_classes: int,
        trainer_config: Dict[str, Scalar]
    ) -> None:
        super().__init__()

        self.node_id = node_id
        self.partition_id = partition_id
        
        
        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.testloader = testloader

        # a model that is randomly initialised at first
        self.model = Net(num_classes)
        self.trainer_config = trainer_config

        # client training optimizer and criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.trainer_config['lr'])
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.trainer_config['device'])
        


    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """
        Train model received by the server (parameters) using the data
        that belongs to this client. Then, send it back to the server.
        """
        
        self.set_parameters(parameters)

        params = {}
        train(
            self.model, 
            self.trainloader, 
            self.trainer_config['num_epochs'], 
            self.trainer_config['device'], 
            self.optimizer,
            self.criterion,
            self.accuracy_metric
        )

        return self.get_parameters({}), len(self.trainloader), params

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on the data this client has."""
        self.set_parameters(parameters)
        
        # TODO skip evaluate if first round when zk strategy is selected
        loss, accuracy = test(
            self.model,
            self.testloader,
            self.trainer_config['device'],
            self.accuracy_metric
        )

        log(INFO, f"Round: {config['server_round']}, Client {self.node_id[:3]} is doing evaluate() with loss: {loss:.4f} and accuracy: {accuracy:.4f}")

        return float(loss), len(self.testloader), {} # metric
