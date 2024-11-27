from collections import OrderedDict
from typing import Dict
from torch.utils.data import DataLoader
import torch

import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from logging import INFO, DEBUG
from flwr.common.logger import log
from torchmetrics import Accuracy

from clientcontributionfl.models import train, test

# relative imports


class FedAvgClient(NumPyClient):
    """Define a standard client acting in FedAvg strategy."""

    def __init__(
        self, 
        node_id: str,
        partition_id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_classes: int,
        model_class: nn.Module,
        config: Dict[str, Scalar]
    ) -> None:
        """
        Initialize the FedAvgClient.

        Args:
            node_id (str): Unique identifier for the client node.
            partition_id (int): Identifier for the data partition assigned to this client.
            trainloader (DataLoader): DataLoader for the training data.
            testloader (DataLoader): DataLoader for the testing data.
            num_classes (int): Number of classes in the classification task.
            config (Dict[str, Scalar]): Configuration dictionary for the trainer.
        """
        super().__init__()

        self.node_id = node_id
        self.partition_id = partition_id
        
        
        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.testloader = testloader

        # a model that is randomly initialised at first
        self.model = model_class(num_classes)
        self.config = config

        # client training optimizer and criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.config['device'])
        


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

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on the data this client has."""
        self.set_parameters(parameters)
        
        
        loss, accuracy = test(
            self.model,
            self.testloader,
            self.config['device'],
            self.accuracy_metric
        )

        log(INFO, f"Round: {config['server_round']}, Client {self.node_id[:3]} is doing evaluate() with loss: {loss:.4f} and accuracy: {accuracy:.4f}")

        return float(loss), len(self.testloader), {} # metric
