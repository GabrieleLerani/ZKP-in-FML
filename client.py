from collections import OrderedDict
from typing import Dict
from torch.utils.data import DataLoader
import torch

from flwr.client import NumPyClient, Client
from flwr.common import Context
from flwr.common import NDArrays, Scalar
from model import Net, train, test
from logging import INFO, DEBUG
from flwr.common.logger import log
from torchmetrics import Accuracy
from utils.score import compute_contribution


class FlowerClient(NumPyClient):
    """Define a Flower Client."""

    def __init__(
        self, 
        node_id: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        testloader: DataLoader,
        client_dataset_score: float, # TODO use the score to decide the weight of the client
        num_classes: int,
        trainer_config: Dict[str, Scalar]
    ) -> None:
        super().__init__()

        self.node_id = node_id
        self.dataset_score = client_dataset_score
        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # a model that is randomly initialised at first
        self.model = Net(num_classes, trainer_config)
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
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # train the model
        train(
            self.model, 
            self.trainloader, 
            self.trainer_config['num_epochs'], 
            self.trainer_config['device'], 
            self.optimizer,
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
            self.trainer_config['device'],
            self.accuracy_metric
        )

        log(INFO, f"Round: {config['server_round']}, Client {self.node_id[:3]} is doing evaluate() with loss: {loss:.4f} and accuracy: {accuracy:.4f}")

        contribution = compute_contribution(loss, self.dataset_score, self.trainer_config['gamma'])

        # send client contriubtion to the server the key is the client id
        return float(loss), len(self.valloader), {f"{self.node_id}": float(contribution)}




def generate_client_fn(
        trainloaders: list[DataLoader], 
        valloaders: list[DataLoader], 
        testloaders: list[DataLoader], 
        scores: dict, 
        num_classes: int, 
        trainer_config: Dict[str, Scalar]
    ):
    """
    Return a function that can be used by the VirtualClientEngine 
    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(context: Context) -> Client:
        
        # usally a random number instantiated by the server
        node_id = context.node_id

        # number from 0 up to num clients, corresponds to dataset partitions
        cid = context.node_config["partition-id"]

        return FlowerClient(
            node_id=str(node_id),
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloaders[int(cid)],
            client_dataset_score=scores[int(cid)],
            num_classes=num_classes,
            trainer_config=trainer_config
        ).to_client()

    # return the function to spawn client
    return client_fn