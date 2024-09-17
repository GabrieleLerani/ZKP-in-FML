from collections import OrderedDict
from typing import Dict
from torch.utils.data import DataLoader
import torch
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.common import NDArrays, Scalar
from flwr.client import Client
from model import Net
import pytorch_lightning as pl
from logging import INFO, DEBUG
from flwr.common.logger import log

class FlowerClient(NumPyClient):
    """Define a Flower Client."""

    def __init__(
        self, 
        trainloader: DataLoader,
        valloader: DataLoader,
        testloader: DataLoader,
        client_dataset_score: float, # TODO use the score to decide the weight of the client
        num_classes: int,
        trainer_config: Dict[str, Scalar]
    ) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        # a model that is randomly initialised at first
        self.model = Net(num_classes, trainer_config)
        

        # pytorch lightning trainer to train the model and avoid boilerplate code
        self.trainer = pl.Trainer(max_epochs=trainer_config['num_epochs'], enable_progress_bar=False)
        
        

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

        log(DEBUG, f"Client {self.cid} is doing fit() with config: {config}")

        self.trainer.fit(self.model, self.trainloader, self.valloader)

        
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on the data this client has."""
        self.set_parameters(parameters)

        # loss returns a dictionary with the loss value and other metrics
        # logged during the test step of the pl trainer in _shared_eval_step 
        # of model.py
        metrics = self.trainer.test(self.model, self.testloader)

        loss = metrics[0]["test_loss"]
        

        return float(loss), len(self.valloader), {}





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
        
        cid = context.node_config["partition-id"]

        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloaders[int(cid)],
            client_dataset_score=scores,
            num_classes=num_classes,
            trainer_config=trainer_config
        ).to_client()

    # return the function to spawn client
    return client_fn