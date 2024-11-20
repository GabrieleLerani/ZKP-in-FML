from collections import OrderedDict
from typing import Dict
from torch.utils.data import DataLoader
import torch
import os

from flwr.client import NumPyClient, Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from flwr.common import NDArrays, Scalar

from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr.common.config import get_project_config
from torchmetrics import Accuracy

# relative imports
from .model import Net, train, test
from .utils.score import compute_contribution, compute_zk_score
from .dataset import load_data, compute_partition_score, compute_partition_counts, load_data_mixed
from .zokrates_proof import Zokrates


class FlowerClient(NumPyClient):
    """Define a Flower Client."""

    def __init__(
        self, 
        node_id: str,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_dataset_score: float, 
        partition_label_counts: list,
        num_classes: int,
        trainer_config: Dict[str, Scalar]
    ) -> None:
        super().__init__()

        self.node_id = node_id
        self.dataset_score = client_dataset_score
        self.partition_label_counts = partition_label_counts

        # zk param
        self.scale = trainer_config["scale"]
        self.beta = trainer_config["beta"]
        self.thr = trainer_config["thr"]

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
        
        self.path_proof_dir = os.path.join("proofs", f"client_{self.node_id}")
        self.zk = Zokrates(self.path_proof_dir)

    def compute_zkp_contribution(self, score):
        # compute the mean of the label distribution
        counts = self.partition_label_counts
        mean_val = int(sum(counts) / len(counts))
        
        # setup zero knowledge proof through zokrates
        self.zk.setup()

        # generate the proof
        self.zk.generate_proof(
            counts=counts, 
            scale=self.scale, 
            beta=self.beta, 
            mean_val=mean_val, 
            thr = self.thr, 
            score=score
        )



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

        params = {}
        # If first round send path of verification key, in a real scenario one should send directly
        # the verification key
        if config["server_round"] == 1:
            # TODO decide to not train the model in this first stage and send empty parameters
            score = compute_zk_score(
                counts=self.partition_label_counts, 
                scale=self.scale, 
                beta=self.beta, 
                thr=self.thr
            )
            self.compute_zkp_contribution(score)
            # send verification key and claimed contribution --> {str: Tuple[str, int]}
            params[f"vrfkey_{self.node_id}"] = self.path_proof_dir
            params[f"score_{self.node_id}"] = score
        
        # train the model in any other rounds
        else:
            
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

        # contribution = compute_contribution(loss, self.dataset_score, self.trainer_config['gamma'])
        
        if config['server_round'] == 1:
            contribution = compute_contribution(loss, self.dataset_score, self.trainer_config['gamma'])
            metric = {f"{self.node_id}": float(contribution)}
        else:
            metric = {}
        # send client contriubtion to the server the key is the client id
        return float(loss), len(self.testloader), metric




def client_fn(context: Context) -> Client:
        
    # usally a random number instantiated by the server
    node_id = context.node_id

    # number from 0 up to num clients, corresponds to dataset partitions
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    
    #train_loader, test_loader, num_classes = load_data_mixed(context.run_config, partition_id, num_partitions)

    # load data with FederatedDataset
    train_loader, test_loader, num_classes = load_data(context.run_config, partition_id, num_partitions)

    # score = compute_partition_score(
    #     num_partitions=num_partitions, 
    #     num_classes=num_classes, 
    #     distribution=context.run_config["distribution"], 
    #     save_path=context.run_config["save_path"]
    # )[int(partition_id)]

    
    partition_counts = compute_partition_counts(
        data_loader=train_loader,
        partition_id=partition_id,
        num_classes=num_classes
    )


    return FlowerClient(
        node_id=str(node_id),
        trainloader=train_loader,
        testloader=test_loader,
        client_dataset_score=1, #score ,
        partition_label_counts = partition_counts, # TODO test it for zk-proof
        num_classes=num_classes,
        trainer_config=context.run_config
    ).to_client()

    


# Load configuration
config = get_project_config(".")["tool"]["flwr"]["app"]["config"]

secaggplus = config.get("secaggplus", True)

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod] if secaggplus else None,
)