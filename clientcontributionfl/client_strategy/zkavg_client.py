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
from clientcontributionfl.utils import compute_score, forge_score_in_proof
from clientcontributionfl import Zokrates


class ZkClient(NumPyClient):
    """Define a Flower Client that utilizes Zero-Knowledge Proofs (ZKP) for secure federated learning.

    This client is designed to participate in a setup where it can train a model
    locally using its own data and then share the model parameters with a central server. The client
    also incorporates a zero-knowledge proof mechanism to ensure the integrity and privacy of the
    contributions made by each client without revealing its actual dataset.

    Attributes:
        node_id (str): Unique identifier for the client node.
        partition_id (int): Identifier for the data partition assigned to this client.
        trainloader (DataLoader): DataLoader for the training data.
        testloader (DataLoader): DataLoader for the testing data.
        partition_label_counts (list): List of label counts in the client's data partition.
        num_classes (int): Number of classes in the classification task.
        config (Dict[str, Scalar]): Configuration dictionary for the trainer.
        model (Net): The neural network model used for training.
        criterion (torch.nn.CrossEntropyLoss): Loss function for training.
        accuracy_metric (Accuracy): Metric to evaluate the model's accuracy.
        path_proof_dir (str): Directory path for storing ZKP proofs.
        zk (Zokrates): Instance of Zokrates for handling ZKP operations.
    """

    def __init__(
        self, 
        node_id: str,
        partition_id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        partition_label_counts: list,
        num_classes: int,
        dishonest: bool,
        config: Dict[str, Scalar]
    ) -> None:
        """
        Initialize the ZkClient with necessary configurations and data loaders.

        Args:
            node_id (str): Unique identifier for the client node.
            partition_id (int): Identifier for the data partition assigned to this client.
            trainloader (DataLoader): DataLoader for the training data.
            testloader (DataLoader): DataLoader for the testing data.
            partition_label_counts (list): List of label counts in the client's data partition.
            num_classes (int): Number of classes in the classification task.
            dishonest (bool): Indicates if the client is dishonest and may send fake scores.
            config (Dict[str, Scalar]): Configuration dictionary for the trainer.
        """
        super().__init__()

        self.node_id = node_id
        self.partition_id = partition_id
        self.partition_label_counts = partition_label_counts
    
        self.dishonest = dishonest
        self.dishonest_value = config["dishonest_value"]

        # Zero-Knowledge Proof parameters
        self.scale = config["scale"]
        self.beta = config["beta"]
        self.thr = config["thr"]

        # Data loaders for training and testing
        self.trainloader = trainloader
        self.testloader = testloader

        # Initialize the model and training configurations
        self.model = Net(num_classes)
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.config['device'])
        
        # Set up the path for storing ZKP proofs
        self.path_proof_dir = os.path.join("proofs", f"client_{self.node_id}")
        self.zk = Zokrates(self.path_proof_dir)

    def compute_zkp_contribution(self, score):
        """Compute and generate a zero-knowledge proof for the client's contribution.

        This method calculates the mean of the label distribution and uses Zokrates to
        set up and generate a proof based on the client's data and ZKP parameters.
        """
        counts = self.partition_label_counts
        mean_val = int(sum(counts) / len(counts))
        
        # Setup and generate the zero-knowledge proof
        self.zk.setup()
        self.zk.generate_proof(
            counts=counts, 
            scale=self.scale, 
            beta=self.beta, 
            mean_val=mean_val, 
            thr=self.thr, 
            score=score
        )

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model.

        Args:
            parameters: The model parameters received from the server.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays.

        Args:
            config (Dict[str, Scalar]): Configuration dictionary.

        Returns:
            List of model parameters as numpy arrays.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model using the client's data and return updated parameters.

        Args:
            parameters: The model parameters received from the server.
            config: Configuration dictionary containing training settings.

        Returns:
            Updated model parameters, the size of the training data, and additional parameters.
        """
        self.set_parameters(parameters)

        params = {}
        if config["server_round"] == 1:
            # Compute ZKP score and generate proof in the first round
            score = compute_score(
                counts=self.partition_label_counts, 
                scale=self.scale, 
                beta=self.beta, 
                thr=self.thr
            )
            self.compute_zkp_contribution(score)
            params[f"vrfkey_{self.node_id}"] = self.path_proof_dir
            
            # add additional value
            if self.dishonest: 
                forge_score_in_proof(os.path.join(self.path_proof_dir, "proof.json"), self.dishonest_value)

            

        else:
            # Train the model in subsequent rounds
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

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on the client's test data.

        Args:
            parameters (NDArrays): The model parameters to evaluate.
            config (Dict[str, Scalar]): Configuration dictionary.

        Returns:
            The loss, size of the test data, and an empty dictionary for additional metrics.
        """
        self.set_parameters(parameters)

        loss, accuracy = test(
            self.model,
            self.testloader,
            self.config['device'],
            self.accuracy_metric
        )

        log(INFO, f"Round: {config['server_round']}, Client {self.node_id[:3]} is doing evaluate() with loss: {loss:.4f} and accuracy: {accuracy:.4f}")

        return float(loss), len(self.testloader), {} # metric
