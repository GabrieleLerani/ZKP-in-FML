

from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerAppComponents
from strategy import ContributionFedAvg
from typing import Dict
import torch
from collections import OrderedDict
from model import Net, test
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict
from flwr.common import Scalar
from omegaconf import DictConfig
from torchmetrics import Accuracy
from typing import List
from flwr.common import Metrics
from flwr.common.logger import log
from logging import INFO

def get_evaluate_metrics_aggregation(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # This function will be executed by the strategy in its
        # `configure_evaluate()` method. The returned config will be sent to the clients
        # for evaluation and they can access it via `config` parameter in their 
        # `evaluate` method.
        for _, metric in metrics:
            log(INFO, f"METRICS: {metric}")

        return {}    
    return evaluate_metrics_aggregation_fn

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method. The returned config will be sent to the clients
        # for training and they can access it via `config` parameter in their 
        # `fit` method.

        return {
            "server_round": server_round,
        }

    return fit_config_fn


def get_on_evaluate_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def evaluate_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_evaluate()` method. The returned config will be sent to the clients
        # for evaluation and they can access it via `config` parameter in their 
        # `evaluate` method.

        return {
            "server_round": server_round,
        }

    return evaluate_config_fn

def get_evaluate_fn(
        num_classes: int, 
        total_rounds: int, 
        testloader: DataLoader, 
        trainer_config: Dict[str, any]
    ):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # this function takes these parameters and evaluates the global model
        # on the server on a pre defined test dataset.

        
        # evaluate global model only at the last round
        if server_round == total_rounds:
            
            model = Net(num_classes, trainer_config)

            
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(trainer_config['device'])

            loss, accuracy = test(model, testloader, trainer_config["device"], accuracy_metric)

            return loss, {"accuracy": accuracy}

    return evaluate_fn


def generate_server_fn(
        cfg: Dict[str, any],
        num_classes: int,
        testloader: DataLoader,
        trainer_config: Dict[str, any]
    ) -> ServerAppComponents:

    def server_fn(context: Context) -> ServerAppComponents:
        # Create the FedAvg strategy
        strategy = ContributionFedAvg(
            fraction_fit=cfg.fraction_fit,
            fraction_evaluate=cfg.fraction_evaluate,
            min_fit_clients=cfg.num_clients_per_round_fit,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            evaluate_fn=get_evaluate_fn(num_classes, cfg.num_rounds, testloader, trainer_config),
            on_fit_config_fn=get_on_fit_config(cfg),
            on_evaluate_config_fn=get_on_evaluate_config(cfg),
            evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation(cfg)
        )
        # Configure the server for 3 rounds of training
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn