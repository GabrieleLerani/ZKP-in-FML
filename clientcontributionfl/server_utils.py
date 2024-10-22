
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from flwr.common import Metrics, Scalar
from flwr.common.logger import log
from flwr.server.strategy import FedAvg, Strategy
from .model import Net, test
from .strategy import ContFedAvg

from logging import INFO

def get_evaluate_metrics_aggregation(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # This function will be executed by the strategy in its
        # `configure_evaluate()` method. The returned config will be sent to the clients
        # for evaluation and they can access it via `config` parameter in their 
        # `evaluate` method.
        # for _, metric in metrics:
        #     log(INFO, f"METRICS: {metric}")

        return {}    
    return evaluate_metrics_aggregation_fn

def get_on_fit_config(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method. The returned config will be sent to the clients
        # for training and they can access it via `config` parameter in their 
        # `fit` method.
        """Construct `config` that clients receive when running `fit()`"""
        # TODO decay learning rate after 10 rounds
        # lr = 0.1
        # # Enable a simple form of learning rate decay
        # if server_round > 10:
        #     lr /= 2
        
        return {
            "server_round": server_round,
            #"lr": lr
        }

    return fit_config_fn


def get_on_evaluate_config(cfg: Dict[str, any]):
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
        device: str,
        num_classes: int, 
        testloader: DataLoader, 
        
    ):
    """Define function for global evaluation on the server. Test loader is the full MNIST test set.
    """

    def evaluate_fn(server_round: int, parameters, config) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # this function takes these parameters and evaluates the global model
        # on the server on a pre defined test dataset.
        
        
        # evaluate global model only at the last round
        if server_round % 2 == 0: #== total_rounds:
            
            model = Net(num_classes)

            
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

            loss, accuracy = test(model, testloader, device, accuracy_metric)

            return loss, {"accuracy": accuracy}

    return evaluate_fn



def get_strategy(
        cfg: Dict[str, any],
        num_classes: int,
        testloader: DataLoader
    ) -> Strategy:

    assert cfg['strategy'] in ['FedAvg', 'ContFedAvg'], "Strategy must be either FedAvg or ContFedAvg"

    strategy_class = FedAvg if cfg['strategy'] == 'FedAvg' else ContFedAvg


    common_args = {
        'fraction_fit': cfg['fraction_fit'],
        'fraction_evaluate': cfg['fraction_evaluate'],
        'min_fit_clients': cfg['num_clients_per_round_fit'],
        'min_evaluate_clients': cfg['num_clients_per_round_eval'],
        'evaluate_fn': get_evaluate_fn(cfg['device'], num_classes, testloader),
        'on_fit_config_fn': get_on_fit_config(cfg),
        'on_evaluate_config_fn': get_on_evaluate_config(cfg),
        'evaluate_metrics_aggregation_fn': get_evaluate_metrics_aggregation(cfg)
    }

    # Add ContFedAvg-specific parameter if the strategy is ContFedAvg
    if cfg['strategy'] == 'ContFedAvg':
        common_args['top_k'] = cfg.get('top_k', 2)


    return strategy_class(**common_args)






