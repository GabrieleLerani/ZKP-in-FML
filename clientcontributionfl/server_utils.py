
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from flwr.common import Metrics, Scalar
from flwr.common.logger import log
from flwr.server.strategy import FedAvg, Strategy

import clientcontributionfl.models as models
from clientcontributionfl.server_strategy import ZkAvg, ContributionAvg, PoCZk, PoC
from clientcontributionfl.utils import get_model_class

from logging import INFO

def get_evaluate_metrics_aggregation(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        return {}    
    return evaluate_metrics_aggregation_fn


def get_on_fit_config(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        
        """Construct `config` that clients receive when running `fit()`"""
        
        # learning rate decay of 0.995 per round
        initial_lr = cfg.get("lr", 0.1)
        lr = initial_lr * (0.995 ** server_round) if server_round > 1 else initial_lr
        
        return {
            "server_round": server_round,
            "lr": lr
        }
    
    def fit_config_fn_poc(server_round: int, state: str):
        
        """Construct `config` that clients receive when running `fit()`"""
        
        # learning rate decay of 0.995 per round
        initial_lr = cfg.get("lr", 0.1)
        lr = initial_lr * (0.995 ** server_round) if server_round > 1 else initial_lr
        
        return {
            "server_round": server_round,
            "lr": lr,
            "state": state
        }
    
    if "PoC" in cfg.get("strategy"):
        return fit_config_fn_poc

    return fit_config_fn




def get_on_evaluate_config(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    def evaluate_config_fn(server_round: int):
        return {
            "server_round": server_round,
        }

    return evaluate_config_fn

def get_evaluate_fn(
        device: str,
        num_classes: int, 
        testloader: DataLoader,
        dataset_name: str 
        
    ):
    """Define function for global evaluation on the server. Test loader is the full MNIST test set.
    """

    def evaluate_fn(server_round: int, parameters, config) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        # evaluate global model every two round
        if server_round % 2 == 0:
            
            model_class = get_model_class(models,dataset_name)
            
            model = model_class(num_classes)

            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

            loss, accuracy = models.test(model, testloader, device, accuracy_metric)

            return loss, {"accuracy": accuracy}

    return evaluate_fn


def get_strategy(
        cfg: Dict[str, any],
        num_classes: int,
        testloader: DataLoader
    ) -> Strategy:

    common_args = {
        'fraction_fit': cfg['fraction_fit'],
        'fraction_evaluate': cfg['fraction_evaluate'],
        'min_fit_clients': cfg['num_clients_per_round_fit'],
        'min_evaluate_clients': cfg['num_clients_per_round_eval'],
        'evaluate_fn': get_evaluate_fn(cfg['device'], num_classes, testloader, cfg["dataset_name"]),
        'on_fit_config_fn': get_on_fit_config(cfg),
        'on_evaluate_config_fn': get_on_evaluate_config(cfg),
        'evaluate_metrics_aggregation_fn': get_evaluate_metrics_aggregation(cfg),
    }

    strategy_class = None

    if cfg['strategy'] == 'FedAvg':
        strategy_class = FedAvg
    elif cfg['strategy'] in ['ContAvg', 'ZkAvg']:
        common_args["fraction_fit"] = 1.0 
        common_args['selection_thr'] = cfg['selection_thr']
        strategy_class = ContributionAvg if cfg['strategy'] == 'ContAvg' else ZkAvg
    elif cfg['strategy'] in ['PoC', 'PoCZk']:
        common_args['d'] = cfg['d']
        strategy_class = PoC if cfg['strategy'] == 'PoC' else PoCZk

    return strategy_class(**common_args)






