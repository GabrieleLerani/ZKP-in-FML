
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from flwr.common import Metrics, Scalar
from flwr.server.strategy import FedAvg, Strategy, FedAvgM, FedAdam

from clientcontributionfl.models import get_model_initial_parameters
import clientcontributionfl.models as models
from clientcontributionfl.server_strategy import ZkAvg, ContributionAvg, PoCZk, PoC, MerkleProofAvg, CLAvg
from clientcontributionfl.utils import get_model_class, generate_zok_client_score_template, write_zok_file
from clientcontributionfl import Zokrates

from .dataset import get_num_classes


def get_evaluate_metrics_aggregation(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        return {}    
    return evaluate_metrics_aggregation_fn


def get_on_fit_config(cfg: Dict[str, any]):
    """Return function that prepares config to send to clients."""

    initial_lr = cfg.get("lr", 0.01)
    decay_per_round = cfg.get("decay_per_round", 0.995)
        
    def fit_config_fn(server_round: int):
        
        lr = initial_lr * (decay_per_round ** server_round) if server_round > 1 else initial_lr
        
        return {
            "server_round": server_round,
            "lr": lr
        }
    
    def fit_config_fn_poc(server_round: int, state: str):
        
    
        lr = initial_lr * (decay_per_round ** server_round) if server_round > 1 else initial_lr
        
        return {
            "server_round": server_round,
            "lr": lr,
            "state": state
        }
    
    def fit_config_fn_fedavgm(server_round: int):
        
        return {
            "server_round": server_round,
            "lr": initial_lr
        }
    
    if "PoC" in cfg.get("strategy"):
        return fit_config_fn_poc
    elif "FedAvgm" in cfg.get("strategy"):
        return fit_config_fn_fedavgm

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
    
    elif cfg['strategy'] == 'FedAvgM':
        model_class = get_model_class(models,cfg["dataset_name"])
        model = model_class(num_classes)
        
        common_args["server_momentum"] = cfg["server_momentum"]
        common_args["server_learning_rate"] = cfg["server_learning_rate"]
        common_args["initial_parameters"] = get_model_initial_parameters(model)
        strategy_class = FedAvgM

    elif cfg['strategy'] == 'FedAdam':
        model_class = get_model_class(models,cfg["dataset_name"])
        model = model_class(num_classes)

        common_args["eta"] = cfg["eta"]
        common_args["eta_l"] = cfg["eta_l"]
        common_args["beta_1"] = cfg["beta_1"]
        common_args["beta_2"] = cfg["beta_2"]
        common_args["tau"] = cfg["tau"]

        common_args["initial_parameters"] = get_model_initial_parameters(model)
        strategy_class = FedAdam

    
    elif cfg['strategy'] == 'MPAvg':
        common_args["fraction_fit"] = 1.0
        common_args['verify_with_smart_contract'] = cfg['smart_contract']
        common_args['zk_prover'] = Zokrates()
        strategy_class = MerkleProofAvg

    elif cfg['strategy'] == 'ContAvg':
        
        common_args["fraction_fit"] = 1.0 
        common_args['selection_thr'] = cfg['selection_thr']
        strategy_class = ContributionAvg 

    elif cfg['strategy'] == 'CLAvg':
        common_args["fraction_fit"] = 0.8 
        common_args['selection_thr'] = cfg['selection_thr']
        strategy_class = CLAvg

    elif cfg['strategy'] == 'ZkAvg':
        common_args['selection_thr'] = cfg['selection_thr']
        common_args['verify_with_smart_contract'] = cfg['smart_contract']
        common_args['zk_prover'] = Zokrates()
        strategy_class = ZkAvg

    elif cfg['strategy'] in ['PoC', 'PoCZk']:
        common_args['d'] = cfg['d']

        if cfg['strategy'] == 'PoCZk':
            common_args['verify_with_smart_contract'] = cfg['smart_contract']
            common_args['zk_prover'] = Zokrates()
        strategy_class = PoC if cfg['strategy'] == 'PoC' else PoCZk
    
    # create a custom zok file based on number of classes of the dataset
    # it is required for zokrates compilation
    if cfg['strategy'] in ['ZkAvg', 'PoCZk']:
        create_contribution_zokrates_file(cfg)

    return strategy_class(**common_args)

def create_contribution_zokrates_file(cfg):

    file_path = cfg["zok_contribution_file_path"]
    dataset_name = cfg["dataset_name"]
    num_classes = get_num_classes(dataset_name)
    template = generate_zok_client_score_template(num_classes)
    write_zok_file(
        directory=file_path, 
        filename="contribution.zok",
        template=template
    )





