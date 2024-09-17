

from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerAppComponents
from typing import Dict
import torch
from collections import OrderedDict
from model import Net
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict
from flwr.common import Scalar

def get_evaluate_fn(
        num_classes: int, 
        total_rounds: int, 
        testloader: DataLoader, 
        trainer_config: Dict[str, any]
    ):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        
        # evaluate global model only at the last round
        if server_round == total_rounds - 1:
            
            
            model = Net(num_classes, trainer_config)

            trainer = pl.Trainer(max_epochs=trainer_config['num_epochs'], enable_progress_bar=False)

            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            metrics = trainer.test(model, testloader)

            loss = metrics[0]["test_loss"]
            accuracy = metrics[0]["test_accuracy"]
            # Report the loss and any other metric (inside a dictionary). In this case
            # we report the global test accuracy.
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
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            min_available_clients=cfg.num_clients,
            evaluate_fn=get_evaluate_fn(num_classes, cfg.num_rounds, testloader, trainer_config)
        )
        # Configure the server for 3 rounds of training
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn