
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple, Union, Optional, Dict
from logging import INFO, DEBUG, WARNING
from flwr.server.strategy.aggregate import weighted_loss_avg
from functools import reduce
import numpy as np
from pprint import PrettyPrinter
import random


class ContFedAvg(FedAvg):
    """Federated Averaging strategy that selects clients based on their contribution scores.

    This strategy extends the FedAvg strategy by sampling the top-k clients
    with the highest contribution scores obtained during evaluation. The
    contribution scores are used to prioritize clients that provide more
    valuable updates to the global model.
    
    
    Parameters:
    -----------
    top_k : int, optional
        The number of top-performing clients to select for each round of training.
        Defaults to 1.
    **kwargs : dict
        Additional keyword arguments to be passed to the FedAvg constructor.
    """
    
    # TODO pass top_k from configuration file
    def __init__(self, top_k: int = 2,**kwargs):
        super().__init__( **kwargs)
        self.top_k = top_k 
        self.contribution_metrics = {}

    

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Use the superclass method to get the initial client selection
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)

        # Useful for the first round when the server does not have contribution metrics
        if self.contribution_metrics:
            
            filtered_client_fit_ins = []
            keep_top_k_clients = keep_top_k(self.contribution_metrics, self.top_k)
            # log(INFO, self.contribution_metrics)
            # log(INFO, keep_top_k_clients)
            for cfi in client_fit_ins:
                if cfi[0].cid in keep_top_k_clients:
                    filtered_client_fit_ins.append(cfi)
            
        else:
            # If contribution_metrics is empty, use the original client selection
            filtered_client_fit_ins = client_fit_ins

        
        return filtered_client_fit_ins
    

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # If it's not the first round, randomly sample 30% of clients
        if server_round > 1:
            num_clients_to_sample = max(1, int(0.3 * len(clients)))
            clients = random.sample(clients, num_clients_to_sample)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
        

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )


        if server_round == 1:

            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            # get metrics sent from clients in evaluate() function
            for _, metric in eval_metrics:
                for key, value in metric.items():
                    self.contribution_metrics[key] = value
        
            

        # TODO         
        # if self.evaluate_metrics_aggregation_fn:
        #     eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    

def keep_top_k(d, k):
    k = min(k, len(d)) # if k is greater than the number of clients, keep all clients
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])




def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime