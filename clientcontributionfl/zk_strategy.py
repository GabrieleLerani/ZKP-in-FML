
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
from .zokrates_proof import *


class ZkAvg(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verification_keys = {}
        self.scores = {}
        self.zk = Zokrates()

    def _normalize_scores(self, scores):
        """
        Normalize the scores so they sum up to 1.
        Returns a dictionary with the same keys but normalized values.
        """
        # Get all score values
        scores_values = list(scores.values())
        
        # Handle edge cases
        if not scores_values:
            return {}
        
        total = sum(scores_values)
        if total == 0:
            # If sum is 0, return equal weights
            equal_weight = 1.0 / len(scores_values)
            return {key: equal_weight for key in scores.keys()}
        
        # Normalize scores
        normalized_scores = {
            key: value / total 
            for key, value in scores.items()
        }
        
        self.scores = normalized_scores
        

    def _aggregate_verificaiton_keys(self, fit_metrics: List[Tuple[int, Dict[str, Scalar]]]):
        """Use verification key to verify the proof"""
        scores = {}
        for _, m in fit_metrics:
            for key, value in m.items():
                if "vrfkey" in key:
                    self.verification_keys[key] = value # store the verification key
                elif "score" in key:
                    scores[key] = value # store the claimed client score  
                
            
                # res = self.zk.verify_proof(value[0])         
                # log(INFO, f"Proof verification:  {res}") # TODO decide how to check the proof
        
        self._normalize_scores(scores)
        
        log(INFO, self.verification_keys)
        log(INFO, self.scores)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        if server_round == 1:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            self._aggregate_verificaiton_keys(fit_metrics)
            

        return parameters_aggregated, {}

    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training."""
    #     # Use the superclass method to get the initial client selection
    #     client_fit_ins = super().configure_fit(server_round, parameters, client_manager)


    

    # def configure_evaluate(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, EvaluateIns]]:
    #     """Configure the next round of evaluation."""
        


    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     """Aggregate evaluation losses using weighted average."""



def aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
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
