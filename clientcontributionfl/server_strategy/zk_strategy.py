
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
from functools import reduce
import numpy as np
from pprint import PrettyPrinter
from clientcontributionfl import Zokrates
from clientcontributionfl.utils import extract_score_from_proof
from collections import defaultdict
import os

class ZkAvg(FedAvg):
    """
    ZkAvg is a strategy that extends the FedAvg algorithm by incorporating
    zero-knowledge proof verification of client contributions. This class ensures that only
    clients with valid proofs and significant contributions are selected during the training
    process. Clients with poor contribution scores are filtered out, enhancing the robustness
    and reliability of the federated learning process.

    Attributes:
        client_data (defaultdict): A dictionary storing verification key paths, contribution scores, proof validity.
        zk (Zokrates): An instance of the Zokrates class used for zero-knowledge proof operations.
        discarding_threshold (float): A threshold below which clients are considered to have poor contributions and are filtered out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_data = defaultdict(lambda: ["", 1, False])
        self.zk = Zokrates()
        self.discarding_threshold: float = 0.1  # TODO automatically compute this threshold

    def _normalize_scores(self):
        """
        Normalize the scores so they sum up to 1.
        Returns a dictionary with the same keys but normalized values.
        """
        # TODO check if it is ok the following
        # compute the sum of all the scores for each client.
        # take score only if client has valid proof
        total = sum(self.client_data[k][1] for k in self.client_data if self.client_data[k][2])
        # edge cases
        if total == 0:
            equal_weight = 1.0 / len(self.client_data)
            self.client_data = {
                key: [value[0], equal_weight]
                for key, value in self.client_data.items()
            }
        else:
            # normalize each score w.r.t to the total
            self.client_data = {
                key: [value[0], value[1] / total, value[2]] 
                for key, value in self.client_data.items()
            }
        

    def _aggregate_verificaiton_keys_and_scores(self, fit_metrics: List[Tuple[int, Dict[str, Scalar]]]):
        """Aggregates verification keys and scores."""
        
        for _, metrics in fit_metrics:
            for key, value in metrics.items():
                client_id = key.split('_')[1]
                self.client_data[client_id][0] = value # path of the verification key
                self.client_data[client_id][1] = extract_score_from_proof(os.path.join(value, "proof.json"))
        
        log(INFO,"Verification keys and score aggregated")

    def _check_proof(self):
        """
        Check zero-knowledge proof of clients. If verification fails, the corresponding client
        is removed from the client data, ensuring they are not selected for future training rounds.
        """
        for k in self.client_data:
            # extract the verification key path
            vrk_key_path = self.client_data[k][0]

            # verify the proof is correct
            res = self.zk.verify_proof(vrk_key_path)

            if "PASSED" in res:                
                self.client_data[k][2] = True 

    def _filter_clients(self, clients: List[ClientProxy]) -> List[ClientProxy]:
        """Filter clients based on their contribution score."""
        # TODO replace this computation with a more complex one
        # where similar clients with a similar score are maintained
        # and other not
        
        filtered_clients = []
        for c in clients:
            score = self.client_data[c.cid][1]
            proof_is_valid = self.client_data[c.cid][2]
            # TODO just not select invalid clients, then use also score
            if proof_is_valid:
                filtered_clients.append(c)
                
        return filtered_clients

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

        # aggregate keys, score and normalize them
        if server_round == 1:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            self._aggregate_verificaiton_keys_and_scores(fit_metrics)
            self._check_proof()
            self._normalize_scores()
            PrettyPrinter(indent=4).pprint(self.client_data)

        return parameters_aggregated, {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # TODO fit_ins is only one but you can create a different one
        # for each client, could be useful for client clustering.
        fit_ins = FitIns(parameters, config)
        
        # TODO Sample clients, sample all if first round in order to collect verification keys
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # do not filter when is the first round
        if server_round != 1:
            clients = self._filter_clients(clients)
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0 or if its first round
        if self.fraction_evaluate == 0.0 or server_round == 1:
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

        # filter clients based on proof and score
        clients = self._filter_clients(clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
        


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
