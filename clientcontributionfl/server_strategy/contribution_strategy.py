
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple, Union, Optional, Dict
from logging import INFO
from pprint import PrettyPrinter

from collections import defaultdict
from clientcontributionfl.utils import aggregate


class ContributionAvg(FedAvg):
    """
    
    ContributionAvg is a strategy that extends the FedAvg algorithm by incorporating
    contribution aggregation of score sent by clients. Scores are computed locally
    by clients and nodes with poor contribution are filtered out and no considered for
    the next round of training.


    Attributes:
        client_data (defaultdict): A dictionary storing contribution scores for each client.
        discarding_threshold (float): A threshold below which clients are considered to have poor contributions and are filtered out.
    """

    def __init__(self, selection_thr: float,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_data = defaultdict(lambda: 1)
        self.discarding_threshold: float = selection_thr 

    def _normalize_scores(self):
        """
        Normalize the scores so they sum up to 1.
        Returns a dictionary with the same keys but normalized values.
        """

        # compute the sum of all the scores for each client
        total = sum(self.client_data[k] for k in self.client_data)
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
                key: value / total
                for key, value in self.client_data.items()
            }

    def _aggregate_scores(self, fit_metrics: List[Tuple[int, Dict[str, Scalar]]]):
        """Aggregates and scores."""
        
        for _, metrics in fit_metrics:
            for key, value in metrics.items():    
                client_id = key.split('_')[1]
                
                self.client_data[client_id] = value # integer score of client on its partition
        
        log(INFO, "Score aggregated")
   
    def _filter_clients(self, clients: List[ClientProxy]) -> List[ClientProxy]:
        """Filter clients based on their contribution score."""
        
        filtered_clients = [c for c in clients if self.client_data[c.cid] >= self.discarding_threshold]
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
            self._aggregate_scores(fit_metrics)
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
        
        fit_ins = FitIns(parameters, config)
        
        
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
        # Do not configure federated evaluation if fraction eval is 0 or first round
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

        # filter clients based on score
        clients = self._filter_clients(clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
        