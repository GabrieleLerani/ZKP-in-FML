
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
from clientcontributionfl.utils import aggregate, aggregate_between_clusters
import numpy as np
from sklearn.cluster import AgglomerativeClustering


class CLAvg(FedAvg):
    
    def __init__(self, selection_thr: float,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_data = defaultdict(lambda: 1)
        self.clusters : dict[str, int] = {}
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
   
    def _cluster_clients(self) -> Dict[str, int]:
        """Filter clients based on their contribution score and return a dictionary of clusters."""

        # Extract client scores into a numpy array
        scores = np.array(list(self.client_data.values())).reshape(-1, 1)  # Reshape for clustering

        # Create and fit the AgglomerativeClustering model
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.0001,
            metric="l1",  
            linkage="complete"
        )
        clustering.fit(scores)
        
        # Create a dictionary where the key is the client cid and the value is its corresponding cluster
        client_clusters = {}
        for client_id, label in zip(self.client_data.keys(), clustering.labels_):
            client_clusters[client_id] = label
        
        return client_clusters

        
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

        # aggregate scores and create cluster based on their similarity
        if server_round == 1:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            self._aggregate_scores(fit_metrics)
            self._normalize_scores()
            self.clusters = self._cluster_clients()
            PrettyPrinter(indent=4).pprint(self.clusters)
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, )
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        else:

            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, client.cid)
                for client, fit_res in results
            ]
            aggregated_ndarrays = aggregate_between_clusters(weights_results, self.clusters)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        return parameters_aggregated, {}

    def _get_clients_by_cluster(self, clients: list[ClientProxy]) -> dict[int, list[str]]:
        clusters = {}
        for client in clients:
            cluster_id = self.clusters[client.cid]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client)
        return clusters


    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        return max(num_available_clients, self.min_fit_clients), self.min_available_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round > 1:
            # Split clients into sublists based on their cluster
            cluster_clients = self._get_clients_by_cluster(clients)

            # Select a fraction of clients from each cluster and print their cid
            selected_clients = []
            for cluster_id, cluster_list in cluster_clients.items():
                num_clients_to_sample = max(1, int(len(cluster_list) * self.fraction_fit))
                sampled_clients = np.random.choice(cluster_list, num_clients_to_sample, replace=False).tolist()
                selected_clients.extend(sampled_clients)
                log(INFO, f"Selected clients from cluster {cluster_id}: {[c.cid for c in sampled_clients]}")
            clients = selected_clients

        fit_ins = FitIns(parameters, config)

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

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
        