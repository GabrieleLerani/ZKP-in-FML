from clientcontributionfl.server_strategy import ZkAvg
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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from clientcontributionfl.utils import aggregate
from pprint import PrettyPrinter

from typing import List, Tuple, Union, Optional, Dict
import random

class PowerOfChoice(ZkAvg):
    """
    PowerOfChoice strategy extends ZkAvg to bias client selection towards clients with the lowest
    local loss. Client selection probability is based on their contribution score rather than on their
    fraction of data. The first round is used to collect dataset score and verify them using Zero-Knowledge
    then in subsequents round the set of active client S is chosen with the following steps:

    1. Biased sampling of a candidate set A based on scores.
    2. Requesting local loss estimates from A.
    3. Selecting clients with the highest local losses from A to participate.
    """

    def __init__(self, d: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d = d # size of candidate set m <= d <= K

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
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample size and minimum number of clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        
        if server_round == 1:
            # First round: sample uniformly without filtering
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            return [(client, fit_ins) for client in clients]

        # Biased sampling based on scores
        candidate_set = self._biased_sampling(client_manager, self.d)
        
        
        # Collect local losses from candidate set
        # client_losses = self._get_client_losses(candidate_set, parameters)
        
        # # Select clients with the highest local losses
        # active_clients = self._select_high_loss_clients(candidate_set, client_losses, sample_size)
        
        
        # TODO send to clients in 
        return [(client, fit_ins) for client in candidate_set]
    
    def _biased_sampling(self, client_manager: ClientManager, d: int) -> List[ClientProxy]:
        """
        Sample d clients from the client manager using probabilities proportional to their scores.
        """
        all_clients = list(client_manager.all())
        probabilities = [self.client_data[client.cid][1] for client in all_clients]
        
        
        # TODO find a way Sample clients without replacement based on their probabilities
        sampled_clients = random.choices(all_clients, weights=probabilities, k=d)
        return sampled_clients

    def _get_client_losses(self, candidate_set: List[ClientProxy], parameters: Parameters) -> Dict[str, float]:
        """
        Request clients to compute their local losses and return the results as a dictionary.
        """
        client_losses = {}
        for client in candidate_set:
            # Simulate loss computation (replace with real interaction if available)
            loss = self._simulate_client_loss(client, parameters)
            client_losses[client.cid] = loss
        return client_losses
    
    def _simulate_client_loss(self, client: ClientProxy, parameters: Parameters) -> float:
        """
        Simulate a client's local loss computation. Replace this with actual client-server interaction.
        """
        # TODO: Replace this with a real implementation where the client computes the loss
        # For now, generate a random loss for testing purposes
        return random.uniform(0, 1)

    def _select_high_loss_clients(
        self, candidate_set: List[ClientProxy], client_losses: Dict[str, float], m: int
    ) -> List[ClientProxy]:
        """
        Select m clients with the highest local losses from the candidate set.
        """
        # Sort candidate set by loss values in descending order
        sorted_candidates = sorted(candidate_set, key=lambda c: client_losses[c.cid], reverse=True)
        return sorted_candidates[:m]