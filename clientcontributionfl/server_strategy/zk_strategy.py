
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
from clientcontributionfl import Zokrates
from clientcontributionfl.utils import extract_score_from_proof, aggregate
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

    def __init__(self, selection_thr: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_data = defaultdict(lambda: ["", 1, False])
        self.zk: Zokrates = Zokrates()
        self.discarding_threshold = selection_thr  

    def _aggregate_score(self, results: list[tuple[ClientProxy, FitRes]]):
        fit_metrics = [(c.cid, res.metrics) for c, res in results]
        self._aggregate_verification_keys_and_scores(fit_metrics)
        self._check_proof()
        self._normalize_scores()
        PrettyPrinter(indent=4).pprint(self.client_data)

    def _normalize_scores(self):
        """
        Normalize the scores so they sum up to 1.
        Returns a dictionary with the same keys but normalized values.
        """
        total = sum(self.client_data[k][1] for k in self.client_data)
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
        

    def _aggregate_verification_keys_and_scores(self, fit_metrics: List[Tuple[str, Dict[str, Scalar]]]):
        """Aggregates verification keys and scores."""
        
        for client_id, metrics in fit_metrics:
            for _, verification_key_path in metrics.items():
                #client_id = key.split('_')[1]
                self.client_data[client_id][0] = verification_key_path # path of the verification key
                self.client_data[client_id][1] = extract_score_from_proof(os.path.join(verification_key_path, "proof.json"))
        
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
            #score = self.client_data[c.cid][1]
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
            self._aggregate_score(results)

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
        # Do not configure federated evaluation if fraction eval is 0 or if its first round
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

        # filter clients based on proof and score
        clients = self._filter_clients(clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
        

