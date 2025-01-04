
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
from clientcontributionfl import Zokrates, ZkSNARK, SmartContractVerifier
from clientcontributionfl.utils import extract_score_from_proof, aggregate, ClientData
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

    def __init__(
            self, 
            zk_prover: ZkSNARK,
            selection_thr: Optional[float] = None, 
            verify_with_smart_contract : bool = False, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.zk_prover = zk_prover
        self.discarding_threshold = selection_thr
        
        self.verify_with_smart_contract = verify_with_smart_contract
        self.client_data: Dict[str, ClientData] = defaultdict(ClientData)

        

    def _aggregate_score(self, results: list[tuple[ClientProxy, FitRes]]):
        fit_metrics = [(c.cid, res.metrics) for c, res in results]
        
        self._aggregate_client_data(fit_metrics)
        self._check_proof()
        self._normalize_scores()
        for client_id, client_info in self.client_data.items():
            PrettyPrinter(indent=4).pprint({client_id: {"proof_valid": client_info.proof_valid, "contribution_score": client_info.contribution_score}})
        

    def _normalize_scores(self):
        """
        Normalize the scores so they sum up to 1.
        Returns a dictionary with the same keys but normalized values.
        """

        total = sum(info.contribution_score for info in self.client_data.values())
        
        if total == 0:
            equal_weight = 1.0 / len(self.client_data)
            for info in self.client_data.values():
                info.contribution_score = equal_weight
        else:
            for info in self.client_data.values():
                info.contribution_score = info.contribution_score / total

    def _create_set_of_clients_with_valid_proofs(self):
        """Pre process the set of valid clients in order to improve look up time during training."""
        self.valid_clients = {cid for cid, data in self.client_data.items() if data.proof_valid}

    def _is_client_valid(self, cid: str):
        return cid in self.valid_clients

    def _aggregate_client_data(self, fit_metrics: List[Tuple[str, Dict[str, Scalar]]]):
        for client_id, metrics in fit_metrics:
            
            
            client_info = self.client_data[client_id]
            client_info.client_files_path = metrics["client_data_path"]
            client_info.contribution_score = extract_score_from_proof(
                os.path.join(client_info.client_files_path, "proof.json")
            )

            if self.verify_with_smart_contract:
                client_info.contract_address = metrics["contract_address"]
                client_info.abi = metrics["abi"]
        
        log(INFO,"Client data aggregated.")


    def _check_proof(self):
        """
        Check zero-knowledge proof of clients. If verification fails, the corresponding client
        is removed from the client data, ensuring they are not selected for future training rounds.
        """
        for cid in self.client_data:
            if self.verify_with_smart_contract and isinstance(self.zk_prover, SmartContractVerifier):
                
                response = self.zk_prover.verify_proof_with_smart_contract(client_data=self.client_data[cid])
                
                self.client_data[cid].proof_valid = response
                
            else: 
                # extract the verification key path
                vrk_key_path = self.client_data[cid].client_files_path

                # verify the proof is correct
                res = self.zk_prover.verify_proof(vrk_key_path)

                if "PASSED" in res:                
                    self.client_data[cid].proof_valid = True 

    def _filter_clients(self, clients: List[ClientProxy]) -> List[ClientProxy]:
        """Filter clients based on their contribution score."""
        filtered_clients = [c for c in clients if self._is_client_valid(c.cid)]
        return filtered_clients

    
    def num_fit_clients(self, num_available_clients: int, fraction_fit: float) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

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
            self._create_set_of_clients_with_valid_proofs()

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
        
        fraction_fit = 1.0 if server_round == 1 else self.fraction_fit

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available(), fraction_fit
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round > 1:
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
        

