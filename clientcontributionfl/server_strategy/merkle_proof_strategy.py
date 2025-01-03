from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


from flwr.server.client_proxy import ClientProxy

from typing import Union, Optional, Dict


from collections import defaultdict
from clientcontributionfl.utils import aggregate, ClientData
from clientcontributionfl.utils.file_utils import write_bytes_to_file
from clientcontributionfl import Zokrates, ZkSNARK
import os


class MerkleProofAvg(FedAvg):
    """
        Aggregates merkle proofs to ensure the integrity and authenticity of the data received from clients.
        By verifying the merkle proofs, the server can ensure that the data received from clients is genuine 
        and has not been tampered with during transmission.
        This validation step adds an additional layer of security to the federated learning process.
    """

    
    def __init__(self, zk_prover: ZkSNARK, verify_with_smart_contract : bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zk_prover = zk_prover
        self.verify_with_smart_contract = verify_with_smart_contract
        self.client_data: Dict[str, ClientData] = defaultdict(ClientData)

    def _validate_merkle_proofs(self, results: list[tuple[ClientProxy, FitRes]]):
        for c, res in results:
            metrics = res.metrics
            proof_bytes = metrics["proof"]
            verification_key_bytes = metrics["verification_key"]
            
            client_proof_info_path = os.path.join("proofs", f"server_{c.cid}")
            
            os.makedirs(client_proof_info_path, exist_ok=True)

            proof_path = os.path.join(client_proof_info_path, "proof.json")
            verification_key_path = os.path.join(client_proof_info_path, "verification.key")

            write_bytes_to_file(verification_key_path, verification_key_bytes)
            write_bytes_to_file(proof_path, proof_bytes)
            
            
            if self.verify_with_smart_contract:
                client_info = self.client_data[c.cid]
                client_info.client_files_path = client_proof_info_path
                client_info.contract_address = metrics["contract_address"]
                client_info.abi = metrics["abi"]
                response = self.zk_prover.verify_proof_with_smart_contract(client_info)
                client_info.proof_valid = response
 
            else:
                res = self.zk_prover.verify_proof(client_proof_info_path)
                self.client_data[c.cid].proof_valid = "PASSED" in res

        
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

        # validate received proofs
        if server_round == 1:
            self._validate_merkle_proofs(results)
            self.fraction_fit = 0.3 

        return parameters_aggregated, {}

    
        