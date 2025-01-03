import os
import json
import subprocess
from typing import Tuple
from .smart_contract_manager import SmartContractManager
from .zero_knowledge import ZkSNARK, SmartContractVerifier
from clientcontributionfl.utils import load_proof_data, ClientData


class Zokrates(ZkSNARK, SmartContractVerifier):
    """A wrapper class for interacting with the ZoKrates zero-knowledge proof system.

    This class provides an interface to compile ZoKrates programs, generate proofs, and verify them with smart contracts.

    Example of usage:
    .. code-block:: python
        # Initialize Zokrates with a working directory
        zokrates = Zokrates()

        # Compile the ZoKrates program and set up the keys
        zokrates.setup("path/to/your/zok_file.zok")

        # Generate a proof with some arguments
        proof = zokrates.generate_proof(("arg1", "arg2", "arg3"))
        print("Generated Proof:", proof)

        # Verify the generated proof
        verification_result = zokrates.verify_proof("path/to/your/verification/key")
        print("Verification Result:", verification_result)
    """

    def __init__(self, working_dir : str = None):
        """Initialize Zokrates with a working directory.
        
        Args:
            working_dir (str, optional): Directory to store ZoKrates files and proofs.
                If provided, the directory will be created if it doesn't exist.
        """
        
        self.working_dir = working_dir
        # Create working directory if it doesn't exist
        if self.working_dir != None:
            os.makedirs(self.working_dir, exist_ok=True)


    def _run_command(self, command, path = None):
        """Run a ZoKrates command using subprocess.
        """
        cmd_parts = command.split(' ', 1)
        full_command = f"~/.zokrates/bin/zokrates {cmd_parts[1]}" if len(cmd_parts) > 1 else "/usr/local/bin/zokrates"
        
        if path != None:
            os.makedirs(path, exist_ok=True)

        # Run command in the specified directory
        result = subprocess.run(
            full_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            shell=True,
            cwd=path if path != None else self.working_dir # Set the working directory
        )
        if result.returncode != 0:
            raise ValueError(f"Error: {result.stdout.decode()}")
        return result.stdout.decode()

    def setup(self, zok_file_path : str):
        self._run_command(f"zokrates compile -i {zok_file_path}")
        self._run_command("zokrates setup")


    def generate_proof(self, arguments: Tuple[str]):
        
        arguments = " ".join(map(str, arguments))

        self._run_command(f"zokrates compute-witness -a {arguments}")
        proof = self._run_command("zokrates generate-proof")
        return proof

    def verify_proof(self, key_path: str):
        return self._run_command("zokrates verify", path=key_path)

    def generate_smart_contract(self, node_id):

        solidity_manager = SmartContractManager()

        self._run_command("zokrates export-verifier")
        solidity_file_path = os.path.join(self.working_dir, "verifier.sol")
        
        abi, bytecode = solidity_manager.compile_smart_contract(solidity_file_path)
        
        contract_address = solidity_manager.deploy_smart_contract(abi, bytecode, node_id)
        return contract_address, abi

    def verify_proof_with_smart_contract(self, client_data : ClientData):
        client_files_path = client_data.client_files_path
        contract_address = client_data.contract_address
        abi = json.loads(client_data.abi)
        proof_path = os.path.join(client_files_path, "proof.json")
        proof_points, inputs = load_proof_data(proof_path)

        solidity_manager = SmartContractManager()
        response = solidity_manager.call_contract_function(
            abi,
            contract_address,
            'verifyTx',
            0, # if server use account 0
            proof_points,
            inputs
        )
        return response
        