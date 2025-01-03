import solcx
from solcx import compile_source
from web3 import Web3
from flwr.common.logger import log
from logging import INFO

SOLIDITY_VERSION = '0.8.0'


class SmartContractManager:
    """
    This class is responsible for managing the lifecycle of smart contracts on the Ethereum blockchain.
    It provides methods for compiling, deploying, and interacting with smart contracts.
    """
    def __init__(self, node_url: str = "http://127.0.0.1:7545"):
        solcx.install_solc(version=SOLIDITY_VERSION)
        
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        if not self.w3.is_connected():
            raise Exception("Failed to connect to the Ethereum node")
        

    def compile_smart_contract(self, solidity_file_path: str):
        with open(solidity_file_path, 'r') as file:
            solidity_source_code = file.read()

        compiled_sol = compile_source(
            solidity_source_code,
            output_values=['abi', 'bin'],
            solc_version=SOLIDITY_VERSION
        )

        _, contract_interface = compiled_sol.popitem()
        abi = contract_interface['abi']
        bytecode = contract_interface['bin']
        return abi, bytecode

    def deploy_smart_contract(self, abi, bytecode, actor_id) -> str:
        if not abi or not bytecode:
            raise ValueError("Contract must be compiled before deployment")

        Contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        transaction = {
            "from": self.w3.eth.accounts[actor_id + 1] # account 0 is of the server
        }

        tx_hash = Contract.constructor().transact(transaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if tx_receipt.status != 1:
            raise Exception("Contract deployment failed")

        log(INFO, f"Account {actor_id + 1} deployed a contract at: {tx_receipt.contractAddress}")

        return tx_receipt.contractAddress

    def call_contract_function(self, abi : str, contract_address : str, function_name: str, id: int, *args):
        '''
        General function able to call any smart contract function and execute it as a transaction
        '''
        contract = self.w3.eth.contract(
            address=contract_address,
            abi=abi
        )
        
        if not contract:
            raise ValueError("Contract not deployed or loaded")
        
        contract_function = getattr(contract.functions, function_name)
        
        response = contract_function(*args).call()
        
        return response


