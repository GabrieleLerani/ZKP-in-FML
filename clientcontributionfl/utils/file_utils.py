import shutil
import os
import json
from typing import List

def cleanup_proofs():
    """Remove the proofs directory and all its contents if it exists."""
    proofs_dir = os.path.join(os.getcwd(), "proofs")
    if os.path.exists(proofs_dir):
        try:
            shutil.rmtree(proofs_dir)
        except Exception as e:
            print(f"Error while removing directory {proofs_dir}: {str(e)}")



def extract_score_from_proof(proof_file_path: str):
    """
    Extract the contribution score (x) from the proof.json file and convert it from hexadecimal to an integer.
    
    Args:
        proof_file_path (str): Path to the proof.json file.
    
    Returns:
        int: The score (x) as an integer.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the required 'inputs' field is not found in the JSON.
        ValueError: If the inputs field is empty or the value is not a valid hexadecimal.
    """
    try:

        # Open and parse the JSON file
        with open(proof_file_path, "r") as proof_file:
            proof_data = json.load(proof_file)
        
        # Extract the public inputs from the proof
        public_inputs = proof_data.get("inputs", [])
        
        # Validate inputs and convert the first input (score) from hex to int
        if public_inputs and len(public_inputs) > 0:
            score_hex = public_inputs[0]  
            score_int = int(score_hex, 16)  
            return score_int
        else:
            raise ValueError("No public inputs found or inputs field is empty.")
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Proof file not found: {proof_file_path}") from e
    except KeyError as e:
        raise KeyError("'inputs' field is missing in the proof.json file.") from e
    except ValueError as e:
        raise ValueError(f"Invalid value in 'inputs' field: {e}") from e


def forge_score_in_proof(proof_file_path, forged_score):
    """
    Modifies the score (public input) in the proof.json file to a forged value.
    
    Args:
        proof_file_path (str): Path to the proof.json file.
        forged_score (int): The forged score to insert into the inputs.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the 'inputs' field is not found in the JSON.
    """
    try:
        # Open and parse the JSON file
        with open(proof_file_path, "r") as proof_file:
            proof_data = json.load(proof_file)
        
        # Extract the public inputs from the proof
        public_inputs = proof_data.get("inputs", [])
        
        if public_inputs and len(public_inputs) > 0:
            # Replace the first input (score) with the forged value in hexadecimal
            proof_data["inputs"][0] = hex(forged_score)
        else:
            raise ValueError("No public inputs found or inputs field is empty.")
        
        # Write the modified proof data back to the file
        with open(proof_file_path, "w") as proof_file:
            json.dump(proof_data, proof_file, indent=4)
        
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Proof file not found: {proof_file_path}") from e
    except KeyError as e:
        raise KeyError("'inputs' field is missing in the proof.json file.") from e
    except ValueError as e:
        raise ValueError(f"Invalid value in 'inputs' field: {e}") from e
    

def check_arguments(args):
    try:
        _check_strategies(args.strategies)
        _check_num_rounds(args.num_rounds)
        _check_num_clients(args.num_nodes)
        _check_iid_ratio(args.iid_ratio)
        _check_dishonest(args.dishonest)
        _check_fraction_fit(args.fraction_fit)
        _check_d(
            c=args.fraction_fit,
            k=args.num_nodes,
            iid_ratio=args.iid_ratio,
            d=args.d
        )
                
        return True
    except ValueError as e:
        print(f"Invalid arguments: {str(e)}")
        return False

def _check_strategies(strategies: List[str]):
    valid_strategies = {"FedAvg", "ZkAvg", "ContAvg", "PoC"}
    invalid_strategies = [s for s in strategies if s not in valid_strategies]
    if invalid_strategies:
        raise ValueError(f"Invalid strategies: {invalid_strategies}. Valid options are: {valid_strategies}")
    return True

def _check_num_clients(clients: int):
    if clients < 1:
        raise ValueError("Number of client must be greater than or equal to 1")
    elif clients > 50:
        raise ResourceWarning("A large number of clients may cause some clients to quit unexpectedly during training.")
    return True


def _check_num_rounds(rounds: int):
    if rounds < 1:
        raise ValueError("Number of rounds must be greater than or equal to 1")
    return True

def _check_iid_ratio(iid_ratio: float):
    if not 0.0 <= iid_ratio <= 1.0:
        raise ValueError("IID ratio must be between 0.0 and 1.0 (inclusive)")
    return True

def _check_dishonest(val: bool):
    # TODO check that val is used only when partitioner is iid_and_non_iid    
    return True

def _check_d(c: int, k: int, iid_ratio: int, d: int):
    """
    Check if the value of d is within the range of max(ck,1) and k.

    Parameters
    ----------
    c : int
        Fraction of clients.
    k : int
        Number of clients
    iid_ratio int:
        Fraction of iid clients 
    d : int
        Size of candidate set

    Returns
    -------
    bool
        True if the value of d is within the range, False otherwise.
    """
    iid_clients = int(k * iid_ratio)

    if not (max(c*iid_clients, 1) <= d <= iid_clients):
        raise ValueError(f"The value of d must be within the range of {max(c*iid_clients, 1)} and {iid_clients}. Current value: {d}")
    return True

def _check_fraction_fit(c: float):
    if not 0.0 <= c <= 1.0:
        raise ValueError("Fraction fit must be between 0.0 and 1.0 (inclusive)")
    return True