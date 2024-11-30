import shutil
import os
import json
from typing import List


def generate_file_suffix(params: dict) -> str:
    """
    Generate a file suffix based on the provided parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing the following keys:
        - num_rounds: int
        - partitioner: str
        - dataset: str
        - secaggplus: bool
        - alpha: float, optional
        - x_non_iid: int, optional
        - iid_ratio: float, optional
        - dishonest: bool, optional
        - balanced: bool, optional
        - iid_data_fraction: float, optional

    Returns
    -------
    str
        Generated file suffix.
    """
    num_rounds = params["num_rounds"]
    partitioner = params["partitioner"]
    dataset = params["dataset_name"]
    secaggplus = params.get("secaggplus", False)
    alpha = params.get("alpha", None)
    x_non_iid = params.get("x_non_iid", None)
    iid_ratio = params.get("iid_ratio", None)
    dishonest = params.get("dishonest", False)
    balanced = params.get("balanced", False)
    iid_data_fraction = params.get("iid_data_fraction", None)

    include_alpha = (f"_alpha={alpha}" if partitioner == "dirichlet" and alpha is not None else "")
    include_x = (f"_x={x_non_iid}" if partitioner == "iid_and_non_iid" and x_non_iid is not None else "")
    include_iid_ratio = (f"_iid_ratio={iid_ratio}" if partitioner == "iid_and_non_iid" and iid_ratio is not None else "")
    include_dishonest = (f"_dishonest" if dishonest else "")
    include_sec_agg = ("SecAgg" if secaggplus else "")
    include_balanced = (f"_bal={balanced}" if partitioner == "iid_and_non_iid" else "")
    include_iid_data_fraction = (f"_iid_df={iid_data_fraction}" if include_balanced and iid_data_fraction is not None else "")

    file_suffix = (
        f"R={num_rounds}"
        f"_P={partitioner}"
        f"_D={dataset}"
        + include_sec_agg
        + include_alpha 
        + include_x 
        + include_iid_ratio 
        + include_dishonest
        + include_balanced
        + include_iid_data_fraction
    )
    
    return file_suffix

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
        _check_iid_data_fraction(args.iid_data_fraction)
        _check_dishonest(args.dishonest, args.partitioner)
        _check_fraction_fit(args.fraction_fit)
        _check_d(
            c=args.fraction_fit,
            k=args.num_nodes,
            iid_ratio=args.iid_ratio,
            d=args.d,
            dishonest=args.dishonest
        )
        _check_balanced(args.partitioner, args.balanced)
                
        return True
    except ValueError as e:
        print(f"Invalid arguments: {str(e)}")
        return False

def _check_strategies(strategies: List[str]):
    valid_strategies = {"FedAvg", "ZkAvg", "ContAvg", "PoC", "PoCZk"}
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

def _check_iid_ratio(iid_data_fraction: float):
    if not 0.0 <= iid_data_fraction <= 1.0:
        raise ValueError("IID ratio must be between 0.0 and 1.0 (inclusive)")
    return True

def _check_iid_data_fraction(iid_ratio: float):
    if not 0.0 <= iid_ratio <= 1.0:
        raise ValueError("IID data fraction must be between 0.0 and 1.0 (inclusive)")
    return True

def _check_dishonest(dishonest: bool, partitioner):
    if dishonest and partitioner != "iid_and_non_iid":
        raise ValueError("Dishonest flag can only be used with iid_and_non_iid partitioner.")
    return True

def _check_d(c: int, k: int, iid_ratio: int, d: int, dishonest: bool):
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
    iid_clients = int(k * iid_ratio) if dishonest else k

    if not (max(c*iid_clients, 1) <= d <= iid_clients):
        raise ValueError(f"The value of d must be within the range of {max(c*iid_clients, 1)} and {iid_clients}. Current value: {d}")
    return True

def _check_fraction_fit(c: float):
    if not 0.0 <= c <= 1.0:
        raise ValueError("Fraction fit must be between 0.0 and 1.0 (inclusive)")
    return True

def _check_balanced(partitioner: str, balanced: bool):
    if balanced and partitioner != "iid_and_non_iid":
        raise ValueError("Balanced flag can only be used with iid_and_non_iid partitioner.")
    return True