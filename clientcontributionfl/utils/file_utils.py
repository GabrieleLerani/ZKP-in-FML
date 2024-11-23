import shutil
import os
import json

def cleanup_proofs():
    """Remove the proofs directory and all its contents if it exists."""
    proofs_dir = os.path.join(os.getcwd(), "proofs")
    if os.path.exists(proofs_dir):
        try:
            shutil.rmtree(proofs_dir)
            print(f"Successfully removed {proofs_dir} and all its contents")
        except Exception as e:
            print(f"Error while removing directory {proofs_dir}: {str(e)}")


import json

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