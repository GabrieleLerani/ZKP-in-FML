from typing import Tuple, List, Union
from clientcontributionfl.merkle_root_proof.hash import FIELD_MODULUS, poseidon_hash

def list_to_field_elements(value: List[int]) -> List[int]:
    """
    Converts a list of int into a list of elements in the finite field F_p.

    Args:
        s: The input string to be hashed.
        
    Returns:
        A list of integers representing the string in F_p.
    """
    
    field_elements = [v % FIELD_MODULUS for v in value]
    
    return field_elements


def bytes_to_field_elements(value: bytes) -> List[int]:
    """
    Converts bytes into a list of elements in the finite field F_p.

    Args:
        s: The input bytes.
        
    Returns:
        A list of integers representing the string in F_p.
    """
    
    chunk_size = len(value) // 2
    
    chunks = [value[i:i + chunk_size] for i in range(0, len(value), chunk_size)]
    
    # Convert each chunk into an integer
    integers = [int.from_bytes(chunk, byteorder='big') for chunk in chunks]
    
    # Map integers into the field F_p
    field_elements = [x % FIELD_MODULUS for x in integers]
    
    return field_elements


def hash(value: Union[bytes, List[int]]) -> int:
    """Computes the poseidon hash of the given value."""
    
    if isinstance(value, bytes):
        input_vec = bytes_to_field_elements(value)
    elif isinstance(value, list):
        input_vec = list_to_field_elements(value)
    else:
        raise ValueError("Input type not supported. Must be bytes or list of integers.")
    
    poseidon_digest = poseidon_hash(input_vec)
    
    return poseidon_digest

def format_proof_path(path: list[int]) -> str:
    return " ".join([str(node) for node in path])