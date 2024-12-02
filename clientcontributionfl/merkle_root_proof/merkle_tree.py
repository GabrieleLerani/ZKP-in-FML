
from torch.utils.data import DataLoader
from typing import Tuple, List
from .utils.hash_utils import format_proof_path


def build_merkle_tree_level(current_level: list[bytes]) -> list:
    """Build a single level of the merkle tree."""
    level = []
    for i in range(0, len(current_level), 2):
        # Combine pairs of hashes to form the next level
        if i + 1 < len(current_level):
            combined = [current_level[i], current_level[i + 1]]
        else:
            # Handle odd number of nodes (duplicate last node)
            combined = [current_level[i], current_level[i]]
        
        level.append(hash(combined))

    return level

def build_merkle_tree(leaves: list[bytes]) -> list[list[bytes]]:
    """Constructs a Merkle tree and returns all levels."""
    tree = [leaves]  # Start with the leaf nodes
    current_level = leaves

    while len(current_level) > 1:
        next_level = build_merkle_tree_level(current_level)
        tree.append(next_level)
        current_level = next_level

    return tree

def compute_merkle_proof(tree: list[list[bytes]], leaf_index: int) -> Tuple[list[bytes], list[int], bytes]:
    """Generates the Merkle proof for a given leaf index."""
    proof = []
    directions = []
    start_index = leaf_index
    for level in range(len(tree) - 1):
        sibling_index = leaf_index ^ 1  # XOR with 1 to get the sibling index
        directions.append(leaf_index % 2)  # 0 if left, 1 if right
        proof.append(tree[level][sibling_index])
        leaf_index //= 2
    return proof, directions, tree[0][start_index]

def hash_batch(batch) -> int:
    """Hashes a batch of images and labels into a single hash."""
    feature, label = list(batch.keys())
    images, labels = batch[feature], batch[label]
    batch_data = []
    # Flatten images and convert to bytes
    for image, label in zip(images, labels):
        image_bytes = image.numpy().tobytes()
        label_bytes = label.numpy().tobytes()
        combined_data = image_bytes + label_bytes
        batch_data.append(combined_data)

    combined_batch_data = b''.join(batch_data)
    return hash(combined_batch_data)  


def compute_tree_leaves_batch(dataloader: DataLoader) -> List[int]:
    """Compute leaves of the tree as hash of each batch in dataloader."""
    leaf_hashes = []
    for batch in dataloader:
        hashed_batch = hash_batch(batch)
        leaf_hashes.append(hashed_batch)  
    return leaf_hashes

def compute_tree_leaves_samples(dataloader: DataLoader) -> List[int]:
    """Compute leaves of the tree as hash of each sample in each batch in dataloader. 
        Usually very expensive for big datasets.
    """
    leaf_hashes = []
    for batch in dataloader:
        images, labels = batch
        for image, label in zip(images, labels):
            image_bytes = image.numpy().tobytes()
            label_bytes = label.numpy().tobytes()
            combined_data = image_bytes + label_bytes
            leaf_hash = hash(combined_data)  # Hash each individual sample
            leaf_hashes.append(leaf_hash)  
    return leaf_hashes



def compute_merkle_tree(dataloader: DataLoader) -> List[List[int]]:
    """
    Computes the Merkle tree of all the dataset images in a PyTorch DataLoader.
    """
    tree_leaves = compute_tree_leaves_batch(dataloader)
    #tree_leaves = compute_tree_leaves_samples(dataloader)
    merkle_tree = build_merkle_tree(tree_leaves)

    return merkle_tree


def verify_merkle_proof(
    root: int, 
    leaf: int, 
    directions: List[int], 
    path: List[int]
) -> Tuple[bool, str]:
    """
    Verifies the Merkle tree proof for a given leaf and its path to the root.

    This function takes the root hash of the Merkle tree, the hash of the leaf node, 
    the directions (left or right) to traverse the tree, and the path (sibling hashes) 
    to the root. It then verifies the proof by recomputing the hashes along the path 
    and comparing the final computed hash with the given root hash.

    Args:
        root (int): The root hash of the Merkle tree.
        leaf (int): The hash of the leaf node.
        directions (List[int]): A list of directions (0 for left, 1 for right) to traverse the tree.
        path (List[int]): A list of sibling hashes along the path to the root.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating whether the proof is valid, 
        and a string containing debugging information.
    """
    
    debug_info = []
    
    current_hash = leaf
    debug_info.append(f"Initial leaf hash: {current_hash}")
    
    for level, (sibling_hash, direction) in enumerate(zip(path, directions)):
        if direction == 0:
            combined = [current_hash, sibling_hash]
            debug_info.append(f"Level {level}: Left child, combine current + sibling")
        else:
            
            combined = [sibling_hash, current_hash]
            debug_info.append(f"Level {level}: Right child, combine sibling + current")
        
        new_hash = hash(combined)
        debug_info.append(f"Level {level}: Combined hash: {new_hash}")
        debug_info.append(f"Level {level}: Expected sibling hash: {sibling_hash}")

        current_hash = new_hash
    
    
    if current_hash != root:
        debug_info.append(f"Final mismatch: computed root {current_hash}, expected root {root}")
        return False, "\n".join(debug_info)
    
    return True, "Proof is valid."