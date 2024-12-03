from typing import Tuple
import torch
import os
from clientcontributionfl.models import train
from clientcontributionfl.merkle_root_proof import compute_merkle_tree, compute_merkle_proof
from .fedavg_client import FedAvgClient
from clientcontributionfl.zokrates_proof import Zokrates
from clientcontributionfl.utils.file_utils import read_file_as_bytes

class MerkleProofClient(FedAvgClient):
    """
    Client that compute a merkle tree of its dataset. Samples from the dataset can be extracted
    randomly and sent to the server along with a merkle proof which is composed by a root, leaf
    and a path. Server can check that the received samples really belongs to the original dataset,
    thus guaranteeing the integrity and authenticity of the data without needing to possess the entire 
    dataset. 
    """
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        #self.merkle_tree = compute_merkle_tree(self.trainloader) 
        self.path_proof_dir = os.path.join("proofs", f"client_{self.node_id}")
        self.zk = Zokrates(working_dir=self.path_proof_dir)


    def _pick_random_batch(self):
        
        batch_index = torch.randint(0, len(self.trainloader), (1,)).item()
        
        # Retrieve the batch at the sampled index
        for i, batch in enumerate(self.trainloader):
            if i == batch_index:
                return batch, batch_index
            
        raise IndexError(f"{batch_index} not in dataloader.")

    def _pick_random_sample(self):
        # Sample a random batch and then a random image from that batch
        (batch_data, batch_labels), batch_index = self._pick_random_batch()
        image_index = torch.randint(0, batch_data.size(0), (1,)).item()  # Random image index in the batch
        return batch_data[image_index], batch_labels[image_index], batch_index

    def _format_proof_arguments(self, merkle_tree, path, direction_selector, leaf) -> tuple:
        direction_selector_str = " ".join(map(str, direction_selector))
        formatted_path = " ".join([str(node) for node in path])
        root = merkle_tree[-1][0]

        return root, leaf, direction_selector_str, formatted_path

    def _generate_proof_files(self, merkle_tree, batch_index) -> Tuple[bytes]:
        """Generate proof and read the corresponding files."""
        path, direction_selector, leaf = compute_merkle_proof(merkle_tree, batch_index)
        
        program_path = self._create_zok_file(tree_depth=len(path))
        
        arguments = self._format_proof_arguments(merkle_tree, path, direction_selector, leaf)
        
        self.zk.setup(zok_file_path=f"../../{program_path}")
        self.zk.generate_proof(arguments)

        proof_bytes = read_file_as_bytes(os.path.join(self.path_proof_dir, "proof.json"))
        verification_bytes = read_file_as_bytes(os.path.join(self.path_proof_dir, "verification.key"))

        return proof_bytes, verification_bytes
    

    def _create_zok_file(self, tree_depth: int) -> str:
        """
        This functions create a dynamic zok file in order to simulate dynamic
        proof generation. (Zokrates only supports static arrays whose size is known
        at compile time).
        """
        template = generate_zokrates_template(tree_depth)
        template_path = os.path.join(self.path_proof_dir, "merkle_proof.zok")
        with open(template_path, "w") as f:
            f.write(template)
        return template_path

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.
        Skip training in first round and only return verification and proof files.
        """
        self.set_parameters(parameters)
        params = {}

        # compute merkle proof 
        if config["server_round"] == 1:
            merkle_tree = compute_merkle_tree(self.trainloader) 
            
            # TODO decide how to use sampled batch
            batch, batch_index = self._pick_random_batch()
            
            proof_bytes, verification_bytes = self._generate_proof_files(merkle_tree,batch_index)

            params["proof"] = proof_bytes
            params["verification_key"] = verification_bytes
        
        # train the model in any other rounds
        else:
            # learning rate decay
            optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
            train(
                self.model, 
                self.trainloader, 
                self.config['num_epochs'], 
                self.config['device'], 
                optimizer,
                self.criterion,
                self.accuracy_metric
            )

        return self.get_parameters({}), len(self.trainloader), params


def generate_zokrates_template(tree_depth):
    """
    Generate a .zok file with a dynamic tree depth.
    """
    template = f"""
import "hashes/poseidon/poseidon" as poseidon;

// Define tree depth
const u32 TREE_DEPTH = {tree_depth};

def select(bool condition, field left, field right) -> (field, field) {{
    return (condition ? right : left, condition ? left : right);
}}

def merkleTreeProof<DEPTH>(
    field root, 
    field leaf, 
    bool[DEPTH] directionSelector, 
    field[DEPTH] path
) -> bool {{
    // Start from the leaf
    field mut digest = leaf;

    // Loop up the tree
    for u32 i in 0..DEPTH {{
        (field, field) s = select(directionSelector[i], digest, path[i]);
        digest = poseidon([s.0, s.1]);
    }}

    return digest == root;
}}

// Main function
def main(
    field treeRoot, 
    field leaf, 
    private bool[TREE_DEPTH] directionSelector, 
    private field[TREE_DEPTH] path
) {{
    assert(merkleTreeProof(treeRoot, leaf, directionSelector, path));
}}
"""
    return template

