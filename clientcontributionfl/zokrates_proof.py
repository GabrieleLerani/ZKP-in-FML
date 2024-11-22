import os
import subprocess
from clientcontributionfl.utils import compute_score
from logging import INFO, DEBUG
from flwr.common.logger import log

class Zokrates:
    """A wrapper class for interacting with the ZoKrates zero-knowledge proof system.

    This class provides an interface to compile ZoKrates programs, generate proofs, and verify them.
    It is used in the federated learning system to generate zero-knowledge proofs of client contributions
    without revealing the actual dataset value.

    Attributes:
        working_dir (str): Directory where ZoKrates files and proofs will be stored

    Example:
        >>> # Initialize ZoKrates with a working directory
        >>> zokrates = Zokrates("proofs/client_1")
        >>> 
        >>> # Set up the proving system
        >>> zokrates.setup()
        >>> 
        >>> # Generate a proof with client contribution data
        >>> counts = [300, 100, 6000, 1000, 4000]
        >>> proof = zokrates.generate_proof(
        ...     counts=counts,
        ...     scale=1000,
        ...     beta=1, 
        ...     mean_val=26400,
        ...     thr=100,
        ...     score=42
        ... )
        >>> 
        >>> # Verify the generated proof
        >>> verification = zokrates.verify_proof("proofs/client_1")
    """

    def __init__(self, working_dir = None):
        """Initialize Zokrates with a working directory.
        
        Args:
            working_dir (str, optional): Directory to store ZoKrates files and proofs.
                If provided, the directory will be created if it doesn't exist.
        """
        if working_dir != None:
            self.working_dir = working_dir
            # Create working directory if it doesn't exist
            os.makedirs(self.working_dir, exist_ok=True)

    def _run_command(self, command, path = None):
        """Run a ZoKrates command using subprocess.
        
        Args:
            command (str): The ZoKrates command to execute
            path (str, optional): Working directory for command execution.
                Defaults to self.working_dir if not specified.
                
        Returns:
            str: Command output as string
            
        Raises:
            ValueError: If command execution fails
        """
        cmd_parts = command.split(' ', 1)
        full_command = f"~/.zokrates/bin/zokrates {cmd_parts[1]}" if len(cmd_parts) > 1 else "/usr/local/bin/zokrates"
        
        
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

    def setup(self):
        """Compile the ZoKrates program and set up the proving and verification keys.
        
        This method compiles the contribution verification program and generates the proving
        and verification keys needed for the zero-knowledge proofs.
        
        Raises:
            ValueError: If compilation or setup fails
        """
        zok_file = "../../clientcontributionfl/contribution.zok "
        
        self._run_command(f"zokrates compile -i {zok_file} --debug")
        self._run_command("zokrates setup")

    # TODO consider to pass parameters as dict
    def generate_proof(self, counts, scale, beta, mean_val, thr, score):
        """Generate a zero-knowledge proof with the given data.
        
        Args:
            counts (List[int]): List of label counts from the client
            scale (int): Scaling factor to convert floats to integers
            beta (int): Weight factor for variance in score calculation
            mean_val (int): Mean value of the counts
            thr (int): Threshold for contribution evaluation
            score (int): Calculated contribution score
            
        Returns:
            str: Generated proof output
            
        Raises:
            ValueError: If proof generation fails
        """
        counts_str = " ".join(map(str, counts))
        self._run_command(f"zokrates compute-witness -a {counts_str} {scale} {beta} {mean_val} {thr} {score}")
        proof = self._run_command("zokrates generate-proof")
        return proof

    def verify_proof(self, key_path: str):
        """Verify the generated proof.
        
        Args:
            key_path (str): Path to the directory containing verification key
            
        Returns:
            str: Verification result output
            
        Raises:
            ValueError: If verification fails
        """
        # TODO include the export verifier for solidity
        return self._run_command("zokrates verify", path=key_path)

# Example usage
if __name__ == "__main__":
    counts = [30000, 1000, 60000, 1000, 40000, 50, 30, 20, 9000, 4000]
    scale = 1000
    beta = 1
    mean_val = int(sum(counts) / len(counts))
    gamma = 100
    working_dir = os.path.join("proofs", f"client_12")
    
    score = compute_score(counts=counts, scale=scale, beta=beta, gamma=gamma)
    
    zokrates = Zokrates(working_dir)
    zokrates.setup()
    
    proof = zokrates.generate_proof(counts, scale, beta, mean_val, gamma, score)
    verification = zokrates.verify_proof()
    print(verification)
