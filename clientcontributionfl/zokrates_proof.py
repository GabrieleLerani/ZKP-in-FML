import os
import subprocess
from clientcontributionfl.utils import compute_zk_score

class Zokrates:
    def __init__(self, working_dir = None):
        """Initialize Zokrates with a working directory."""
        if working_dir != None:
            self.working_dir = working_dir
            # Create working directory if it doesn't exist
            os.makedirs(self.working_dir, exist_ok=True)

    def _run_command(self, command, path = None):
        """Run a ZoKrates command using subprocess."""
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
        """Compile the ZoKrates program and set up the proving and verification keys."""
        zok_file = "../../clientcontributionfl/contribution.zok "
        
        self._run_command(f"zokrates compile -i {zok_file} --debug")
        self._run_command("zokrates setup")

    # TODO consider to pass parameters ad dict or other structure
    def generate_proof(self, counts, scale, beta, mean_val, thr, score):
        """Generate a zero-knowledge proof with the given data."""
        counts_str = " ".join(map(str, counts))
        self._run_command(f"zokrates compute-witness -a {counts_str} {scale} {beta} {mean_val} {thr} {score}")
        proof = self._run_command("zokrates generate-proof")
        return proof

    def verify_proof(self, key_path: str):
        """Verify the generated proof."""
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
    
    score = compute_zk_score(counts=counts, scale=scale, beta=beta, gamma=gamma)
    
    zokrates = Zokrates(working_dir)
    zokrates.setup()
    
    proof = zokrates.generate_proof(counts, scale, beta, mean_val, gamma, score)
    verification = zokrates.verify_proof()
    print(verification)
