import os
import subprocess
from typing import Tuple

class Zokrates:
    """A wrapper class for interacting with the ZoKrates zero-knowledge proof system.

    This class provides an interface to compile ZoKrates programs, generate proofs, and verify them.
    """

    def __init__(self, working_dir : str = None):
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
        
        # TODO test if it works
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
        """Compile the ZoKrates program and set up the proving and verification keys.
        
        This method compiles the contribution verification program and generates the proving
        and verification keys needed for the zero-knowledge proofs.
        
        Raises:
            ValueError: If compilation or setup fails
        """
        self._run_command(f"zokrates compile -i {zok_file_path}")
        self._run_command("zokrates setup")

    def generate_proof(self, arguments: Tuple[str]):
        """Generate a zero-knowledge proof with the given data.
        
        Args:
            arguments: Tuple[str] Arbitrary positional arguments for the ZoKrates command
            
        Returns:
            str: Generated proof output
        """
        
        arguments = " ".join(map(str, arguments))

        self._run_command(f"zokrates compute-witness -a {arguments}")
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


