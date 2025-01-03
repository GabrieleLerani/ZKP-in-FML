from abc import ABC, abstractmethod
from typing import Tuple, Any

class ZkSNARK(ABC):
    """Abstract base class for zero-knowledge proof operations.
    
    This interface defines the standard operations that any zero-knowledge proof
    implementation should support, regardless of the underlying tool (ZoKrates, SnarkJS, etc.).
    """
    
    @abstractmethod
    def setup(self, program_file: str) -> None:
        """Set up the zero-knowledge proof system.
        
        Args:
            program_file (str): Path to the program file that defines the circuit
        """
        pass
    
    @abstractmethod
    def generate_proof(self, arguments: Tuple[Any, ...]) -> Any:
        """Generate a zero-knowledge proof with the given arguments.
        
        Args:
            arguments: Tuple of arguments required for the proof generation
            
        Returns:
            The generated proof in the implementation-specific format
        """
        pass
    
    @abstractmethod
    def verify_proof(self, verification_key: str) -> bool:
        """Verify a generated proof.
        
        Args:
            verification_key (str): Path or key data needed for verification
            
        Returns:
            bool: True if the proof is valid, False otherwise
        """
        pass


class SmartContractVerifier(ABC):
    """Interface for ZKP systems that support smart contract verification."""
    
    @abstractmethod
    def verify_proof_with_smart_contract(self, client_data: Any):
        """Verify a proof using a deployed smart contract."""
        pass
    
    @abstractmethod
    def generate_smart_contract(self, node_id: str) -> Tuple[str, Any]:
        """Generate and deploy a verification smart contract."""
        pass


    