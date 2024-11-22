import shutil
import os

# TODO move into this file other file functions
def cleanup_proofs():
    """Remove the proofs directory and all its contents if it exists."""
    proofs_dir = os.path.join(os.getcwd(), "proofs")
    if os.path.exists(proofs_dir):
        try:
            shutil.rmtree(proofs_dir)
            print(f"Successfully removed {proofs_dir} and all its contents")
        except Exception as e:
            print(f"Error while removing directory {proofs_dir}: {str(e)}")