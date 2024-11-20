import subprocess
import shutil
import os

# TODO move into other module with only utils on file
def cleanup_proofs():
    """Remove the proofs directory and all its contents if it exists."""
    proofs_dir = os.path.join(os.getcwd(), "proofs")
    if os.path.exists(proofs_dir):
        try:
            shutil.rmtree(proofs_dir)
            print(f"Successfully removed {proofs_dir} and all its contents")
        except Exception as e:
            print(f"Error while removing directory {proofs_dir}: {str(e)}")

def run_simulation(alpha):
    command = [
        "flower-simulation",
        "--app", ".",
        "--num-supernodes", "10",
        "--run-config", f"num_rounds=30"
    ]
    
    print(" ".join(command))
    result = subprocess.run(command, stderr=subprocess.STDOUT, stdout=None, text=True)
    if result.returncode != 0:
            raise ValueError(f"Error: {result.stderr.decode()}")
    print(f"Simulation completed for alpha = {alpha}\n")

def main():
    
    cleanup_proofs()
    
    alpha_values = [0.03,]# 0.1, 0.5, 1.0, 2.0]
    
    for alpha in alpha_values:
        run_simulation(alpha)

if __name__ == "__main__":
    main()