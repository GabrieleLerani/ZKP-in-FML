import subprocess

def run_simulation(alpha):
    command = [
        "flower-simulation",
        "--app", ".",
        "--num-supernodes", "10",
        "--run-config", f"num_rounds=30 alpha={alpha}"
    ]
    
    print(f"Running simulation with alpha = {alpha}")
    subprocess.run(command, check=True)
    print(f"Simulation completed for alpha = {alpha}\n")

def main():
    alpha_values = [0.03, 0.1, 0.5, 1.0, 2.0]
    
    for alpha in alpha_values:
        run_simulation(alpha)

if __name__ == "__main__":
    main()