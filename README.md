# Privacy preserving contribution in Federated Learning

This project implements strategies for verifying and enforcing honest client contribution reporting in Federated Learning using the Flower framework. It includes both a basic contribution scoring system and a zero-knowledge proof verification approach.

## Project Overview

The project tests three main strategies for federated learning:

1. **FedAvg**: Standard Federated Averaging algorithm (baseline)
2. **ContAvg**: Contribution-based client selection where clients report dataset quality scores
3. **ZkAvg**: Zero-knowledge proof verification of client contributions using Zokrates

### Key Features

- Custom client contribution scoring based on dataset characteristics
- Zero-knowledge proof verification using Zokrates
- Support for simulating honest/dishonest clients
- Custom data partitioning for IID and non-IID distribution testing
- Comparative analysis between strategies using centralized accuracy metrics

## Project Structure

clientcontributionfl/
├── client_strategy/
│ ├── fedavg_client.py # Base FedAvg client implementation
│ ├── contribution_client.py # Client with contribution scoring
│ └── zkavg_client.py # Client with ZK proof generation
├── server_strategy/
│ ├── contribution_strategy.py # Server-side contribution verification
│ └── zk_strategy.py # Server-side ZK proof verification
├── utils/
│ ├── file_utils.py # Proof file handling utilities
│ ├── plot_util.py # Visualization utilities
│ └── score.py # Contribution scoring functions
├── dataset.py # Dataset loading and partitioning
├── models.py # Neural network model definitions
├── server_app.py # Server configuration
├── client_app.py # Client configuration
└── zokrates_proof.py # Zokrates integration
  

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ZKP-in-FML.git
   cd ZKP-in-FML
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies defined in `pyproject.toml` as well as the clientcontributionfl package.
   ```
   pip install -e .
   ```

4. Install ZoKrates for Zero-Knowledge proof:
   ```
   curl -LSfs get.zokrat.es | sh
   ```

## Implementation Details

### ContAvg Strategy
Clients compute a score based on their local dataset distribution and submit it to the server. The server uses these scores to prioritize client selection during training. However, dishonest clients can submit fake scores to increase their selection probability.

### ZkAvg Strategy
To prevent score manipulation, clients must provide zero-knowledge proofs of their contribution using Zokrates. The process works as follows:

1. Each client compiles a `contribution.zok` circuit file
2. Client provides private input (e.g., [40,30,100] representing label distribution)
3. Circuit computes the contribution score and verifies it matches the claimed value
4. Server verifies the proof before allowing client participation

### Custom partitioner
The beforementioned strategies are well suited whene data are not IID between clients. Flower already comes with many way partitioner (Dirichlet, Linear, Size, Pathological, etc.) however none of them permits to have a portion of client with IID data and another with non-IID, simulating a scenario where a group of nodes has good quality data and another not. To cope with this limitation I implemented a flower Partitioner called `LabelBasedPartitioner`.

The `LabelBasedPartitioner` allows you to specify:
- `num_partitions`: Total number of clients/partitions
- `iid_ratio`: Fraction of clients that will receive IID data (between 0 and 1)
- `x`: The label each non_iid partition will have 

```python
# Initialize the partitioner with desired parameters
partitioner = LabelBasedPartitioner(num_partitions=10, iid_ratio=0.7, x=2)

# Create federated dataset using the partitioner
fds = FederatedDataset(
    dataset="ylecun/mnist",
    partitioners={"train": partitioner}
)

# Visualize the label distribution across partitions
fig, ax, df = plot_label_distributions(
    partitioner=fds.partitioners["train"],
    label_name="label",
    plot_type="bar",
    size_unit="absolute",
    legend=True,
)
```


Note: Due to Flower's limitations in file transfer, the implementation uses shared working directory paths between clients and server for proof verification.

### Testing Setup
- Custom partitioner creating both IID and non-IID client distributions
- Simulation of dishonest non-IID clients submitting fake scores
- Comparative analysis against standard FedAvg
- Centralized accuracy evaluation using a pre-defined test dataset

## Tools and Dependencies

- Flower (FL framework)
- Zokrates (Zero-knowledge proof system)
- PyTorch (Deep learning)
- Python 3.8+


## Configuration

The project configuration is managed through the `pyproject.toml` file. Key configurations include:

- Number of rounds: 10
- Fraction of clients for fitting and evaluation: 100%
- Number of clients per round: 1
- Dataset: MNIST
- Distribution: Dirichlet (α = 0.03)
- Batch size: 10
- SecAgg+ parameters: 3 shares, reconstruction threshold of 2
- Training parameters: learning rate 0.1, 2 epochs per round

You can modify these parameters in the `pyproject.toml` file under the `[tool.flwr.app.config]` section.

## Running the Project

To run the Federated Learning simulation:
   ```
   python main.py --strategies ZkAvg,ContAvg,FedAvg --num_rounds 10 --num_nodes 10
   ```
You can override other configurations parameters (learning rate, batch size etc.)directly changing `pyproject.toml`.

## Results

The results of the training, including accuracy scores and any generated plots, will be saved in the `results/` directory.
