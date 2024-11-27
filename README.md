# Privacy preserving contribution in Federated Learning 🔐

This project implements strategies for verifying and enforcing honest client contribution reporting in Federated Learning using the Flower framework. It includes both a basic contribution scoring system and a zero-knowledge proof verification approach.

## Table of Contents 📑
- [Project Overview](#project-overview-)
- [Project Structure](#project-structure-)
- [Tools and Dependencies](#tools-and-dependencies-%EF%B8%8F)
- [Installation](#installation-%EF%B8%8F)
- [Implementation Details](#implementation-details-)
  - [Score contribution](#score-contribution-)
  - [ContAvg Strategy](#contavg-strategy-)
  - [ZkAvg Strategy](#zkavg-strategy-)
  - [Custom partitioner](#custom-partitioner-)
  - [Testing Setup](#testing-setup-)
- [Configuration](#configuration-%EF%B8%8F)
- [Running simulation](#running-simulation-%EF%B8%8F)
- [Results](#results-)

## Project Overview 🎯
This project explores innovative strategies to enhance **Federated Learning (FL)** while preserving privacy and optimizing client contributions. Federated Learning relies on decentralized data from client devices, but its effectiveness depends on fair and accurate evaluation of client contributions. This project introduces three strategies aimed at improving contribution evaluation, ensuring privacy, and accelerating convergence:  

### 1. ContAvg  
The **Contribution Average (ContAvg)** strategy establishes a baseline by introducing a threshold-based client selection mechanism. Each client computes a contribution score based on its local data and submits this score to the server. The server selects only the clients with scores exceeding the defined threshold for the training process.  
- **Challenge**: A malicious client can submit fake scores, compromising model integrity.  

### 2. ZkAvg  
To address the limitations of ContAvg, the **Zero-Knowledge Contribution Average (ZkAvg)** strategy ensures that contribution scores remain trustworthy without exposing sensitive dataset details.  
- **Approach**: Clients submit **zero-knowledge proofs** of their contribution scores. The server verifies these proofs without accessing the underlying data.  
- **Advantage**: This technique mitigates malicious activity by only accepting clients with valid proofs, upholding both privacy and reliability.  

### 3. Enhanced Power of Choice  
Building on the concepts of contribution evaluation, this strategy adapts the **Power of Choice** algorithm to better align with Federated Learning needs.  
- **Key Innovations**:  
  - The probability $`p_k`$ of selecting a client is based on normalized contribution scores. These scores consider not just the size of the dataset but also its **label distribution**, capturing richer information about the dataset's quality.  
  - The server selects clients with the highest local loss from a candidate set, ensuring efficient utilization of the most promising participants.  
- **Integration with ZkAvg**: By combining this strategy with ZkAvg, the framework achieves both **privacy-preserving guarantees** and **faster model convergence**, outperforming simpler approaches like ContAvg and FedAvg.  

### Why It Matters  
This project advances Federated Learning by providing methods that:  
- Protect client data privacy while ensuring trustworthy contribution evaluations.  
- Optimize client selection for faster and more robust model convergence.  
- Enable adaptive strategies that account for dataset quality and distribution.  

These strategies can be instrumental in scenarios where trust, efficiency, and privacy are critical, such as healthcare, finance, and edge computing applications.  

---

## Features  
- **ContAvg**: Threshold-based client selection with contribution scores.  
- **ZkAvg**: Zero-knowledge proofs for privacy-preserving contribution evaluation.  
- **Enhanced Power of Choice**: Advanced client selection based on contribution scores and local loss.  

## Project Structure 📁
```
clientcontributionfl/
  ├── client_strategy/
  │ └── *.py # ZkAvg, ContAvg, FedAvg standard clients
  ├── server_strategy/
  │ └── *.py # server strategies like ZkAvg and ContAvg
  ├── models/
  │ └── *.py # Neural network models and training functions
  ├── utils/
  │ └── *.py # Many utilities
  ├── dataset.py # Dataset loading and partitioning
  ├── custom_partitioner.py # label based partitioner as described below
  ├── server_utils.py # server functions used during training
  ├── server_app.py # Server configuration
  ├── client_app.py # Client configuration
  ├── zokrates_proof.py # Class to interact with zokrates tool
  └── contribution.zok # zokrates file to prove contribution

results/
  ├── label_dist/
  │ └── *.png # many plots of different partitioner
  └── simulation/ # simultions result for different configurations

notebook/
  └── *.ipynb # notebook with some example, still in progress

main.py # Entry point to run simulations
pyproject.toml # configuration file

```

## Tools and Dependencies 🛠️

- Flower (FL framework)
- Zokrates (Zero-knowledge proof system)
- PyTorch (Deep learning)
- Python 3.8+

## Installation ⚙️

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

## Implementation Details 🔍
[`ContAvg`](clientcontributionfl/server_strategy/contribution_strategy.py) and [`ZkAvg`](clientcontributionfl/server_strategy/zk_strategy.py) are very similar strategies: the general idea is to compute a score which represents client contribution to the global model training. However when client are dishonest they can submit a fake score and increase the likelihood of being selected for training. Zero-knowledge proofs guarantee to discard malicious clients and that's the main contribution of `ZkAvg`. Trivially, when all actor are honest `ZkAvg` and `ContAvg` are the same algorithm.

In this implementation, the probability of a client being selected in the candidate set is based on its contribution score rather than the fraction of data it holds. This score reflects not only the amount of data but also its diversity, allowing for a more nuanced selection of clients for training.

The strategy operates in the following manner:
1. In the first round, all clients compute their dataset scores and generate zero-knowledge proofs to verify these scores.
2. In subsequent rounds, a candidate set of clients is sampled based on their scores.
3. Local loss estimates are requested from the sampled clients.
4. Clients with the highest local losses are selected to participate in training.

This approach aims to enhance the robustness of client selection by prioritizing clients that not only have a significant amount of data but also diverse datasets, thereby improving the overall training process.

### Score contribution 
Implement a function to evaluate the dataset quality is not trivial, especially if the computation must be executed in a ZoKrates program, where all variables are defined as elements of a prime field, which means no floating number operations are permitted. Therefore one of the simplest idea I had is to quantify the variance and the diversity of labels on each partition.

The [`compute_score`](clientcontributionfl/utils/score.py) function, which evaluates the quality of a client's dataset based on its label distribution considers two main factors:

1. **Variance**: Measures how spread out the label counts are from their mean value. Higher variance indicates less balanced data.
2. **Diversity**: Counts how many labels appear above a certain threshold. Higher diversity indicates more representative data.

The score is calculated using the formula:
```python
score = (beta * variance) + (diversity * scale)
```

Where:
- `counts`: List of integers representing the count of each label in the client's dataset
- `scale`: Scaling factor to weight the diversity component
- `beta`: Weight factor for the variance component
- `thr`: Threshold value to determine significant label presence

For example:
```python
counts = [100, 200, 100]  # Label distribution
scale = 1000  # Scaling factor
beta = 1      # Variance weight
thr = 100     # Threshold for counting diverse labels
score = compute_score(counts, scale, beta, thr)  # Returns weighted score
print(scores)
10667
```

A high score indicates:
- Uneven distribution of labels (high variance)
- Good representation of multiple classes above threshold (high diversity)

A low score indicates:
- More uniform distribution of labels (low variance)
- Few classes represented above threshold (low diversity)

### ContAvg Strategy
Clients compute a score based on their local dataset distribution and submit it to the server. The server uses these scores to prioritize client selection during training. However, dishonest clients can submit fake scores to increase their selection probability.

### ZkAvg Strategy
To prevent score manipulation, clients must provide zero-knowledge proofs of their contribution using Zokrates. The process works as follows:

1. Each client compiles a [`contribution.zok`](clientcontributionfl/contribution.zok) circuit file
2. Client provides private input (e.g., [40,30,100] representing label distribution) and their score as witness.
3. Circuit computes the contribution score and verifies it matches the claimed value.
4. Server verifies the proof before allowing client participation.

Note: Due to Flower's limitations in file transfer, the implementation uses shared working directory paths between clients and server for proof verification.

### Custom partitioner
The beforementioned strategies are well suited when data are not IID between clients. Flower already comes with many partitioners (Dirichlet, Linear, Size, Pathological, etc.) however none of them permits to have a portion of client with IID data and another with non-IID, simulating a scenario where a group of nodes has good quality data and another not. To cope with this limitation I implemented a flower Partitioner called [`LabelBasedPartitioner`](clientcontributionfl/custom_partitioner.py).

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
The last function outputs the following result:
![partitioner_iid_non_iid](https://github.com/user-attachments/assets/2170f1e3-0b24-431d-bb7b-847acc67723b)


### Testing Setup
I tested on MNIST dataset with 10 clients for 10 rounds. The main metrics are the centralized accuracy and loss. One interesting case is when non-IID clients submit fake scores, under this setting is expected that `ZkAvg` achieves better accuracy than `ContAvg`.
To represent non-IID clients I used the `LabelBasedPartitioner` with `x=2` and `iid_ratio=0.7`

## Configuration ⚙️

The project configuration is managed through the `pyproject.toml` file. Key configurations include:

- Number of rounds: 10
- Fraction of clients for fitting and evaluation: 100%
- Number of clients per round: 1
- Dataset: MNIST
- Distribution: "iid_and_non_iid" # Is the partitioner type
- Batch size: 10
- SecAgg+ parameters: 3 shares, reconstruction threshold of 2
- Training parameters: learning rate 0.1, 2 epochs per round

You can modify these parameters in the `pyproject.toml` file under the `[tool.flwr.app.config]` section.

## Running simulation ▶️

To run the Federated Learning simulation:
   ```
   python main.py --strategies ZkAvg,ContAvg,FedAvg --num_rounds 10 --num_nodes 10
   ```
You can override other configurations parameters (learning rate, batch size etc.)directly changing `pyproject.toml`.

## Results 📊
The results of the training, including accuracy scores and any generated plots, will be saved in the `results/` directory.
