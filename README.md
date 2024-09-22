# ZKP-in-FML: Zero Knowledge Proof for Client Contribution in Federated Machine Learning

This project is part of a research thesis on Zero Knowledge Proof to prove client contribution in Federated Machine Learning. It implements a Flower-based Federated Learning system using FedAvg and SecAgg+ protocol.

## Project Overview

This project, named "clientcontributionfl", is a Federated Learning implementation that focuses on:

1. Using the Flower framework for Federated Learning
2. Implementing FedAvg (Federated Averaging) strategy
3. Incorporating SecAgg+ protocol for secure aggregation
4. Using the MNIST dataset for training
5. Implementing a Dirichlet distribution for non-IID data partitioning

## Features

- Federated Learning with custom strategy strategy
- Secure Aggregation using SecAgg+ protocol
- Zero knowledge proofs

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
   flwr run .
   ```
You can change the running configuration and the number of clients and round. Check the `pyproject.toml` for other settings:
```
flower-simulation --app . --num-supernodes 5 --run-config "num_rounds=10"
```
