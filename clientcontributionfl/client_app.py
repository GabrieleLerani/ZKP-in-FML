
from flwr.client import Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from flwr.common.config import get_project_config
from clientcontributionfl import load_data, compute_partition_counts
from clientcontributionfl.client_strategy import FedAvgClient, FedProxClient, ZkClient, ContributionClient, PoCZkClient, PoCClient, MerkleProofClient
from clientcontributionfl.utils import get_model_class
import clientcontributionfl.models as models
from clientcontributionfl import Zokrates

def create_client(strategy: str, **kwargs) -> Client:
    client_classes = {
        "FedAvg": FedAvgClient,
        "FedAvgM": FedAvgClient,
        "FedAdam": FedAvgClient,
        "FedProx": FedProxClient,
        "ZkAvg": ZkClient,
        "ContAvg": ContributionClient,
        "CLAvg": ContributionClient,
        "PoC": PoCClient,
        "PoCZk": PoCZkClient,
        "MPAvg": MerkleProofClient
    }
    
    client_class = client_classes.get(strategy)
    if client_class:
        return client_class(**kwargs).to_client()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def client_fn(context: Context) -> Client:
    
    # usally a random number instantiated by the server
    node_id = context.node_id

    # number from 0 up to num clients, corresponds to dataset partitions
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    config = context.run_config
    
    # load data with FederatedDataset
    train_loader, test_loader, num_classes = load_data(config, partition_id, num_partitions)

    model_class = get_model_class(models, config["dataset_name"])

    # Prepare common arguments for client instantiation
    common_args = {
        "node_id": str(node_id),
        "partition_id": partition_id,
        "trainloader": train_loader,
        "testloader": test_loader,
        "num_classes": num_classes,
        "config": config,
        "model_class": model_class,   
    }

    if config["strategy"] in ["FedAvg","FedAvgM", "FedAdam", "FedProx", "PoC"]:
        return create_client(config["strategy"], **common_args)

    elif config["strategy"] == "MPAvg":
        common_args.update({
            "smart_contract": config["smart_contract"],
            "zk_prover": Zokrates(working_dir=f"proofs/client_{node_id}")
        })

        return create_client(config["strategy"], **common_args)

    elif config["strategy"] in ["ZkAvg", "PoCZk"]:
        
        partition_counts = compute_partition_counts(
            data_loader=train_loader,
            partition_id=partition_id,
            num_classes=num_classes
        )
        iid_clients = num_partitions * config["iid_ratio"]
        dishonest = config["dishonest"]
        
        # Add specific arguments for ZkAvg and PoCZk
        common_args.update({
            "partition_label_counts": partition_counts,
            "dishonest": partition_id >= iid_clients if dishonest else False,
            "zk_program_file": f"../../{config["zok_contribution_file_path"]}/contribution.zok",
            "zk_prover": Zokrates(working_dir=f"proofs/client_{node_id}")
        })
        
        return create_client(config["strategy"], **common_args)
    
    elif config["strategy"] in ["ContAvg", "CLAvg"]:
        partition_counts = compute_partition_counts(
            data_loader=train_loader,
            partition_id=partition_id,
            num_classes=num_classes
        )
        iid_clients = num_partitions * config["iid_ratio"]
        dishonest = config["dishonest"]
        
        # Add specific arguments for ContAvg
        common_args.update({
            "partition_label_counts": partition_counts,
            "dishonest": partition_id >= iid_clients if dishonest else False,
        })
        
        return create_client(config["strategy"], **common_args)

    raise ValueError(f"Unknown strategy: {config['strategy']}")
    
    
    
# Load configuration
config = get_project_config(".")["tool"]["flwr"]["app"]["config"]

secaggplus = config.get("secaggplus", True)

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod] if secaggplus else None,
)