
from flwr.client import Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from flwr.common.config import get_project_config

from clientcontributionfl import load_data, compute_partition_counts
from clientcontributionfl.client_strategy import FedAvgClient, ZkClient


def client_fn(context: Context) -> Client:
        
    # usally a random number instantiated by the server
    node_id = context.node_id

    # number from 0 up to num clients, corresponds to dataset partitions
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    config = context.run_config
    
    # load data with FederatedDataset
    train_loader, test_loader, num_classes = load_data(config, partition_id, num_partitions)

    if config["strategy"] == "FedAvg":
        
        return FedAvgClient(
            node_id=str(node_id),
            partition_id=partition_id,
            trainloader=train_loader,
            testloader=test_loader,
            num_classes=num_classes,
            trainer_config=config
        ).to_client()

    elif config["strategy"] == "ZkAvg":
        
        partition_counts = compute_partition_counts(
            data_loader=train_loader,
            partition_id=partition_id,
            num_classes=num_classes
        )

        return ZkClient(
            node_id=str(node_id),
            partition_id=partition_id,
            trainloader=train_loader,
            testloader=test_loader,
            partition_label_counts = partition_counts, # TODO test it for zk-proof
            num_classes=num_classes,
            trainer_config=config
        ).to_client()

    
    
    


# Load configuration
config = get_project_config(".")["tool"]["flwr"]["app"]["config"]

secaggplus = config.get("secaggplus", True)

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod] if secaggplus else None,
)