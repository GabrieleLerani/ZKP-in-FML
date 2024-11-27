
from flwr.client import Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from flwr.common.config import get_project_config

from clientcontributionfl import load_data, compute_partition_counts
from clientcontributionfl.client_strategy import FedAvgClient, ZkClient, ContributionClient, PowerOfChoiceClient
import clientcontributionfl.models as models
from clientcontributionfl.utils import get_model_class

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

    if config["strategy"] == "FedAvg":
        
        return FedAvgClient(
            node_id=str(node_id),
            partition_id=partition_id,
            trainloader=train_loader,
            testloader=test_loader,
            num_classes=num_classes,
            config=config,
            model_class=model_class
        ).to_client()

    elif config["strategy"] == "ZkAvg":
        
        partition_counts = compute_partition_counts(
            data_loader=train_loader,
            partition_id=partition_id,
            num_classes=num_classes
        )

        iid_clients = num_partitions * config["iid_ratio"]
        
        dishonest = config["dishonest"]

        return ZkClient(
            node_id=str(node_id),
            partition_id=partition_id,
            trainloader=train_loader,
            testloader=test_loader,
            partition_label_counts=partition_counts,
            num_classes=num_classes,
            dishonest=partition_id >= iid_clients if dishonest else False,
            config=config,
            model_class=model_class
        ).to_client()

    elif config["strategy"] == "ContAvg":
        
        partition_counts = compute_partition_counts(
            data_loader=train_loader,
            partition_id=partition_id,
            num_classes=num_classes
        )

        iid_clients = num_partitions * config["iid_ratio"]
        dishonest = config["dishonest"]

        return ContributionClient(
            node_id=str(node_id),
            partition_id=partition_id,
            trainloader=train_loader,
            testloader=test_loader,
            partition_label_counts=partition_counts,
            num_classes=num_classes,
            dishonest=partition_id >= iid_clients if dishonest else False, # set as many dishonest as non iid clients
            config=config,
            model_class=model_class
        ).to_client()
    
    elif config["strategy"] == "PoC":
        
        partition_counts = compute_partition_counts(
            data_loader=train_loader,
            partition_id=partition_id,
            num_classes=num_classes
        )

        iid_clients = num_partitions * config["iid_ratio"]
        dishonest = config["dishonest"]

        return PowerOfChoiceClient(
            node_id=str(node_id),
            partition_id=partition_id,
            trainloader=train_loader,
            testloader=test_loader,
            partition_label_counts=partition_counts,
            num_classes=num_classes,
            dishonest=partition_id >= iid_clients if dishonest else False,
            config=config,
            model_class=model_class
        ).to_client()
    
    
# Load configuration
config = get_project_config(".")["tool"]["flwr"]["app"]["config"]

secaggplus = config.get("secaggplus", True)

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod] if secaggplus else None,
)