import hydra
from omegaconf import OmegaConf
from dataset import DatasetLoader
from client import generate_client_fn
from server import generate_server_fn
from flwr.server import ServerApp
from flwr.client import ClientApp
from flwr.simulation import run_simulation
import flwr as fl
from logging import INFO, DEBUG
from flwr.common.logger import log


@hydra.main(config_path="config", config_name="base.yaml", version_base=None)
def main(
    cfg: OmegaConf,
):
    # print configuration that will be sent to clients
    print(OmegaConf.to_yaml(cfg))


    # 1. Get data loaders for each client
    dataset_loader = DatasetLoader(
        cfg.num_clients,
        cfg.dataset.num_classes_per_partition,
        cfg.dataset.distribution,
        cfg.dataset.plot_label_distribution,
        cfg.dataset.alpha,
        cfg.dataset.batch_size
    )
    
    train_loaders, val_loaders, test_loaders = dataset_loader.load_data()

    # 2. compute scores of dataset for each client
    scores = dataset_loader.compute_partition_score()    
    log(INFO, f"scores: {scores}")

    client_fn = generate_client_fn(
        train_loaders,
        val_loaders,
        test_loaders,
        scores,
        dataset_loader.num_classes,
        cfg['trainer']
    )

    server_fn = generate_server_fn(
        cfg, 
        dataset_loader.num_classes, 
        dataset_loader.centralized_test_loader, 
        cfg['trainer']
    )

    # 3. generate client function
    client_app = ClientApp(client_fn=client_fn)

    # 4. generate server function
    server_app = ServerApp(server_fn=server_fn)

    # TODO change to start_simulation so you can collect history and plot metrics

    # 5. run simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        verbose_logging=False
    )
    

if __name__ == "__main__":
    main()