import hydra
from omegaconf import OmegaConf
from dataset import DatasetLoader
from client import generate_client_fn
import flwr as fl
from server import generate_server_fn
from flwr.server import ServerApp
from flwr.client import ClientApp
from flwr.simulation import run_simulation

@hydra.main(config_path="config", config_name="base.yaml", version_base=None)
def main(
    cfg: OmegaConf,
):
    # print configuration that will be sent to clients
    print(OmegaConf.to_yaml(cfg))


    # 1. Get data loaders for each client
    dataset_loader = DatasetLoader(
        cfg.num_clients,
        cfg.dataset.distribution,
        cfg.dataset.plot_label_distribution,
        cfg.dataset.alpha,
        cfg.dataset.batch_size
    )
    
    train_loaders, val_loaders, test_loaders = dataset_loader.load_data()

    # 2. compute scores of dataset for each client
    scores = dataset_loader.compute_partition_score()    

    # 3. generate client function to pass to the server
    client_fn = generate_client_fn(
        train_loaders,
        val_loaders,
        test_loaders,
        scores,
        dataset_loader.num_classes,
        cfg['trainer']
    )

    # 4. train model
    server_app = ServerApp(server_fn=generate_server_fn(cfg))
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,
    #     min_fit_clients=cfg.num_clients,
    #     min_evaluate_clients=cfg.num_clients,
    #     min_available_clients=cfg.num_clients,
    #     # evaluate_metrics_aggregation_fn=None, # TODO use the metrics to decide the weight of the client
    #     # on_fit_config_fn=None, # TODO use the config to decide the config to send to the client
    #     # evaluate_fn=None # TODO define the evaluation function on the server
    # )


    client_app = ClientApp(client_fn=client_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        verbose_logging=True
    )
    

    # 5. start simulation
    # history = fl.simulation.start_simulation(
    #     client_fn=client_fn,
    #     num_clients=cfg.num_clients,
    #     config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
    #     strategy=strategy
    # )   

    # 6. save the simulation history



if __name__ == "__main__":
    main()