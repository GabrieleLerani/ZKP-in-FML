import hydra
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from dataset import DatasetLoader
from client import generate_client_fn
from server import get_strategy, get_server_config
from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr.simulation.app import start_simulation
from flwr.server.history import History
import utils.plot_util as plt_util

@hydra.main(config_path="config", config_name="base.yaml", version_base=None)
def main(
    cfg: OmegaConf,
):
    # print configuration that will be sent to clients
    # print(OmegaConf.to_yaml(cfg))


    # # 1. Get data loaders for each client
    # dataset_loader = DatasetLoader(
    #     cfg.num_clients,
    #     cfg.dataset.num_classes_per_partition,
    #     cfg.dataset.distribution,
    #     cfg.dataset.plot_label_distribution,
    #     cfg.dataset.alpha,
    #     cfg.dataset.batch_size
    # )
    
    # train_loaders, val_loaders, test_loaders = dataset_loader.load_data()

    # # 2. compute scores of dataset for each client
    # scores = dataset_loader.compute_partition_score()    
    # log(INFO, f"scores: {scores}")
    

    # # 3. generate client function
    # client_fn = generate_client_fn(
    #     train_loaders,
    #     val_loaders,
    #     test_loaders,
    #     scores,
    #     dataset_loader.num_classes,
    #     cfg['trainer']
    # )


    # # 4 build strategy
    # strategy = get_strategy(
    #     cfg,
    #     dataset_loader.num_classes,
    #     dataset_loader.centralized_test_loader,
    # )
    
    # # 5. run simulation
    # history = start_simulation(
    #     client_fn=client_fn,
    #     num_clients=cfg.num_clients,
    #     config=get_server_config(cfg),
    #     strategy=strategy,
    #     client_resources = {
    #         "num_cpus": 6,
    #         "num_gpus": 0.0
    #     }
    # )

    

    # file_suffix: str = (
    #     f"_S={cfg.strategy}"
    #     f"_C={cfg.num_clients}"
    #     f"_R={cfg.num_rounds}"
    #     f"_D={cfg.dataset.distribution}"
    # )

    # # save history
    # np.save(
    #     Path(cfg.save_path) / Path(f"history{file_suffix}"), 
    #     history
    # )


    # # plot results
    # plt_util.plot_metric_from_history(
    #     history,
    #     cfg.save_path,
    #     cfg.strategy,
    #     file_suffix,
    # )


    plt_util.plot_comparison_from_files(
        cfg.save_path,
        cfg.num_clients,
        cfg.num_rounds,
        cfg.dataset.distribution
    )

if __name__ == "__main__":
    main()