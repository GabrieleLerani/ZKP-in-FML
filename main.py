import hydra
from omegaconf import OmegaConf
from dataset import DatasetLoader

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
    

    scores = dataset_loader.compute_partition_score()    
    #train_loaders, val_loaders, test_loaders = dataset_loader.load_data()

    

if __name__ == "__main__":
    main()