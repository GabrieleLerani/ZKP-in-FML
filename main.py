import hydra
from omegaconf import OmegaConf
from dataset import DatasetLoader
from utils import read_scores, plot_scores, DISTRIBUTIONS
from tqdm import tqdm

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
    
    # 2. compute scores of dataset
    scores = dataset_loader.compute_partition_score()    

    # 3. TODO set clients and strategies


    #train_loaders, val_loaders, test_loaders = dataset_loader.load_data()

    if cfg.dataset.plot_scores:
        plot_all(cfg)


def plot_all(cfg):
    print("Plotting scores...")
    
    # for d in tqdm(DISTRIBUTIONS, desc="Processing distributions"):
        
    #     print(cfg)
    #     dataset_loader = DatasetLoader(
    #         cfg.num_clients,
    #         d,
    #         cfg.dataset.plot_label_distribution,
    #         cfg.dataset.alpha,
    #         cfg.dataset.batch_size
    #     )

    #     dataset_loader.compute_partition_score()
        
    file_scores = read_scores()
    plot_scores(file_scores)
    

if __name__ == "__main__":
    main()