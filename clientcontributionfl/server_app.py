import numpy as np
from pathlib import Path
from .dataset import load_centralized_dataset
from .server_utils import get_strategy
from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr.common import Context, Metrics
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from .utils import plot_util as plt_util


app = ServerApp()

@app.main()
def main(driver: Driver, context: Context):

    # TODO pretty print configuration that will be sent to clients
    # print(context.run_config)
    

    # Store all accessed run_config values
    #num_clients = context.run_config["num_clients"]
    num_rounds = context.run_config["num_rounds"]
    distribution = context.run_config["distribution"]
    save_path = context.run_config['save_path']
    num_shares = context.run_config["num_shares"]
    reconstruction_threshold = context.run_config["reconstruction_threshold"]
    max_weight = context.run_config["max_weight"]
    strategy_name = context.run_config['strategy']
    


    dataset_loader, num_classes = load_centralized_dataset(context.run_config)

    # 2. build strategy
    strategy = get_strategy(
        context.run_config,
        num_classes,
        dataset_loader,
    )
    
    # 3. build LegacyContext, useful for SecAgg+
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    fit_workflow = SecAggPlusWorkflow(
        num_shares=num_shares,
        reconstruction_threshold=reconstruction_threshold,
        max_weight=max_weight,
    )


    workflow = DefaultWorkflow(
        fit_workflow=fit_workflow,
    )

    workflow(driver, context)


    history = context.history
    

    
    file_suffix: str = (
        f"_S={strategy_name}"
        #f"_C={num_clients}"
        f"_R={num_rounds}"
        f"_D={distribution}"
    )

    # save history
    np.save(
        Path(save_path) / Path(f"history{file_suffix}"), 
        history
    )

    # TODO it gives an error saying that in main thread you cannot plot
    # plot results
    # plt_util.plot_metric_from_history(
    #     history,
    #     save_path,
    #     strategy_name,
    #     file_suffix,
    # )


    # plt_util.plot_comparison_from_files(
    #     save_path,
    #     num_clients,
    #     num_rounds,
    #     distribution
    # )

