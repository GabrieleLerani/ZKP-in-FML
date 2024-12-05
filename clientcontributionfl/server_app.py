import numpy as np
from pathlib import Path
from .dataset import load_centralized_dataset
from .server_utils import get_strategy
from logging import INFO
from flwr.common.logger import log
from flwr.common import Context, Metrics
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow
from clientcontributionfl.server_strategy import DefaultWorkflow, PoCWorkflow
from pprint import PrettyPrinter
from clientcontributionfl.utils import generate_file_suffix

app = ServerApp()

@app.main()
def main(driver: Driver, context: Context):
    # Step 1: Load and print configuration
    config = context.run_config
    print_config(config)
    
    # Step 2: Extract run parameters
    run_params = extract_run_params(config)
    
    # Step 3: Load dataset and get strategy
    dataset_loader, num_classes = load_centralized_dataset(config)
    strategy = get_strategy(config, num_classes, dataset_loader)
    
    # Step 4: Create legacy context and workflow
    legacy_context = create_legacy_context(context, run_params['num_rounds'], strategy)
    workflow = create_workflow(run_params)
    
    # Step 5: Execute workflow
    workflow(driver, legacy_context)
    
    # Step 6: Save history
    save_history(legacy_context.history, run_params)

def print_config(config):
    print("Run configuration:")
    PrettyPrinter(indent=4).pprint(config)

def extract_run_params(config):
    return {
        "num_rounds": config["num_rounds"],
        "partitioner": config["partitioner"],
        "save_path": config['save_path'],
        "num_shares": config["num_shares"],
        "reconstruction_threshold": config["reconstruction_threshold"],
        "max_weight": config["max_weight"],
        "strategy_name": config['strategy'],
        "secaggplus": config.get("secaggplus", False),
        "alpha": config.get("alpha", 0.05),
        "x_non_iid": config.get("x_non_iid", 2),
        "iid_ratio": config.get("iid_ratio", 0.5),
        "iid_data_fraction": config.get("iid_data_fraction", 0.5),
        "dishonest": config.get("dishonest", False),
        "balanced": config.get("balanced", False),
        "dataset_name": config.get("dataset_name", ""),
    }

def create_legacy_context(context, num_rounds, strategy):
    return LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

def create_workflow(params):
    fit_workflow = SecAggPlusWorkflow(
        num_shares=params['num_shares'],
        reconstruction_threshold=params['reconstruction_threshold'],
        max_weight=params['max_weight'],
    ) if params['secaggplus'] else None
    
    if "PoC" in params['strategy_name']:
        workflow = PoCWorkflow(fit_workflow=fit_workflow)
    else:
        workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    return workflow

def save_history(history, params):

    strategy = params["strategy_name"]
    dataset_name = params["dataset_name"]
    honest_or_dishonest_client = "dishonest" if params["dishonest"] else "honest"

    file_suffix = generate_file_suffix(params)
    
    simulation_path = Path(params['save_path']) / Path("simulation") / Path(dataset_name) / Path(honest_or_dishonest_client)

    save_dir = simulation_path / Path(file_suffix.lstrip('_'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the history in the new directory
    np.save(
        save_dir / Path(f"history_S={strategy}.npy"),
        history
    )
