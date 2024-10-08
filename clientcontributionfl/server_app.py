import numpy as np
from pathlib import Path
from .dataset import load_centralized_dataset
from .server_utils import get_strategy
from flwr.common.logger import log
from flwr.common import Context, Metrics
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from pprint import PrettyPrinter
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
        "distribution": config["distribution"],
        "save_path": config['save_path'],
        "num_shares": config["num_shares"],
        "reconstruction_threshold": config["reconstruction_threshold"],
        "max_weight": config["max_weight"],
        "strategy_name": config['strategy'],
        "secaggplus": config.get("secaggplus", True),
        "alpha": config.get("alpha", 0.05)
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
    
    return DefaultWorkflow(fit_workflow=fit_workflow)

def save_history(history, params):
    file_suffix = (
        f"_S={params['strategy_name']}"
        f"_R={params['num_rounds']}"
        f"_D={params['distribution']}"
        f"_SecAgg={'On' if params['secaggplus'] else 'Off'}"
        f"_alpha={params['alpha']}"
    )
    np.save(
        Path(params['save_path']) / Path("results") / Path(f"history{file_suffix}"), 
        history
    )