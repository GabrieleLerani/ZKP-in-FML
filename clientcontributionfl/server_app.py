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
        "alpha": config.get("alpha", 0.05),
        "x_non_iid": config.get("x_non_iid", 2),
        "iid_ratio": config.get("iid_ratio", 0.5),
        "dishonest": config.get("dishonest", 0.5)
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
    partitioner = params["distribution"]
    alpha = params["alpha"]
    x_non_iid = params["x_non_iid"]
    iid_ratio = params["iid_ratio"]
    dishonest = params["dishonest"]

    include_alpha = (f"_alpha={alpha}" if partitioner == "dirichlet" else "")
    include_x = (f"_x={x_non_iid}" if partitioner == "iid_and_non_iid" else "")
    include_iid_ratio = (f"_iid_ratio={iid_ratio}" if partitioner == "iid_and_non_iid" else "")
    include_dishonest = (f"_dishonest" if dishonest else "")
    include_sec_agg = ("SecAgg" if params['secaggplus'] else "")

    file_suffix = (
        f"_S={params['strategy_name']}"
        f"_R={params['num_rounds']}"
        f"_P={params['distribution']}"
        + include_sec_agg
        + include_alpha 
        + include_x 
        + include_iid_ratio 
        + include_dishonest
        
    )
    np.save(
        Path(params['save_path']) / Path("results") / Path(f"history{file_suffix}"), 
        history
    )