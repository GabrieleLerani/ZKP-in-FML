
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from typing import List, Tuple, Union, Optional, Dict
from logging import INFO, DEBUG, WARNING
from flwr.server.strategy.aggregate import weighted_loss_avg

def keep_top_k(d, k):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:k])

class ContributionFedAvg(FedAvg):
    """Federated Averaging strategy that selects clients based on their contribution scores.

    This strategy extends the FedAvg strategy by sampling the top-k clients
    with the highest contribution scores obtained during evaluation. The
    contribution scores are used to prioritize clients that provide more
    valuable updates to the global model.

    Parameters:
    -----------
    top_k : int, optional
        The number of top-performing clients to select for each round of training.
        Defaults to 1.
    **kwargs : dict
        Additional keyword arguments to be passed to the FedAvg constructor.
    """
    
    def __init__(self, top_k: int = 2,**kwargs):
        super().__init__( **kwargs)
        self.top_k = top_k
        self.contribution_metrics = {}

    

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Use the superclass method to get the initial client selection
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)

        
        # Only sort and filter clients if contribution_metrics is not empty
        # Useful for the first round when the server does not have contribution metrics
        if self.contribution_metrics:
            
            filtered_client_fit_ins = []
            for cfi in client_fit_ins:
                if cfi[0].cid in keep_top_k(self.contribution_metrics, self.top_k):
                    filtered_client_fit_ins.append(cfi)
            #log(INFO, f"filtered_client_fit_ins: {len(filtered_client_fit_ins)}")
            
            
        else:
            # If contribution_metrics is empty, use the original client selection
            filtered_client_fit_ins = client_fit_ins

        
        return filtered_client_fit_ins
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        # get metrics sent from clients in evaluate() function
        for _, metric in eval_metrics:
            for key, value in metric.items():
                self.contribution_metrics[key] = value
        log(INFO, f"contribution_metrics: {self.contribution_metrics}")
            

        
        # if self.evaluate_metrics_aggregation_fn:
        #     eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    


