import torch
import os
from clientcontributionfl.models import train, test_random_batch
from clientcontributionfl.utils import compute_score, forge_score_in_proof, string_to_enum, SelectionPhase
from .zkavg_client import ZkClient
from .fedavg_client import FedAvgClient
from clientcontributionfl.utils import measure_cpu_and_time

class PoCZkClient(ZkClient):
    
    #@measure_cpu_and_time(csv_file="poc_avg_metric.csv")
    def fit(self, parameters, config):
        """Train the model using the client's data and return updated parameters.

        Args:
            parameters: The model parameters received from the server.
            config: Configuration dictionary containing training settings.

        Returns:
            Updated model parameters, the size of the training data, and additional parameters.
        """
        self.set_parameters(parameters)

        params = {}
    
        state = string_to_enum(SelectionPhase, config["state"])
        
        if state == SelectionPhase.SCORE_AGGREGATION : 
            # Compute ZKP score and generate proof in the first round
            score = compute_score(
                counts=self.partition_label_counts, 
                scale=self.scale, 
                beta=self.beta, 
                thr=self.thr
            )
            self.compute_zkp_contribution(score)
            self.forge_score()
            self.set_client_params(params)
            

        elif state == SelectionPhase.STORE_LOSSES:
            
            loss = test_random_batch(self.model, self.trainloader, self.config['device'])
            # include the loss in params
            params["loss"] = loss
            

        elif state == SelectionPhase.AGGREGATE_FROM_ACTIVE_SET: 
            optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
            
            loss, _ = train(
                self.model, 
                self.trainloader, 
                self.config['num_epochs'], 
                self.config['device'], 
                optimizer,
                self.criterion,
                self.accuracy_metric
            )


        return self.get_parameters({}), len(self.trainloader), params


class PoCClient(FedAvgClient):
    
    def fit(self, parameters, config):
        """Train the model using the client's data and return updated parameters.

        Args:
            parameters: The model parameters received from the server.
            config: Configuration dictionary containing training settings.

        Returns:
            Updated model parameters, the size of the training data, and additional parameters.
        """
        self.set_parameters(parameters)

        params = {}
    
        state = string_to_enum(SelectionPhase, config["state"])
            
        if state == SelectionPhase.STORE_LOSSES:
            
            loss = test_random_batch(self.model, self.trainloader, self.config['device'])
            # include the loss in params
            params["loss"] = loss
            
        elif state == SelectionPhase.AGGREGATE_FROM_ACTIVE_SET: 
            optimizer = torch.optim.SGD(self.model.parameters(), lr=config["lr"])
            
            loss, _ = train(
                self.model, 
                self.trainloader, 
                self.config['num_epochs'], 
                self.config['device'], 
                optimizer,
                self.criterion,
                self.accuracy_metric
            )


        return self.get_parameters({}), len(self.trainloader), params