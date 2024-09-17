

from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerAppComponents
from typing import Dict

# def get_evaluate_fn(num_classes: int, testloader):
#     """Define function for global evaluation on the server."""

#     def evaluate_fn(server_round: int, parameters, config):
#         # This function is called by the strategy's `evaluate()` method
#         # and receives as input arguments the current round number and the
#         # parameters of the global model.
#         # this function takes these parameters and evaluates the global model
#         # on a evaluation / test dataset.

#         model = Net(num_classes)

#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)

#         # Here we evaluate the global model on the test set. Recall that in more
#         # realistic settings you'd only do this at the end of your FL experiment
#         # you can use the `server_round` input argument to determine if this is the
#         # last round. If it's not, then preferably use a global validation set.
#         loss, accuracy = test(model, testloader, device)

#         # Report the loss and any other metric (inside a dictionary). In this case
#         # we report the global test accuracy.
#         return loss, {"accuracy": accuracy}

#     return evaluate_fn


def generate_server_fn(cfg):

    def server_fn(context: Context) -> ServerAppComponents:
        # Create the FedAvg strategy
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            min_available_clients=cfg.num_clients,
        )
        # Configure the server for 3 rounds of training
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn