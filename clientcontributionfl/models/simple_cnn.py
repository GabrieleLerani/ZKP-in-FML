import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common import Parameters

from logging import INFO, DEBUG
import random

class NetMnist(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(NetMnist, self).__init__()

        # define layers
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512) 
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NetCifar10(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(NetCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)




def train_proximal_term(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    proximal_mu: float,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    proximal_mu : float
        Parameter for the weight of the proximal term.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net, global_params, trainloader, device, criterion, optimizer, proximal_mu
        )

def _train_one_epoch(  # pylint: disable=too-many-arguments
    net: nn.Module,
    global_params: List[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    proximal_mu: float,
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    global_params : List[Parameter]
        The parameters of the global model (from the server).
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training
    proximal_mu : float
        Parameter for the weight of the proximal term.

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for batch in trainloader:
        feature, label = list(batch.keys()) # take the column name of image
        images, labels = batch[feature], batch[label]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return net


def train(
        net: nn.Module, 
        trainloader: DataLoader, 
        epochs: int, 
        device: str, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.CrossEntropyLoss,
        accuracy_metric: Accuracy
    ):
    """Train the network on the training set."""
    
    net.train()
    net.to(device)
    
    for _ in range(epochs):
        epoch_loss = 0.0
        accuracy_metric.reset()
        
        for batch in trainloader:
            feature, label = list(batch.keys()) # take the column name of image
            images, labels = batch[feature], batch[label]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            accuracy_metric.update(outputs, labels)
            del loss, outputs 

        epoch_loss /= len(trainloader)
        epoch_acc = accuracy_metric.compute()
        

    return epoch_loss, epoch_acc.item()


def test(net: nn.Module, testloader: DataLoader, device: str, accuracy_metric: Accuracy):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    
    
    net.eval()
    net.to(device)
    test_loss = 0.0
    accuracy_metric.reset()
    
    with torch.no_grad():
        for batch in testloader:
            feature, label = list(batch.keys()) # take the column name of image
            images, labels = batch[feature], batch[label]

            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()
            accuracy_metric.update(outputs, labels)

    test_loss /= len(testloader)
    accuracy = accuracy_metric.compute()
    
    return test_loss, accuracy.item()

def test_random_batch(net: nn.Module, testloader: DataLoader, device: str) -> float:
    """Evaluate the network on a single random batch from the test set. This is the 
        computation efficient variant of Power-Of-Choice, in the original version
        the loss is computed over all the test dataset.
    """

    criterion = torch.nn.CrossEntropyLoss()
    
    # set to evaluation mode
    net.eval()
    net.to(device)
    
    # select a batch
    batch = sample_random_batch(testloader)
    
    with torch.no_grad():
        feature, label = list(batch.keys()) # take the column name of image
        images, labels = batch[feature], batch[label]
        images, labels = images.to(device), labels.to(device)
        
        outputs = net(images)
        test_loss = criterion(outputs, labels).item() / len(batch) 
    
    
    return test_loss


def sample_random_batch(dataloader: DataLoader):
    num_batches = len(dataloader)

    # Randomly select a batch index
    random_batch_idx = random.randint(0, num_batches - 1)

    # Retrieve the randomly selected batch
    for idx, batch in enumerate(dataloader):
        if idx == random_batch_idx:
            sampled_batch = batch
            break
    
    return sampled_batch

def get_model_initial_parameters(model: nn.Module) -> Parameters:
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)
    return parameters