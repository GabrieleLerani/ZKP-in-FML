import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from flwr.common.logger import log
from logging import INFO, DEBUG

class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()


        # define layers
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO Maybe move this into Model so to avoid passing parameters
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
            
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            accuracy_metric.update(outputs, labels)

        epoch_loss /= len(trainloader)
        epoch_acc = accuracy_metric.compute()
        
        #log(INFO, f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

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
            
            images, labels = batch["image"], batch["label"]

            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()
            accuracy_metric.update(outputs, labels)

    test_loss /= len(testloader)
    accuracy = accuracy_metric.compute()
    
    return test_loss, accuracy.item()
