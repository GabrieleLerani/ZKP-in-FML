import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict
from logging import INFO, DEBUG
from flwr.common.logger import log

class Net(pl.LightningModule):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int, trainer_config: Dict[str, any]) -> None:
        super(Net, self).__init__()

        self.trainer_config = trainer_config
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)


        # define layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.trainer_config['lr'])

    def _shared_eval_step(self, batch, batch_idx, stage):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_hat, y)
        metrics = {f"{stage}_loss": loss, f"{stage}_accuracy": accuracy}
        self.log_dict(metrics)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        # TODO check it
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        return y_hat

    def _common_step(self, batch, batch_idx):    
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss, y_hat, y

    
