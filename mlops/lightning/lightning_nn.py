from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

class LightningNeuralNet(LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.lr = lr
        self.criterion = torch.nn.NLLLoss(reduction="sum")
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images)
        loss = torch.nn.functional.nll_loss(output, targets)
        self.log("Train loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch
        output = self.model(images)
        loss = torch.nn.functional.nll_loss(output, targets)
        self.log("Validation loss", loss.item())

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)