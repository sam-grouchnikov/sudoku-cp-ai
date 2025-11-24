import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class SudokuFC(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2430, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 9 * 9)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 9, 9)
        return x

class SudokuLightningFC(L.LightningModule):
    def __init__(self, logger=None, lr=1e-3):
        super().__init__()
        self.model = SudokuFC()
        self.lr = lr
        self.wandb_logger = logger

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(x.size(0), -1)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(x.size(0), -1)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(x.size(0), -1)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()


        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
