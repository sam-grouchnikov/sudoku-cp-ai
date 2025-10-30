import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class SudokuCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_out(x)
        x = x.squeeze(1)  # output shape: (B, 9, 9)
        return x

class SudokuLightning(L.LightningModule):
    def __init__(self, logger=None, lr=1e-3):
        super().__init__()
        self.model = SudokuCNN()
        self.lr = lr
        self.wandb_logger = logger

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B,1,9,9), y: scalar 0..80
        logits = self(x).view(x.size(0), -1)  # flatten to (B, 81)
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
