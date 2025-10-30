from lightning.pytorch.strategies import DDPStrategy
import lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from lightning.pytorch.loggers import WandbLogger

from Dataset import SudokuDataset
from model import SudokuLightning


def __main__():
    batch = 8
    epochs = 5

    devices = torch.cuda.device_count()
    pl.seed_everything(42)
    wandb_logger = WandbLogger(project="sudoku-testing", name="rb-b-512")


    dataset = SudokuDataset("\home\sam\sudoku\labeled_sudoku.csv")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.2 * len(dataset))

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    model = SudokuLightning()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision="16",
        strategy=DDPStrategy(),
        val_check_interval=0.2
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, dataloaders=test_loader)

