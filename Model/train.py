from lightning.pytorch.strategies import DDPStrategy
import lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


from Model.Dataset import SudokuDataset
from Model.model_cnn import SudokuLightningCNN
from Model.model_normal import SudokuLightningFC


def main():
    batch = 128
    epochs = 7

    devices = torch.cuda.device_count()
    pl.seed_everything(42)
    wandb_logger = WandbLogger(project="sudoku-cnn-comps", name="cnn-8")


    dataset = SudokuDataset("/home/sam/sudoku/row_data.csv")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size-val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    model = SudokuLightningCNN(wandb_logger)



    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision="16",
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=100,
        logger=wandb_logger,

    )


    trainer.fit(model, train_loader, val_loader)



    trainer.test(model, dataloaders=test_loader)
    trainer.save_checkpoint("latest.ckpt")


if __name__ == "__main__":
    main()