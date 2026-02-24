import wandb
import time
import random

wandb.init(project="logging-simulation", name="test-run-1")

total_epochs = 10
batches_per_epoch = 5


for epoch in range(total_epochs):
    for batch in range(batches_per_epoch):
        time.sleep(0.5)

        loss = 1.0 / (epoch + 1) + (random.random() * 0.1)
        acc = 1.0 - (1.0 / (epoch + 2)) + (random.random() * 0.05)

        wandb.log({
            "epoch": epoch,
            "batch": batch,
            "loss": loss,
            "accuracy": acc,
            "learning_rate": 0.001 * (0.9 ** epoch)
        })

print("Simulation complete.")
wandb.finish()