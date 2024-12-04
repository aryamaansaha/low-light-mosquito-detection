import time
import os
from tqdm import tqdm
import torch

def train(model, optimizer, scheduler, trainloader, valloader, device, N_epochs, save_dir, save_name="best_model.pth"):
    """
    Train and validate the model for N_epochs

    Args:
        model: SSD model to train.
        optimizer: Optimizer for the model.
        scheduler: Learning rate scheduler.
        trainloader: DataLoader for training data.
        valloader: DataLoader for validation data.
        device: Device to use for training (e.g., "cuda" or "cpu").
        N_epochs: Number of training epochs.
        save_dir: Directory to save model checkpoints.
        save_name: Name of the best model to save.
    """
    os.makedirs(save_dir, exist_ok=True)  
    best_val_loss = float("inf")  # Track the best validation loss
    train_losses, val_losses, lr_history = [], [], []

    for epoch in range(N_epochs):
        start_time = time.time()
        model.train()  # Set model to training mode

        # Training loop
        train_loss = 0
        for images, targets in tqdm(trainloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0

        with torch.no_grad():
            for images, targets in tqdm(valloader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                val_loss += losses.item()

        val_loss /= len(valloader)
        val_losses.append(val_loss)

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "ssd_best_model.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            print(f"Best model saved with val loss: {val_loss:.4f}")

        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        print(f"Epoch [{epoch + 1}/{N_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {int(mins)} min {int(secs)} s")

        lr_history.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    return train_losses, val_losses, lr_history
