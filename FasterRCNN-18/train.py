import os
import torch
from tqdm import tqdm
import time

def train(model, optimizer, scheduler, trainloader, valloader, device, N_epochs, save_dir, save_name):
    """
    Train and validate the model for N_epochs
    
    Args:
    - model: The Faster R-CNN model.
    - optimizer: Optimizer for training.
    - scheduler: Learning rate scheduler.
    - trainloader: DataLoader for the training dataset.
    - valloader: DataLoader for the validation dataset.
    - device: Device to use ('cuda' or 'cpu').
    - N_epochs: Number of epochs to train.
    - save_dir: Directory to save the models.
    - save_name: Name of the best model to save.
    """
    best_val_loss = float("inf")
    train_losses, val_losses, lr_history = [], [], []
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    for epoch in range(N_epochs):
        start_time = time.time()  # Start timing the epoch
        
        # Training Loop
        model.train()
        train_loss = 0.0

        for images, targets in tqdm(trainloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and loss computation
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimizer step
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        # Validation Loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in valloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Compute loss (requires switching back to train mode temporarily)
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()  # Switch back to evaluation mode

                val_loss += losses.item()

        val_loss /= len(valloader)
        val_losses.append(val_loss)

        # Calculate epoch time
        epoch_time = time.time() - start_time
        minutes, seconds = divmod(epoch_time, 60)

        print(f"Epoch [{epoch+1}/{N_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {int(minutes)} min {int(seconds)} s")

        # Save the model for the current epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            print(f"Best model saved with val loss: {val_loss:.4f}")

        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    return train_losses, val_losses, lr_history
