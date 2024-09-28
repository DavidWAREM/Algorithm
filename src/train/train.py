# src/train/train.py
import torch
import numpy as np
import logging

def train(train_loader, model, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average training loss.
    """
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0
    try:
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Training Loss: {avg_loss:.4f}")
        return avg_loss
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def validate(val_loader, model, device, criterion):
    """
    Validates the model for one epoch.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): The model to validate.
        device (torch.device): Device to run the validation on.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average validation loss.
    """
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise

def test(test_loader, model, device):
    """
    Tests the model and collects predictions.

    Args:
        test_loader (DataLoader): DataLoader for test data.
        model (torch.nn.Module): The trained model.
        device (torch.device): Device to run the testing on.

    Returns:
        tuple: True labels and predicted logits.
    """
    logger = logging.getLogger(__name__)
    model.eval()
    y_true = []
    y_pred = []
    try:
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                y_true.append(batch.y.cpu().numpy())
                y_pred.append(out.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        logger.info("Testing completed successfully.")
        return y_true, y_pred
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise
