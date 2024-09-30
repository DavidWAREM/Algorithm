# src/train/training_process.py
import logging
from src.train.train import train, validate


class Trainer:
    """
    Trainer class manages the training loop, including early stopping.
    """

    def __init__(self, model, optimizer, criterion, scheduler, device, num_epochs, patience, logger):
        """
        Initializes the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (torch.nn.Module): Loss function.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            device (torch.device): Device to run the training on.
            num_epochs (int): Maximum number of epochs.
            patience (int): Number of epochs to wait for improvement before stopping.
            logger (logging.Logger): Logger instance.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.logger = logger

    def train_model(self, train_loader, val_loader):
        """
        Executes the training process with early stopping.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            torch.nn.Module: The best trained model.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.num_epochs}")
            try:
                train_loss = train(train_loader, self.model, self.optimizer, self.device, self.criterion)
                val_loss = validate(val_loader, self.model, self.device, self.criterion)

                # Scheduler step
                self.scheduler.step(val_loss)
                self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                    self.logger.info(f"Validation loss improved to {best_val_loss:.4f}.")
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement in validation loss for {patience_counter} epoch(s).")

                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

            except Exception as e:
                self.logger.error(f"An error occurred during epoch {epoch}: {e}")
                break

        # Load the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info("Loaded the best model state based on validation loss.")

        return self.model
