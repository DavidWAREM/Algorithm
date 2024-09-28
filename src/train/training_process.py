# src/train/training_process.py
import torch
from src.train.train import train, validate


def train_model(train_loader, val_loader, model, optimizer, criterion, scheduler, device, num_epochs, patience, logger):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Trainingsschritt
        train_loss = train(train_loader, model, optimizer, device, criterion)
        # Validierungsschritt
        val_loss = validate(val_loader, model, device, criterion)

        # Lernrate anpassen
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch:03d}, Trainingsverlust: {train_loss:.4f}, Validierungsverlust: {val_loss:.4f}')

        # Early Stopping basierend auf Validierungsverlust
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Geduldszähler zurücksetzen, wenn sich der Verlust verbessert
            # Bestes Modell speichern
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early Stopping bei Epoch {epoch}.")
            break

    # Laden des besten Modells
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
