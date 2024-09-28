# src/models/model_setup.py
import torch
from src.models.gat import EdgeGAT
import logging

def initialize_model(num_node_features, num_edge_features, hidden_dim, dropout, device):
    """
    Initializes the EdgeGAT model.

    Args:
        num_node_features (int): Number of input node features.
        num_edge_features (int): Number of input edge features.
        hidden_dim (int): Dimension of hidden layers.
        dropout (float): Dropout rate.
        device (torch.device): Device to load the model on.

    Returns:
        EdgeGAT: Initialized model.
    """
    logger = logging.getLogger(__name__)
    try:
        model = EdgeGAT(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)
        logger.info("Model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

def initialize_optimizer(model, learning_rate, weight_decay):
    """
    Initializes the optimizer.

    Args:
        model (torch.nn.Module): The model to optimize.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    logger = logging.getLogger(__name__)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logger.info("Optimizer initialized successfully.")
        return optimizer
    except Exception as e:
        logger.error(f"Error initializing optimizer: {e}")
        raise
