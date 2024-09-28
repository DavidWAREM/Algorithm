import torch
from src.models.gat import EdgeGAT

def initialize_model(num_node_features, num_edge_features, hidden_dim, dropout, device):
    model = EdgeGAT(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    return model

def initialize_optimizer(model, learning_rate, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer
