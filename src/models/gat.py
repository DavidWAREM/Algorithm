# src/models/gat.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EdgeGAT(torch.nn.Module):
    """
    Edge-based Graph Attention Network (GAT) for binary classification.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_dim, dropout):
        """
        Initializes the EdgeGAT model.

        Args:
            num_node_features (int): Number of input node features.
            num_edge_features (int): Number of input edge features.
            hidden_dim (int): Dimension of hidden layers.
            dropout (float): Dropout rate.
        """
        super(EdgeGAT, self).__init__()
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            edge_dim=num_edge_features,
            heads=1,
            dropout=dropout
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=num_edge_features,
            heads=1,
            dropout=dropout
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)  # For binary classification

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch.Tensor: Output logits.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.fc(x)
        return x
