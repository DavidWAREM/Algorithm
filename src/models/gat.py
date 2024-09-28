import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=16, dropout=0.2):
        super(EdgeGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 4)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, 1)  # Ausgabe ist ein Logit
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)

        edge_features = self.edge_mlp(edge_attr)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_logits = self.fc_edge(edge_embeddings).squeeze()

        return edge_logits  # Logits zurückgeben
