import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc_edge = torch.nn.Linear(32, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_predictions = self.fc_edge(edge_embeddings).squeeze()
        return edge_predictions


class GCNTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_model(self, train_data, num_epochs=100, log_interval=10):
        self.model = self.model.to(self.device)
        for epoch in range(num_epochs):
            total_loss = 0
            for data, y in train_data:
                data = data.to(self.device)
                y = y.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = F.mse_loss(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % log_interval == 0:
                print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}')
