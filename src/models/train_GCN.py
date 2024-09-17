import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, output_dim, hidden_dim=27):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Neural Network layers for edge features
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # Fully connected layer to combine node and edge information
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, output_dim)

    def forward(self, data, edge_attr):
        x, edge_index = data.x, data.edge_index

        # Node feature processing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Edge feature processing
        edge_features = self.edge_mlp(edge_attr)

        # Combine node and edge information
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_predictions = self.fc_edge(edge_embeddings).squeeze()

        return edge_predictions


class GCNTrainer:
    def __init__(self, model, optimizer=None, lr=9.63e-05, weight_decay=6.81e-05, device=None):
        self.model = model.to(device)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        self.device = device

    def train_model(self, train_data, num_epochs=100, log_interval=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for data, y, edge_attr in train_data:  # y corresponds to RAU, edge_attr has other edge features
                data = data.to(self.device)
                y = y.to(self.device)  # Target values (RAU)
                edge_attr = edge_attr.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()
                out = self.model(data, edge_attr)  # Pass edge_attr to the model
                loss = F.mse_loss(out, y)  # Compute MSE loss between predicted and actual RAU
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % log_interval == 0:
                print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}')
