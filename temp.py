import os
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


# Function to load data with additional node features
def load_data(node_file, edge_file):
    # Load node and edge data
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    # Create a mapping for node names to indices
    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    # One-hot encode the 'ROHRTYP' column in the edge data
    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')

    # Create the edge index tensor
    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

    # Use node features: Geographic features (XRECHTS, YHOCH, GEOH) + Physical properties (ZUFLUSS, PMESS, PRECH, DP, HP)
    geographic_features = ['XRECHTS', 'YHOCH', 'GEOH']
    physical_features = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    node_features = nodes_df[geographic_features + physical_features].values

    # Use edge attributes (RORL, DM, FLUSS, etc.)
    edge_attributes = edges_df[
        ['RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)].values

    # Standardize the node and edge features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    edge_attributes = scaler.fit_transform(edge_attributes)

    # Convert the data to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    y = torch.tensor(edges_df['RAU'].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


# GCN Model for edge regression
class EdgeGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=16, output_dim=1):
        super(EdgeGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        edge_features = self.edge_mlp(edge_attr)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_predictions = self.fc_edge(edge_embeddings).squeeze()

        return edge_predictions


# Training function
def train(loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Plot true vs predicted values
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('GCN - True vs Predicted')
    plt.show()


# Main function for the training process
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

    datasets = []
    for i in range(1, 1000):  # Load data for 10 networks
        node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
        data = load_data(node_file, edge_file)
        datasets.append(data)

    loader = DataLoader(datasets, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGCN(num_node_features=datasets[0].x.shape[1], num_edge_features=datasets[0].edge_attr.shape[1]).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train(loader, model, optimizer, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Training Loss: {loss:.4f}')

    # Get predictions for plotting
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(model(batch).cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Plot true vs predicted values
    plot_predictions(y_true, y_pred)

    model_path = os.path.join(directory, 'edge_gcn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at: {model_path}')


if __name__ == "__main__":
    main()
