import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Function to load data with One-Hot Encoding for 'ROHRTYP'
def load_data(node_file, edge_file):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    # One-Hot Encoding on 'ROHRTYP'
    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
    node_features = nodes_df.drop(columns=['KNAM']).values

    # Concatenating new physical properties
    node_features = nodes_df[['ZUFLUSS', 'PMESS', 'GEOH', 'PRECH', 'DP', 'XRECHTS', 'YHOCH', 'HP']].values

    # Standardize physical properties
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    edge_attributes = edges_df[[
                                   'RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE'] +
                               list(edges_df.filter(like='ROHRTYP').columns)
                               ].values

    edge_attributes = edge_attributes.astype(float)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    y = torch.tensor(edges_df['RAU'].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


# GAT Model for Edge Prediction
class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=78, output_dim=1, dropout=0.205):
        super(EdgeGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, dropout=dropout)
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
    plt.title('GAT - True vs Predicted')
    plt.show()


# Main function with hyperparameter tuning, learning rate scheduler, and early stopping
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

    datasets = []

    for i in range(1, 10000):
        node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
        data = load_data(node_file, edge_file)
        datasets.append(data)

    loader = DataLoader(datasets, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGAT(num_node_features=datasets[0].x.shape[1], num_edge_features=datasets[0].edge_attr.shape[1]).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0088, weight_decay=2.9e-5)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    num_epochs = 100
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        loss = train(loader, model, optimizer, device)
        print(f'Epoch {epoch:03d}, Training Loss: {loss:.4f}')

        # Early stopping based on best validation loss
        scheduler.step(loss)
        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1

        if early_stop_counter > 10:
            print("Early stopping due to no improvement")
            break

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
    print(f'Model saved at: {model_path}')


if __name__ == "__main__":
    main()
