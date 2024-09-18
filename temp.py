import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import pi


# Function to add positional encodings using sine and cosine for geographic coordinates
def add_positional_encoding(df, columns, max_value=10000):
    for col in columns:
        df[f'{col}_sin'] = np.sin(df[col] * (2 * pi / max_value))
        df[f'{col}_cos'] = np.cos(df[col] * (2 * pi / max_value))
    return df


# Function to load data with One-Hot-Encoding for 'ROHRTYP' and scaling of node/edge features
def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
    node_features = nodes_df.drop(columns=['KNAM']).values

    # Scale physical and geographical data
    physical_columns = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Apply scaling
    nodes_df[physical_columns] = physical_scaler.transform(nodes_df[physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])

    # Add positional encoding for geographic columns
    nodes_df = add_positional_encoding(nodes_df, geo_columns)

    node_features = nodes_df.drop(columns=['KNAM']).values

    # Scale edge attributes
    edge_columns = ['RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)
    edges_df[edge_columns] = edge_scaler.transform(edges_df[edge_columns])

    edge_attributes = edges_df[edge_columns].values

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    y = torch.tensor(edges_df['RAU'].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


# GAT model with edge regression
class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=16, output_dim=1, dropout=0.2):
        super(EdgeGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
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


# Test function
def test(loader, model, device):
    model.eval()
    total_mse = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            total_mse += F.mse_loss(out, batch.y).item()
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(out.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return total_mse / len(loader), y_true, y_pred


# Plot true vs predicted values
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('GAT - True vs Predicted')
    plt.show()


# Main function with extended training and early stopping
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

    datasets = []

    # Physical and geographic columns
    physical_columns = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Load the first dataset for scaling
    node_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_Node.csv'
    edge_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_Pipes.csv'

    nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
    edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')

    # Apply one-hot encoding to 'ROHRTYP' column
    edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP')

    # Fit scalers for physical and geographical data
    physical_scaler = StandardScaler()
    geo_scaler = MinMaxScaler()
    edge_scaler = StandardScaler()

    physical_scaler.fit(nodes_df_first[physical_columns])
    geo_scaler.fit(nodes_df_first[geo_columns])

    # Update edge_columns to include 'ROHRTYP' dummy variables
    edge_columns = ['RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE'] + list(edges_df_first.filter(like='ROHRTYP').columns)

    edge_scaler.fit(edges_df_first[edge_columns])

    # Load all datasets with scaling and positional encoding
    for i in range(1, 11):  # Limiting to 10 datasets for faster execution
        node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
        data = load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler)
        datasets.append(data)

    loader = DataLoader(datasets, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGAT(num_node_features=datasets[0].x.shape[1], num_edge_features=datasets[0].edge_attr.shape[1],
                    hidden_dim=78, dropout=0.205).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0088, weight_decay=2.9e-05)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = 300  # Extended training
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training step
        loss = train(loader, model, optimizer, device)

        # Check learning rate schedule
        scheduler.step(loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Training Loss: {loss:.4f}')

        # Check early stopping condition
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0  # Reset patience counter if loss improves
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Get predictions and test metrics
    test_mse, y_true, y_pred = test(loader, model, device)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_true, y_pred)

    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'R² Score: {test_r2:.4f}')

    # Plot true vs predicted values
    plot_predictions(y_true, y_pred)

    model_path = os.path.join(directory, 'edge_gat_model_positional_encoding.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at: {model_path}')


if __name__ == "__main__":
    main()
