import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Funktion zum Laden der Daten mit One-Hot-Encoding für 'ROHRTYP'
def load_data(node_file, edge_file, physical_scaler, geo_scaler):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
    node_features = nodes_df.drop(columns=['KNAM']).values

    # Skalierung der physikalischen und geografischen Daten
    physical_columns = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    nodes_df[physical_columns] = physical_scaler.transform(nodes_df[physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])

    node_features = nodes_df.drop(columns=['KNAM']).values

    edge_attributes = edges_df[
        ['RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)].values

    edge_attributes = edge_attributes.astype(float)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    y = torch.tensor(edges_df['RAU'].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


# GAT-Modell für Kantenregression
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


# Trainingsfunktion
def train(loader, model, optimizer, device, early_stopping=None):
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

    # Early stopping if provided
    if early_stopping:
        early_stopping(total_loss)

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


# Hauptfunktion für den Trainingsprozess
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

    datasets = []

    # Physikalische und geografische Spalten
    physical_columns = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Lade die erste Datei, um die Skaler zu fitten
    node_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_Node.csv'
    edge_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_Pipes.csv'

    nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')

    # Fitte die StandardScaler für physikalische und geografische Daten
    physical_scaler = StandardScaler()
    geo_scaler = MinMaxScaler()

    physical_scaler.fit(nodes_df_first[physical_columns])
    geo_scaler.fit(nodes_df_first[geo_columns])

    # Lade alle Datensätze mit den Skalierten Werten
    for i in range(1, 10000):
        node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
        data = load_data(node_file, edge_file, physical_scaler, geo_scaler)
        datasets.append(data)

    loader = DataLoader(datasets, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGAT(num_node_features=datasets[0].x.shape[1], num_edge_features=datasets[0].edge_attr.shape[1],
                    hidden_dim=78, dropout=0.205).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0088, weight_decay=2.9e-05)

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
    print(f'Modell wurde gespeichert unter: {model_path}')


if __name__ == "__main__":
    main()
