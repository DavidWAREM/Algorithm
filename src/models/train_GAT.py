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
from sklearn.model_selection import train_test_split


# Funktion zur Hinzufügung von Positionscodierung mittels Sinus und Kosinus für geografische Koordinaten
def add_positional_encoding(df, columns, max_value=10000):
    for col in columns:
        df[f'{col}_sin'] = np.sin(df[col] * (2 * pi / max_value))
        df[f'{col}_cos'] = np.cos(df[col] * (2 * pi / max_value))
    return df


# Funktion zum Laden der Daten mit One-Hot-Encoding für 'ROHRTYP' und Skalierung der Knoten-/Kantenfeatures
def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

    # Skalierung der physikalischen und geografischen Daten
    physical_columns = ['ZUFLUSS_WOL', 'ZUFLUSS_WL', 'PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Anwenden der Skalierung
    nodes_df[physical_columns] = physical_scaler.transform(nodes_df[physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])

    # Positionscodierung für geografische Spalten hinzufügen
    nodes_df = add_positional_encoding(nodes_df, geo_columns)

    node_features = nodes_df.drop(columns=['KNAM']).values

    # Skalierung der Kantenattribute
    edge_columns = ['RORL', 'DM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)
    edges_df[edge_columns] = edge_scaler.transform(edges_df[edge_columns])

    edge_attributes = edges_df[edge_columns].values

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    y = torch.tensor(edges_df['RAU'].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


# GAT-Modell mit Kantenvorhersage
class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=16, output_dim=1, dropout=0.2):
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
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, output_dim)
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
        edge_predictions = self.fc_edge(edge_embeddings).squeeze()

        return edge_predictions


# Trainingsfunktion
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


# Validierungsfunktion
def validate(loader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = F.mse_loss(out, batch.y)
            total_loss += loss.item()
    return total_loss / len(loader)


# Testfunktion
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


# Wahre vs. vorhergesagte Werte plotten
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('Wahre Werte')
    plt.ylabel('Vorhergesagte Werte')
    plt.title('GAT - Wahre vs. Vorhergesagte Werte')
    plt.show()


# Hauptfunktion mit erweitertem Training und Early Stopping basierend auf Validierungsverlust
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

    datasets = []

    # Physikalische und geografische Spalten
    physical_columns = ['ZUFLUSS_WOL', 'ZUFLUSS_WL', 'PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Laden des ersten Datasets für die Skalierung
    node_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_combined_Node.csv'
    edge_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_combined_Pipes.csv'

    nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
    edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')

    # One-Hot-Encoding für 'ROHRTYP'
    edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP')

    # Skalierer für physikalische und geografische Daten anpassen
    physical_scaler = StandardScaler()
    geo_scaler = MinMaxScaler()
    edge_scaler = StandardScaler()

    physical_scaler.fit(nodes_df_first[physical_columns])
    geo_scaler.fit(nodes_df_first[geo_columns])

    # Aktualisieren der edge_columns nach One-Hot-Encoding
    edge_columns = ['RORL', 'DM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'RAISE'] + list(edges_df_first.filter(like='ROHRTYP').columns)

    edge_scaler.fit(edges_df_first[edge_columns])

    # Laden aller Datasets mit Skalierung und Positionscodierung
    for i in range(1, 4000):  # Begrenzung auf 10 Datasets für schnellere Ausführung
        node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_combined_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_combined_Pipes.csv'
        data = load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler)
        datasets.append(data)

    # Aufteilen in Trainings- und Validierungsdaten
    train_data, val_data = train_test_split(datasets, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGAT(num_node_features=datasets[0].x.shape[1], num_edge_features=datasets[0].edge_attr.shape[1],
                    hidden_dim=64, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Lernraten-Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    num_epochs = 300  # Erweitertes Training
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Trainingsschritt
        train_loss = train(train_loader, model, optimizer, device)
        # Validierungsschritt
        val_loss = validate(val_loader, model, device)

        # Lernrate anpassen
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Trainingsverlust: {train_loss:.4f}, Validierungsverlust: {val_loss:.4f}')

        # Early Stopping basierend auf Validierungsverlust
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Geduldszähler zurücksetzen, wenn sich der Verlust verbessert
            # Bestes Modell speichern
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early Stopping bei Epoch {epoch}.")
            break

    # Laden des besten Modells
    model.load_state_dict(best_model_state)

    # Zusammenführen von Trainings- und Validierungsdaten für den Test
    test_loader = DataLoader(datasets, batch_size=4, shuffle=False)

    # Vorhersagen und Testmetriken erhalten
    test_mse, y_true, y_pred = test(test_loader, model, device)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_true, y_pred)

    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'R² Score: {test_r2:.4f}')

    # Wahre vs. vorhergesagte Werte plotten
    plot_predictions(y_true, y_pred)

    model_path = os.path.join(directory, 'edge_gat_model_positional_encoding.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Modell wurde gespeichert unter: {model_path}')


if __name__ == "__main__":
    main()
