import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Funktion zum Laden der Daten
def load_data(node_file, edge_file):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
    node_features = nodes_df.drop(columns=['KNAM']).values
    edge_labels = edges_df['RAU'].values

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(edge_labels, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    return data, y

# GCN-Modell für Kantenregression
class EdgeGCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim):
        super(EdgeGCN, self).__init__()
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

# Trainingsfunktion
def train(data, y, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Testfunktion
def test(data, y, model):
    model.eval()
    with torch.no_grad():
        out = model(data)
        mse = F.mse_loss(out, y)
    return mse.item()

# Lade mehrere Datensätze (z.B. 10 Datensätze für das Training)
datasets = []

# Verzeichnis für die Dateien angeben
directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

for i in range(1, 10000):
    node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Node.csv'
    edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
    data, y = load_data(node_file, edge_file)
    datasets.append((data, y))


# Teile die Daten in 80% Training und 20% Testen auf
train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=42)

# Überprüfe die Größe der aufgeteilten Datensätze
print(f"Anzahl der Trainingsdatensätze: {len(train_data)}")
print(f"Anzahl der Testdatensätze: {len(test_data)}")


# Initialisiere das Modell
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeGCN(num_node_features=train_data[0][0].x.shape[1], output_dim=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Trainiere das Modell auf den Trainingsdatensätzen
for epoch in range(100):
    total_loss = 0
    for data, y in train_data:
        data = data.to(device)
        y = y.to(device)
        total_loss += train(data, y, model, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}')

# Teste das Modell
total_mse = 0
for data, y in test_data:
    data = data.to(device)
    y = y.to(device)
    total_mse += test(data, y, model)
print(f'Total MSE: {total_mse:.4f}')



# Holen des Wurzelverzeichnisses des Pycharm-Projekts (2 Ebenen über der aktuellen Datei)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Erstelle das results/models Verzeichnis im Projektverzeichnis
results_dir = os.path.join(project_root, 'results', 'models')
os.makedirs(results_dir, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert

# Dateipfad zum Speichern des Modells
model_file = os.path.join(results_dir, 'edge_gcn_model.pth')

# Speichere das Modell
torch.save(model.state_dict(), model_file)

print(f'Modell wurde gespeichert unter {model_file}')
