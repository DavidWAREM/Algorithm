import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


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
        x = self.conv2(x, edge_index)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_predictions = self.fc_edge(edge_embeddings).squeeze()
        return edge_predictions


# Funktion zum Laden der Daten (ohne RAU)
def load_new_data(node_file, edge_file):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
    node_features = nodes_df.drop(columns=['KNAM']).values

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    return data


# Lade das gespeicherte Modell und mache Vorhersagen
def predict_rau(node_file, edge_file, model_file):
    # Lade den neuen Datensatz
    data = load_new_data(node_file, edge_file)

    # Initialisiere das Modell und lade die gespeicherten Gewichte
    model = EdgeGCN(num_node_features=data.x.shape[1], output_dim=1)
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Setze das Modell in den Evaluierungsmodus

    # Konvertiere die Daten in das richtige Format für das Modell
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Mache Vorhersagen
    with torch.no_grad():
        rau_predictions = model(data)

    print("Vorhergesagte RAU-Werte:", rau_predictions.cpu().numpy())


# Beispiel für die Anwendung
node_file = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Predicted/SyntheticData-Spechbach_Roughness_0_Node.csv'
edge_file = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Predicted/SyntheticData-Spechbach_Roughness_0_Pipes.csv'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
results_dir = os.path.join(project_root, 'results', 'models')
model_file = os.path.join(results_dir, 'edge_gcn_model.pth')

model_file = 'results/models/edge_gcn_model.pth'

predict_rau(node_file, edge_file, model_file)
