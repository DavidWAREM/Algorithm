import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
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

    # Stellen Sie sicher, dass die relevanten Spalten numerisch sind
    edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']].astype(float)

    # Definieren Sie ein kleines Epsilon für die Näherung an Null
    epsilon = 1e-6

    # **Zielvariable erstellen, bevor die Skalierung erfolgt**
    target_condition = (edges_df['FLUSS_WL'].abs() < epsilon) | (edges_df['FLUSS_WOL'].abs() < epsilon) | \
                       (edges_df['VM_WL'].abs() < epsilon) | (edges_df['VM_WOL'].abs() < epsilon)
    y = torch.tensor(target_condition.astype(float).values, dtype=torch.float)

    # Anzahl der positiven Beispiele in diesem Datensatz
    num_positive_samples = y.sum().item()
    if num_positive_samples == 0:
        print(f"Warnung: Keine positiven Beispiele im Datensatz {edge_file}")
    else:
        print(f"Anzahl positiver Beispiele in {edge_file}: {num_positive_samples}")

    # Optional: Anzeigen der Zeilen, die als positive Beispiele identifiziert wurden
    if num_positive_samples > 0:
        positive_edges = edges_df[target_condition]
        print("Positive Beispiele in diesem Datensatz:")
        print(positive_edges[['ANFNAM', 'ENDNAM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']])

    # **Ab hier erfolgt die Skalierung der Daten**

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
    edge_columns = ['RORL', 'DM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'RAISE'] + list(
        edges_df.filter(like='ROHRTYP').columns)
    edges_df[edge_columns] = edge_scaler.transform(edges_df[edge_columns])

    edge_attributes = edges_df[edge_columns].values

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data



# GAT-Modell mit Kantenvorhersage für binäre Klassifikation
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

# Trainingsfunktion
def train(loader, model, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validierungsfunktion
def validate(loader, model, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Testfunktion
def test(loader, model, device):
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Überprüfen der einzigartigen Klassen in y_true
    unique_classes = np.unique(y_true)
    print("Einzigartige Klassen in y_true:", unique_classes)

    return y_true, y_pred

# Wahre vs. vorhergesagte Werte plotten
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('Wahre Werte')
    plt.ylabel('Vorhergesagte Wahrscheinlichkeiten')
    plt.title('GAT - Wahre vs. Vorhergesagte Wahrscheinlichkeiten')
    plt.show()

# Hauptfunktion mit erweitertem Training und Early Stopping basierend auf Validierungsverlust
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten_Valve/Zwischenspeicher/'

    datasets = []

    # Physikalische und geografische Spalten
    physical_columns = ['ZUFLUSS_WOL', 'ZUFLUSS_WL', 'PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Laden des ersten Datasets für die Skalierung
    node_file_first = f'{directory}SyntheticData-Spechbach_Valve_1_combined_Node.csv'
    edge_file_first = f'{directory}SyntheticData-Spechbach_Valve_1_combined_Pipes.csv'

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
    edge_columns = ['RORL', 'DM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'RAISE'] + list(
        edges_df_first.filter(like='ROHRTYP').columns)

    edge_scaler.fit(edges_df_first[edge_columns])

    # Laden aller Datasets mit Skalierung und Positionscodierung
    for i in range(1, 109):
        node_file = f'{directory}SyntheticData-Spechbach_Valve_{i}_combined_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Valve_{i}_combined_Pipes.csv'
        data = load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler)
        datasets.append(data)

    # Überprüfen der Anzahl positiver Beispiele in allen Datensätzen
    total_positive = sum([data.y.sum().item() for data in datasets])
    print(f'Gesamtzahl positiver Beispiele: {total_positive}')

    # Überprüfen, ob insgesamt positive Beispiele vorhanden sind
    if total_positive == 0:
        print("Warnung: Keine positiven Beispiele in allen Datensätzen gefunden. Bitte überprüfen Sie die Daten und die Bedingung für die Zielvariable.")
        return

    # Aufteilen in Trainings- und Validierungsdaten
    train_data, val_data = train_test_split(datasets, test_size=0.2, random_state=42)

    # Überprüfen, ob positive Beispiele in Trainings- und Validierungsdaten vorhanden sind
    train_positive = sum([data.y.sum().item() for data in train_data])
    val_positive = sum([data.y.sum().item() for data in val_data])
    print(f'Positive Beispiele im Training: {train_positive}, im Validierung: {val_positive}')

    # Überprüfen, ob positive Beispiele vorhanden sind
    if train_positive == 0 or val_positive == 0:
        print("Warnung: Keine positiven Beispiele im Training oder in der Validierung. Bitte überprüfen Sie die Datenaufteilung.")
        return

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGAT(num_node_features=datasets[0].x.shape[1], num_edge_features=datasets[0].edge_attr.shape[1],
                    hidden_dim=64, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Verlustfunktion mit Klassengewichtung
    pos_weight = torch.tensor([len(train_data[0].edge_index[0]) - 1], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Lernraten-Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    num_epochs = 300  # Erweitertes Training
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Trainingsschritt
        train_loss = train(train_loader, model, optimizer, device, criterion)
        # Validierungsschritt
        val_loss = validate(val_loader, model, device, criterion)

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
    test_loader = DataLoader(datasets, batch_size=1, shuffle=False)

    # Vorhersagen und Testmetriken erhalten
    y_true, y_pred = test(test_loader, model, device)

    # Überprüfen der einzigartigen Klassen in y_true
    unique_classes = np.unique(y_true)
    print("Einzigartige Klassen in y_true (Testdaten):", unique_classes)

    # Überprüfen, ob y_true mindestens zwei Klassen enthält
    if len(unique_classes) < 2:
        print("Warnung: y_true enthält nur eine Klasse. ROC AUC Score kann nicht berechnet werden.")
        auc = None
    else:
        # Metriken berechnen
        auc = roc_auc_score(y_true, y_pred)

    # Binäre Vorhersagen
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)

    print(f'Genauigkeit: {accuracy:.4f}')
    if auc is not None:
        print(f'AUC: {auc:.4f}')

    # Wahre vs. vorhergesagte Wahrscheinlichkeiten plotten
    plot_predictions(y_true, y_pred)

    model_path = os.path.join(directory, 'edge_gat_model_classification.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Modell wurde gespeichert unter: {model_path}')

if __name__ == "__main__":
    main()
