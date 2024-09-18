import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score
from math import pi

# Konstanten für physikalische Berechnungen
GRAVITY = 9.80665  # Beschleunigung durch Gravitation in m/s^2
KINEMATIC_VISCOSITY = 1e-6  # Kinematische Viskosität von Wasser bei ~20°C in m^2/s

# Funktion zur Hinzufügung von Positionskodierungen für geografische Koordinaten
def add_positional_encoding(df, columns, max_value=10000):
    for col in columns:
        df[f'{col}_sin'] = np.sin(df[col] * (2 * pi / max_value))
        df[f'{col}_cos'] = np.cos(df[col] * (2 * pi / max_value))
    return df

# Funktion zur Behandlung fehlender Daten
def handle_missing_data(df, columns):
    for col in columns:
        if df[col].isnull().any():
            df[col + '_mask'] = df[col].notnull().astype(float)
            df[col] = df[col].fillna(0)
        else:
            df[col + '_mask'] = 1.0
    return df

# Funktion zur Berechnung der Kantenlängen basierend auf den Knotenkoodinaten
def compute_edge_lengths(edges_df, nodes_df):
    # Zuordnung der Knotennamen zu ihren Koordinaten
    node_coords = nodes_df.set_index('KNAM')[['XRECHTS', 'YHOCH']].to_dict('index')

    def calculate_length(row):
        start_coords = node_coords.get(row['ANFNAM'])
        end_coords = node_coords.get(row['ENDNAM'])
        if start_coords is None or end_coords is None:
            return np.nan  # Behandlung fehlender Koordinaten
        # Euklidische Distanz berechnen
        length = np.sqrt(
            (start_coords['XRECHTS'] - end_coords['XRECHTS']) ** 2 +
            (start_coords['YHOCH'] - end_coords['YHOCH']) ** 2
        )
        return length

    edges_df['L'] = edges_df.apply(calculate_length, axis=1)
    return edges_df

# Funktion zum Laden der Daten mit One-Hot-Encoding für 'ROHRTYP' und Skalierung der Merkmale
def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, onehot_encoder, all_rohrtyp_columns):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    # Zuordnung der Knotennamen zu Indizes
    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    # Kantenlängen berechnen
    edges_df = compute_edge_lengths(edges_df, nodes_df)

    # Behandlung fehlender Daten in den Knotenmerkmalen
    physical_columns = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
    nodes_df = handle_missing_data(nodes_df, physical_columns + geo_columns)

    # Skalierung der Knotenmerkmale
    nodes_df[physical_columns] = physical_scaler.transform(nodes_df[physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])

    # Positionskodierung für geografische Spalten hinzufügen
    nodes_df = add_positional_encoding(nodes_df, geo_columns)

    # Vorbereitung der Knotenmerkmale
    node_features = nodes_df.drop(columns=['KNAM']).values
    x = torch.tensor(node_features, dtype=torch.float)

    # Behandlung fehlender Daten in den Kantenmerkmalen
    edge_columns = ['RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE', 'L']
    edges_df = handle_missing_data(edges_df, edge_columns)

    # One-Hot-Encoding für 'ROHRTYP' mit dem angepassten Encoder
    rohrtyp_encoded = onehot_encoder.transform(edges_df[['ROHRTYP']])
    rohrtyp_df = pd.DataFrame(rohrtyp_encoded, columns=onehot_encoder.get_feature_names_out(['ROHRTYP']))

    # Sicherstellen, dass alle ROHRTYP-Spalten vorhanden sind
    for col in all_rohrtyp_columns:
        if col not in rohrtyp_df.columns:
            rohrtyp_df[col] = 0  # Fehlende Spalte mit Nullen hinzufügen

    # Spalten neu anordnen, um die Reihenfolge beim Fitten beizubehalten
    rohrtyp_df = rohrtyp_df[all_rohrtyp_columns]

    # Kantenmerkmale mit One-Hot-encodiertem 'ROHRTYP' kombinieren
    edges_df = pd.concat([edges_df, rohrtyp_df], axis=1)

    # Skalierung der Kantenmerkmale
    scaled_edge_features = edge_scaler.transform(edges_df[edge_columns + all_rohrtyp_columns])

    # Vorbereitung der Masken
    mask_columns = [col + '_mask' for col in edge_columns]
    edge_masks = edges_df[mask_columns].values

    # Für One-Hot-encodierte Spalten Masken von Einsen hinzufügen (da keine fehlenden Werte)
    onehot_mask_columns = [col + '_mask' for col in all_rohrtyp_columns]
    for col in onehot_mask_columns:
        edges_df[col] = 1.0
    onehot_masks = edges_df[onehot_mask_columns].values

    # Gesamte Maskenmatrix
    total_edge_masks = np.hstack([edge_masks, onehot_masks])

    # Vorbereitung der Kantenattribute
    edge_attributes = np.hstack([scaled_edge_features, total_edge_masks])
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    # Vorbereitung der Kantenindizes
    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Vorbereitung der Labels (RAU-Werte)
    y = torch.tensor(edges_df['RAU'].values, dtype=torch.float)

    # Erstellen des Data-Objekts
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Speichern der Kantenattributnamen und Anzahl der Merkmale und Masken
    data.edge_attr_names = edge_columns + all_rohrtyp_columns + mask_columns + onehot_mask_columns
    data.num_edge_features = scaled_edge_features.shape[1]
    data.num_edge_masks = total_edge_masks.shape[1]

    return data

# GAT-Modell mit Kantenvorhersage
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

        # Extrahieren der Anzahl der Merkmale und Masken
        num_edge_features = data.num_edge_features
        num_edge_masks = data.num_edge_masks

        # Trennung von Kantenattributen und Masken
        edge_features = edge_attr[:, :num_edge_features]
        edge_masks = edge_attr[:, num_edge_features:]

        # Anwendung der Masken auf die Kantenmerkmale
        edge_features = edge_features * edge_masks

        # GAT-Convolution-Layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Edge MLP
        edge_features = self.edge_mlp(edge_features)

        # Kanteneinbettungen
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_predictions = self.fc_edge(edge_embeddings).squeeze()

        return edge_predictions

# Berechnung der physikalischen Residuen mit der Darcy-Weisbach-Gleichung
def compute_physics_residuals(predictions, data):
    # ... (keine Änderungen an dieser Funktion)

    # Extrahieren der Kantenattribute
    edge_attr = data.edge_attr

    # Indizes der benötigten Variablen in edge_attr
    DM_index = data.edge_attr_names.index('DM')
    FLUSS_index = data.edge_attr_names.index('FLUSS')
    L_index = data.edge_attr_names.index('L')
    DP_index = data.edge_attr_names.index('DP')

    # Offset für Masken
    num_edge_features = data.num_edge_features
    num_edge_masks = data.num_edge_masks

    # Extrahieren von Variablen
    D = edge_attr[:, DM_index]
    Q = edge_attr[:, FLUSS_index]
    L = edge_attr[:, L_index]
    DP = edge_attr[:, DP_index]

    # Extrahieren von Masken
    D_mask = edge_attr[:, num_edge_features + data.edge_attr_names.index('DM_mask') - num_edge_features]
    Q_mask = edge_attr[:, num_edge_features + data.edge_attr_names.index('FLUSS_mask') - num_edge_features]
    L_mask = edge_attr[:, num_edge_features + data.edge_attr_names.index('L_mask') - num_edge_features]
    DP_mask = edge_attr[:, num_edge_features + data.edge_attr_names.index('DP_mask') - num_edge_features]

    # Anwendung der Masken
    D = D * D_mask
    Q = Q * Q_mask
    L = L * L_mask
    DP = DP * DP_mask

    # Nur Residuen berechnen, wo alle Masken 1 sind
    valid_indices = (D_mask * Q_mask * L_mask * DP_mask) == 1.0

    if valid_indices.sum() == 0:
        # Keine gültigen Daten zur Berechnung der Residuen
        return torch.tensor(0.0, device=predictions.device)

    # Filterung der Variablen
    D = D[valid_indices]
    Q = Q[valid_indices]
    L = L[valid_indices]
    DP = DP[valid_indices]
    epsilon = predictions[valid_indices]

    # Sicherstellen, dass der Durchmesser nicht null oder negativ ist
    D = torch.clamp(D, min=1e-6)

    # Querschnittsfläche berechnen
    A = np.pi * (D / 2) ** 2  # Fläche in m^2

    # Strömungsgeschwindigkeit berechnen
    V = Q / A  # Geschwindigkeit in m/s

    # Reynolds-Zahl berechnen
    Re = (V * D) / KINEMATIC_VISCOSITY

    # Sicherstellen, dass die Reynolds-Zahl nicht null ist
    Re = torch.clamp(Re, min=1.0)

    # Relative Rauheit
    relative_roughness = epsilon / D

    # Reibungsfaktor mit der Swamee-Jain-Gleichung berechnen
    epsilon_over_D = relative_roughness / 3.7
    Re_term = 5.74 / Re ** 0.9
    log_argument = epsilon_over_D + Re_term
    # Sicherstellen, dass log_argument positiv ist
    log_argument = torch.clamp(log_argument, min=1e-8)
    f = 0.25 / (torch.log10(log_argument)) ** 2

    # Höhenverlust h_f berechnen
    h_f = f * (L / D) * (V ** 2) / (2 * GRAVITY)

    # Druckabfall in Höhenverlust umrechnen (Annahme: Wasserdichte von 1000 kg/m^3)
    rho = 1000.0  # Dichte in kg/m^3
    observed_h_f = DP / (rho * GRAVITY)

    # Residuen: Differenz zwischen berechnetem und beobachtetem Höhenverlust
    residuals = h_f - observed_h_f

    return residuals

# Trainingsfunktion mit physikalisch informiertem Verlust
def train(loader, model, optimizer, device, phys_loss_weight=1.0):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        # Datenverlust: Mittlerer quadratischer Fehler zwischen vorhergesagten und tatsächlichen RAU-Werten
        mse_loss = F.mse_loss(out, batch.y)

        # Physikalisch informierter Verlust: Mittlere quadratische Residuen der hydraulischen Gleichungen
        phys_residuals = compute_physics_residuals(out, batch)
        phys_loss = torch.mean(phys_residuals ** 2)

        # Gesamtverlust: Summe aus Datenverlust und gewichteten physikalischen Verlust
        loss = mse_loss + phys_loss_weight * phys_loss
        loss.backward()
        optimizer.step()
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

# Plotten von tatsächlichen vs. vorhergesagten Werten
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('Tatsächliche Werte')
    plt.ylabel('Vorhergesagte Werte')
    plt.title('GAT - Tatsächlich vs. Vorhergesagt')
    plt.show()

# Hauptfunktion mit erweitertem Training und Early Stopping
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten/Zwischenspeicher/'

    datasets = []

    # Physikalische und geografische Spalten
    physical_columns = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP']
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
    edge_columns = ['RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE', 'L']

    # Alle einzigartigen 'ROHRTYP'-Kategorien aus allen Datensätzen sammeln
    all_rohrtyp_categories = set()
    for i in range(1, 11):  # Bereich nach Bedarf anpassen
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
        edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
        all_rohrtyp_categories.update(edges_df['ROHRTYP'].unique())

    # One-Hot Encoder auf alle möglichen 'ROHRTYP'-Kategorien fitten
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoder.fit(np.array(list(all_rohrtyp_categories)).reshape(-1, 1))
    all_rohrtyp_columns = onehot_encoder.get_feature_names_out(['ROHRTYP']).tolist()

    # Ersten Datensatz für die Skalierung laden
    node_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_Node.csv'
    edge_file_first = f'{directory}SyntheticData-Spechbach_Roughness_1_Pipes.csv'

    nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
    edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')

    # Kantenlängen berechnen
    edges_df_first = compute_edge_lengths(edges_df_first, nodes_df_first)

    # Behandlung fehlender Daten vor dem Fitten der Skalierer
    nodes_df_first = handle_missing_data(nodes_df_first, physical_columns + geo_columns)
    edges_df_first = handle_missing_data(edges_df_first, edge_columns)

    # One-Hot-Encoding für 'ROHRTYP' im ersten Datensatz
    rohrtyp_encoded = onehot_encoder.transform(edges_df_first[['ROHRTYP']])
    rohrtyp_df_first = pd.DataFrame(rohrtyp_encoded, columns=all_rohrtyp_columns)

    # Sicherstellen, dass alle ROHRTYP-Spalten vorhanden sind
    for col in all_rohrtyp_columns:
        if col not in rohrtyp_df_first.columns:
            rohrtyp_df_first[col] = 0  # Fehlende Spalte mit Nullen hinzufügen

    # Kantenmerkmale mit One-Hot-encodiertem 'ROHRTYP' kombinieren
    edges_df_first = pd.concat([edges_df_first, rohrtyp_df_first], axis=1)

    # Skalierer für physikalische und geografische Daten fitten
    physical_scaler = StandardScaler()
    geo_scaler = MinMaxScaler()
    edge_scaler = StandardScaler()

    physical_scaler.fit(nodes_df_first[physical_columns])
    geo_scaler.fit(nodes_df_first[geo_columns])
    edge_scaler.fit(edges_df_first[edge_columns + all_rohrtyp_columns])

    # Alle Datensätze mit Skalierung und Positionskodierung laden
    for i in range(1, 11):  # Bereich nach Bedarf anpassen
        node_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
        data = load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, onehot_encoder,
                         all_rohrtyp_columns)
        datasets.append(data)

    loader = DataLoader(datasets, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGAT(
        num_node_features=datasets[0].x.shape[1],
        num_edge_features=datasets[0].edge_attr.shape[1] - datasets[0].num_edge_masks,  # Anzahl der Merkmale ohne Masken
        hidden_dim=78,
        dropout=0.205
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0088, weight_decay=2.9e-05)

    # Lernraten-Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    num_epochs = 300  # Erweitertes Training
    phys_loss_weight = 0.1  # Anpassen je nach Wichtigkeit des physikalischen Verlusts
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Trainingsschritt mit physikalisch informiertem Verlust
        loss = train(loader, model, optimizer, device, phys_loss_weight=phys_loss_weight)

        # Lernrate anpassen
        scheduler.step(loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Training Loss: {loss:.4f}')

        # Überprüfung der Early-Stopping-Bedingung
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0  # Geduldzähler zurücksetzen, wenn sich der Verlust verbessert
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Vorhersagen und Testmetriken erhalten
    test_mse, y_true, y_pred = test(loader, model, device)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_true, y_pred)

    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'R² Score: {test_r2:.4f}')

    # Tatsächliche vs. vorhergesagte Werte plotten
    plot_predictions(y_true, y_pred)

    model_path = os.path.join(directory, 'edge_gat_model_pinn.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at: {model_path}')

if __name__ == "__main__":
    main()
