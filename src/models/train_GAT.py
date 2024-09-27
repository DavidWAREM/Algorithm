import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, auc

# Funktion zur Hinzufügung von Positionscodierung mittels Sinus und Kosinus für geografische Koordinaten
def add_positional_encoding(df, columns, max_value=10000):
    for col in columns:
        df[f'{col}_sin'] = np.sin(df[col] * (2 * pi / max_value))
        df[f'{col}_cos'] = np.cos(df[col] * (2 * pi / max_value))
    return df

# Graph-based Imputation für fehlende ZUFLUSS_WL-Werte (global definiert)
def graph_based_imputation(df, edge_index, feature_name):
    node_values = df[feature_name].values
    missing_mask = np.isnan(node_values)
    # Erstellen Sie einen Adjazenzlisten-Dictionary
    adjacency = {i: [] for i in range(len(df))}
    for src, dst in edge_index.T:
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    # Iterieren über fehlende Werte
    for idx in np.where(missing_mask)[0]:
        neighbors = adjacency[idx]
        neighbor_values = [node_values[n] for n in neighbors if not np.isnan(node_values[n])]
        if neighbor_values:
            # Physikalisches Modell: Mittelwert der Nachbarn
            node_values[idx] = np.mean(neighbor_values)
        else:
            # Wenn keine Nachbarn bekannt sind, setzen wir den Mittelwert aller bekannten Werte
            node_values[idx] = np.nanmean(node_values)
    df[feature_name] = node_values
    return df

# Funktion zum Laden der Daten mit graph-based Imputation
def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, included_nodes, zfluss_wl_nodes):
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    nodes_df['node_idx'] = nodes_df['KNAM'].map(node_mapping)
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

    # Stellen Sie sicher, dass die relevanten Spalten numerisch sind
    edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']].astype(float)

    # Definieren Sie ein kleines Epsilon für die Näherung an Null
    epsilon = 1e-6

    # Zielvariable erstellen, bevor die Skalierung erfolgt
    target_condition = (edges_df['FLUSS_WL'].abs() < epsilon) | (edges_df['FLUSS_WOL'].abs() < epsilon) | \
                       (edges_df['VM_WL'].abs() < epsilon) | (edges_df['VM_WOL'].abs() < epsilon)
    y = torch.tensor(target_condition.astype(float).values, dtype=torch.float)

    # Ab hier erfolgt die Anpassung der Knotenattribute

    # Geografische Spalten
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Physikalische Spalten
    adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    additional_physical_columns = ['ZUFLUSS_WL']
    all_physical_columns = adjusted_physical_columns + additional_physical_columns

    # Für Knoten, die nicht in included_nodes sind, die Werte der adjusted_physical_columns auf NaN setzen
    nodes_df['Included'] = nodes_df['KNAM'].isin(included_nodes)
    for col in adjusted_physical_columns:
        nodes_df.loc[~nodes_df['Included'], col] = np.nan

    # Umgang mit ZUFLUSS_WL nur für bestimmte Knoten
    nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
        lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in zfluss_wl_nodes else np.nan,
        axis=1
    )

    # Indikatorspalten hinzufügen, die angeben, ob der Wert fehlt
    for col in all_physical_columns:
        nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)

    # Graph-based Imputation für fehlende ZUFLUSS_WL-Werte
    nodes_df = graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

    # Fehlende Werte für andere physikalische Spalten mit KNN-Imputation behandeln
    imputer = KNNImputer(n_neighbors=5)
    nodes_df[adjusted_physical_columns] = imputer.fit_transform(nodes_df[adjusted_physical_columns])

    # Entfernen der Hilfsspalte
    nodes_df = nodes_df.drop(columns=['Included'])

    # Anwenden der Skalierung
    nodes_df[all_physical_columns] = physical_scaler.transform(nodes_df[all_physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])

    # Positionscodierung für geografische Spalten hinzufügen
    nodes_df = add_positional_encoding(nodes_df, geo_columns)

    # Knotenfeatures erstellen
    node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values

    # Entfernen des Features 'RAU' aus den Edge-Features
    edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)
    # Stellen Sie sicher, dass 'RAU' nicht in edge_columns enthalten ist
    if 'RAU' in edges_df.columns:
        edges_df = edges_df.drop(columns=['RAU'])

    # Skalierung der Kantenattribute
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

    return y_true, y_pred

# ROC-Kurve plotten
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC-Kurve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Precision-Recall-Kurve plotten
def plot_precision_recall(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall-Kurve (AP = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall-Kurve')
    plt.legend(loc="lower left")
    plt.show()

# Konfusionsmatrix plotten
def plot_confusion_matrix(y_true, y_pred_binary):
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Konfusionsmatrix')
    plt.show()

# Hauptfunktion mit Anpassungen für fehlende Werte
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten_Valve/Zwischenspeicher/'

    datasets = []

    # Liste der Knoten, für die Messdaten vorhanden sind
    included_nodes = [
        'K0003', 'K0004', 'K0005', 'K0006', 'K0011', 'K0012', 'K0014', 'K0016',
        'K0017', 'K0020', 'K0021', 'K0028', 'K0032', 'K0033', 'K0034', 'K0061',
        'K0062', 'K0063', 'K0064', 'K0080', 'K0086', 'K0087', 'K0088', 'K0090',
        'K0091', 'K0093', 'K0094', 'K0115', 'K0119', 'K0121', 'K0125', 'K0135',
        'K0136', 'K0138', 'K0144', 'K0145', 'K0147', 'K0148', 'K0151', 'K0155',
        'K0156', 'K0163', 'K0164', 'K0165', 'K0166', 'K0173', 'K0174', 'K0176',
        'K0179', 'K0180', 'K0183', 'K0184', 'K0185', 'K0186', 'K0191', 'K0195',
        'K0201', 'K0207', 'K0208', 'K0209', 'K0210', 'K0211', 'K0213', 'K0214',
        'K0220', 'K0229', 'K0230', 'K0238', 'K0239', 'K0245', 'K0254', 'K0260',
        'K0261', 'K0294', 'K0300', 'K0301', 'K0305', 'K0311', 'K0314', 'K0315',
        'K0316', 'K0317', 'K0318', 'K0319', 'K0327', 'K0328', 'K0329', 'K0330',
        'K0333', 'K0334', 'K0335', 'K0336', 'K0340', 'K0347', 'K0357', 'K0215',
        'K0388'
    ]

    # Liste der Knoten, für die ZUFLUSS_WL verfügbar ist (sortiert und ohne Duplikate)
    zfluss_wl_nodes = [
        'K0003', 'K0004', 'K0005', 'K0006', 'K0007', 'K0008', 'K0011', 'K0012', 'K0013', 'K0014',
        'K0015', 'K0016', 'K0021', 'K0022', 'K0023', 'K0025', 'K0026', 'K0028', 'K0029', 'K0030',
        'K0031', 'K0033', 'K0034', 'K0036', 'K0037', 'K0038', 'K0040', 'K0041', 'K0042', 'K0045',
        'K0046', 'K0059', 'K0060', 'K0061', 'K0062', 'K0063', 'K0065', 'K0066', 'K0067', 'K0070',
        'K0071', 'K0073', 'K0074', 'K0076', 'K0077', 'K0078', 'K0079', 'K0080', 'K0081', 'K0082',
        'K0083', 'K0084', 'K0085', 'K0086', 'K0089', 'K0090', 'K0091', 'K0093', 'K0094', 'K0095',
        'K0101', 'K0106', 'K0108', 'K0109', 'K0110', 'K0111', 'K0112', 'K0113', 'K0115', 'K0118',
        'K0119', 'K0121', 'K0122', 'K0125', 'K0127', 'K0128', 'K0129', 'K0130', 'K0131', 'K0132',
        'K0133', 'K0135', 'K0136', 'K0137', 'K0138', 'K0140', 'K0141', 'K0142', 'K0147', 'K0148',
        'K0151', 'K0152', 'K0155', 'K0156', 'K0160', 'K0161', 'K0162', 'K0163', 'K0164', 'K0165',
        'K0166', 'K0168', 'K0169', 'K0170', 'K0171', 'K0172', 'K0173', 'K0174', 'K0176', 'K0177',
        'K0179', 'K0180', 'K0181', 'K0182', 'K0183', 'K0184', 'K0185', 'K0186', 'K0188', 'K0189',
        'K0190', 'K0191', 'K0193', 'K0195', 'K0196', 'K0197', 'K0198', 'K0199', 'K0200', 'K0201',
        'K0202', 'K0204', 'K0205', 'K0206', 'K0207', 'K0208', 'K0209', 'K0210', 'K0211', 'K0213',
        'K0214', 'K0215', 'K0219', 'K0220', 'K0222', 'K0223', 'K0226', 'K0229', 'K0230', 'K0232',
        'K0233', 'K0234', 'K0235', 'K0237', 'K0238', 'K0239', 'K0244', 'K0245', 'K0248', 'K0249',
        'K0250', 'K0251', 'K0252', 'K0255', 'K0256', 'K0260', 'K0261', 'K0262', 'K0264', 'K0265',
        'K0266', 'K0277', 'K0280', 'K0281', 'K0282', 'K0283', 'K0287', 'K0289', 'K0290', 'K0291',
        'K0292', 'K0295', 'K0296', 'K0298', 'K0299', 'K0300', 'K0301', 'K0304', 'K0305', 'K0306',
        'K0307', 'K0308', 'K0309', 'K0311', 'K0312', 'K0313', 'K0314', 'K0315', 'K0316', 'K0317',
        'K0318', 'K0319', 'K0320', 'K0321', 'K0322', 'K0323', 'K0324', 'K0325', 'K0327', 'K0328',
        'K0329', 'K0330', 'K0332', 'K0333', 'K0334', 'K0336', 'K0340', 'K0342', 'K0346', 'K0347',
        'K0357', 'K0364', 'K0365', 'K0388'
    ]

    # Geografische Spalten
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Physikalische Spalten
    adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    additional_physical_columns = ['ZUFLUSS_WL']
    all_physical_columns = adjusted_physical_columns + additional_physical_columns

    # Laden des ersten Datasets für die Skalierung
    node_file_first = f'{directory}SyntheticData-Spechbach_Valve_1_combined_Node.csv'
    edge_file_first = f'{directory}SyntheticData-Spechbach_Valve_1_combined_Pipes.csv'

    nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
    edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')

    # Für Knoten, die nicht in included_nodes sind, die Werte der adjusted_physical_columns auf NaN setzen
    nodes_df_first['Included'] = nodes_df_first['KNAM'].isin(included_nodes)
    for col in adjusted_physical_columns:
        nodes_df_first.loc[~nodes_df_first['Included'], col] = np.nan

    # Umgang mit ZUFLUSS_WL nur für bestimmte Knoten
    nodes_df_first['ZUFLUSS_WL'] = nodes_df_first.apply(
        lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in zfluss_wl_nodes else np.nan,
        axis=1
    )

    # Indikatorspalten hinzufügen, die angeben, ob der Wert fehlt
    for col in all_physical_columns:
        nodes_df_first[f'{col}_missing'] = nodes_df_first[col].isna().astype(float)

    # KNN-Imputation für adjusted_physical_columns
    imputer = KNNImputer(n_neighbors=5)
    nodes_df_first[adjusted_physical_columns] = imputer.fit_transform(nodes_df_first[adjusted_physical_columns])

    # Graph-based Imputation für ZUFLUSS_WL im ersten Datensatz
    node_mapping_first = {name: idx for idx, name in enumerate(nodes_df_first['KNAM'])}
    edges_df_first['ANFNR'] = edges_df_first['ANFNAM'].map(node_mapping_first)
    edges_df_first['ENDNR'] = edges_df_first['ENDNAM'].map(node_mapping_first)
    edge_index_first = edges_df_first[['ANFNR', 'ENDNR']].values.T

    nodes_df_first = graph_based_imputation(nodes_df_first, edge_index_first, 'ZUFLUSS_WL')

    # Entfernen der Hilfsspalte
    nodes_df_first = nodes_df_first.drop(columns=['Included'])

    # Skalierer für physikalische und geografische Daten anpassen
    physical_scaler = StandardScaler()
    geo_scaler = MinMaxScaler()
    edge_scaler = StandardScaler()

    physical_scaler.fit(nodes_df_first[all_physical_columns])
    geo_scaler.fit(nodes_df_first[geo_columns])

    # One-Hot-Encoding für 'ROHRTYP'
    edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP')

    # Entfernen des Features 'RAU' aus edges_df_first, falls vorhanden
    if 'RAU' in edges_df_first.columns:
        edges_df_first = edges_df_first.drop(columns=['RAU'])

    # Aktualisieren der edge_columns nach One-Hot-Encoding
    edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df_first.filter(like='ROHRTYP').columns)

    edge_scaler.fit(edges_df_first[edge_columns])

    # Laden aller Datasets mit Skalierung und Positionscodierung
    for i in range(1, 109):
        node_file = f'{directory}SyntheticData-Spechbach_Valve_{i}_combined_Node.csv'
        edge_file = f'{directory}SyntheticData-Spechbach_Valve_{i}_combined_Pipes.csv'
        data = load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, included_nodes, zfluss_wl_nodes)
        datasets.append(data)

    # Überprüfen, ob mindestens ein gültiger Datensatz vorhanden ist
    if not datasets:
        print("Keine gültigen Datensätze verfügbar. Bitte überprüfen Sie die Daten.")
        return

    # Aufteilen in Trainings- und Validierungsdaten
    train_data, val_data = train_test_split(datasets, test_size=0.2, random_state=42)

    # Überprüfen, ob positive Beispiele in Trainings- und Validierungsdaten vorhanden sind
    train_positive = sum([data.y.sum().item() for data in train_data])
    val_positive = sum([data.y.sum().item() for data in val_data])

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

    # Überprüfen, ob y_true mindestens zwei Klassen enthält
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print("Warnung: y_true enthält nur eine Klasse. ROC AUC Score kann nicht berechnet werden.")
        auc_score = None
    else:
        # Metriken berechnen
        auc_score = roc_auc_score(y_true, y_pred)

    # Binäre Vorhersagen
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)

    print(f'Genauigkeit: {accuracy:.4f}')
    if auc_score is not None:
        print(f'AUC: {auc_score:.4f}')

    # Plots erzeugen
    plot_roc_curve(y_true, y_pred)
    plot_precision_recall(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred_binary)

    model_path = os.path.join(directory, 'edge_gat_model_classification.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Modell wurde gespeichert unter: {model_path}')

if __name__ == "__main__":
    main()
