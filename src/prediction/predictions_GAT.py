import os
import pandas as pd
import torch
import numpy as np
import yaml
import joblib
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import logging
import glob

# Initialisiere das Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definiere die EdgeGAT-Modellklasse (muss mit dem gespeicherten Modell übereinstimmen)
class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, dropout=0.15):
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
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, 1)  # Output ist ein Logit
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass durch conv1.")

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass durch conv2.")

        x = self.conv3(x, edge_index)
        logger.debug("Forward pass durch conv3.")

        edge_features = self.edge_mlp(edge_attr)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_logits = self.fc_edge(edge_embeddings).squeeze()
        logger.debug("Berechnet edge_logits.")

        return edge_logits  # Gibt Logits zurück

# Funktion zur Hinzufügung der Positionskodierung
def add_positional_encoding(df, columns, max_value=10000):
    for col in columns:
        df[f'{col}_sin'] = np.sin(df[col] * (2 * np.pi / max_value))
        df[f'{col}_cos'] = np.cos(df[col] * (2 * np.pi / max_value))
    return df

# Funktion zur Graph-basierten Imputation
def graph_based_imputation(df, edge_index, feature_name):
    node_values = df[feature_name].values
    missing_mask = np.isnan(node_values)
    # Adjazenzliste erstellen
    adjacency = {i: [] for i in range(len(df))}
    for src, dst in edge_index.T:
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    # Iterieren über fehlende Werte
    for idx in np.where(missing_mask)[0]:
        neighbors = adjacency[idx]
        neighbor_values = [node_values[n] for n in neighbors if not np.isnan(node_values[n])]
        if neighbor_values:
            node_values[idx] = np.mean(neighbor_values)
        else:
            node_values[idx] = np.nanmean(node_values)
    df[feature_name] = node_values
    logger.debug(f"Graph-basierte Imputation für Feature '{feature_name}' durchgeführt.")
    return df

# Funktion zur Vorverarbeitung der neuen Daten
def process_new_data(nodes_df, edges_df, included_nodes, zfluss_wl_nodes,
                     physical_scaler, geo_scaler, edge_scaler,
                     adjusted_physical_columns, additional_physical_columns, geo_columns):
    # Entferne 'RAU' Spalte falls vorhanden
    if 'RAU' in edges_df.columns:
        edges_df = edges_df.drop(columns=['RAU'])
        logger.debug("'RAU' Spalte gefunden und aus den Kantendaten entfernt.")

    # Knotennamen bereinigen und zu Indizes mappen
    nodes_df['KNAM'] = nodes_df['KNAM'].astype(str).str.strip().str.lower()
    edges_df['ANFNAM'] = edges_df['ANFNAM'].astype(str).str.strip().str.lower()
    edges_df['ENDNAM'] = edges_df['ENDNAM'].astype(str).str.strip().str.lower()

    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    nodes_df['node_idx'] = nodes_df['KNAM'].map(node_mapping)
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    # Überprüfe auf fehlende Knotenindices
    missing_anfnr = edges_df['ANFNR'].isnull()
    missing_endnr = edges_df['ENDNR'].isnull()

    if missing_anfnr.any() or missing_endnr.any():
        missing_anfnam = edges_df.loc[missing_anfnr, 'ANFNAM'].unique()
        missing_endnam = edges_df.loc[missing_endnr, 'ENDNAM'].unique()
        logger.error(f"Fehlende Knotenindices für ANFNAMs: {missing_anfnam}, ENDNAMs: {missing_endnam}")
        raise ValueError("Kantendaten enthalten Knoten, die nicht in den Knotendaten gefunden wurden.")

    # Konvertiere Indizes zu Integer
    edges_df['ANFNR'] = edges_df['ANFNR'].astype(int)
    edges_df['ENDNR'] = edges_df['ENDNR'].astype(int)

    # One-Hot-Encoding für 'ROHRTYP' mit dtype=float
    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP', dtype=float)
    logger.debug("One-Hot-Encoding für 'ROHRTYP' durchgeführt.")

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

    # Sicherstellen, dass relevante Spalten numerisch sind
    edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']].astype(float)
    logger.debug("Relevante Kantenspalten in float konvertiert.")

    # Anpassen der Knotenattribute
    nodes_df['Included'] = nodes_df['KNAM'].isin([n.lower() for n in included_nodes])
    for col in adjusted_physical_columns:
        nodes_df.loc[~nodes_df['Included'], col] = np.nan
        logger.debug(f"{col} auf NaN gesetzt für nicht inkludierte Knoten.")

    # 'ZUFLUSS_WL' nur für spezifische Knoten behandeln
    nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
        lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in zfluss_wl_nodes] else np.nan,
        axis=1
    )
    logger.debug("'ZUFLUSS_WL' für spezifische Knoten behandelt.")

    # Indikatorspalten für fehlende Werte hinzufügen
    all_physical_columns = adjusted_physical_columns + additional_physical_columns
    for col in all_physical_columns:
        nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)
        logger.debug(f"Fehlender Indikator für {col} hinzugefügt.")

    # Graph-basierte Imputation für fehlende 'ZUFLUSS_WL' Werte
    nodes_df = graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

    # Fehlende Werte für andere physikalische Spalten mit KNN Imputer behandeln
    imputer = KNNImputer(n_neighbors=5)
    nodes_df[adjusted_physical_columns] = imputer.fit_transform(nodes_df[adjusted_physical_columns])
    logger.debug("KNN-Imputation für angepasste physikalische Spalten durchgeführt.")

    # Hilfsspalte entfernen
    nodes_df = nodes_df.drop(columns=['Included'])
    logger.debug("'Included' Spalte aus den Knotendaten entfernt.")

    # Skalierung anwenden
    nodes_df[all_physical_columns] = physical_scaler.transform(nodes_df[all_physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])
    logger.debug("Skalierung auf physikalische und geografische Spalten angewendet.")

    # Positionskodierung hinzufügen
    nodes_df = add_positional_encoding(nodes_df, geo_columns)
    logger.debug("Positionskodierung zu geografischen Spalten hinzugefügt.")

    # Knotenfeatures erstellen
    node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values
    logger.debug("Knotenfeatures erstellt.")

    # Edge-Spalten nach One-Hot-Encoding aktualisieren
    continuous_edge_columns = ['RORL', 'DM', 'RAISE']
    one_hot_edge_columns = list(edges_df.filter(like='ROHRTYP').columns)
    edge_columns = continuous_edge_columns + one_hot_edge_columns

    # Sicherstellen, dass alle Edge-Attribute numerisch sind
    edges_df[edge_columns] = edges_df[edge_columns].apply(pd.to_numeric, errors='coerce')

    # Fehlende Werte in One-Hot-Encoder-Spalten auffüllen
    edges_df[one_hot_edge_columns] = edges_df[one_hot_edge_columns].fillna(0)

    # Skalierung nur auf kontinuierliche Edge-Attribute anwenden
    edges_df[continuous_edge_columns] = edge_scaler.transform(edges_df[continuous_edge_columns])
    logger.debug("Kontinuierliche Edge-Attribute skaliert.")

    # Edge-Attribute kombinieren
    edge_attributes = edges_df[edge_columns].values

    # Überprüfe, ob edge_attributes numerisch sind
    if not np.issubdtype(edge_attributes.dtype, np.number):
        logger.error(f"edge_attributes hat falschen Datentyp: {edge_attributes.dtype}")
        raise ValueError("edge_attributes müssen numerisch sein.")

    # In Tensoren konvertieren
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    logger.debug("PyTorch Geometric Data Objekt erstellt.")
    return data, edges_df

# Hauptfunktion zur Durchführung der Vorhersage
def main():
    # Pfad zur Konfigurationsdatei
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Zwei Ebenen hoch
    config_file = os.path.join(project_root, 'config', 'config.yaml')  # Pfad zur config.yml

    # Laden der Konfiguration
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Konfigurationsdatei erfolgreich geladen.")
    except FileNotFoundError:
        logger.error(f"Konfigurationsdatei nicht gefunden: {config_file}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Fehler beim Parsen der Konfigurationsdatei: {e}")
        return

    # Gerät einstellen (CPU oder GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Verwende Gerät: {device}")

    # Pfade zu den gespeicherten Modellen und Skalern
    models_dir = os.path.join(project_root, 'results', 'models')

    physical_scaler_path = os.path.join(models_dir, 'physical_scaler.pkl')
    geo_scaler_path = os.path.join(models_dir, 'geo_scaler.pkl')
    edge_scaler_path = os.path.join(models_dir, 'edge_scaler.pkl')
    model_path = os.path.join(models_dir, 'edge_gat_model_classification.pth')

    # Skalierer laden
    try:
        physical_scaler = joblib.load(physical_scaler_path)
        geo_scaler = joblib.load(geo_scaler_path)
        edge_scaler = joblib.load(edge_scaler_path)
        logger.info("Skalierer erfolgreich geladen.")
    except FileNotFoundError as e:
        logger.error(f"Skalierer-Datei nicht gefunden: {e}")
        return
    except Exception as e:
        logger.error(f"Fehler beim Laden der Skalierer: {e}")
        return

    # Liste der Knoten aus der Konfiguration laden
    included_nodes = config['nodes']['included_nodes']
    zfluss_wl_nodes = config['nodes']['zfluss_wl_nodes']

    # Definiere die benötigten Spalten
    adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    additional_physical_columns = ['ZUFLUSS_WL']
    all_physical_columns = adjusted_physical_columns + additional_physical_columns
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']

    # Pfad zum neuen Datensatzordner
    new_data_dir = config['paths']['folder_path_data_predicted']

    # Suche nach allen Node- und Pipes-Dateien
    node_files = glob.glob(os.path.join(new_data_dir, '*_Node.csv'))
    edge_files = glob.glob(os.path.join(new_data_dir, '*_Pipes.csv'))

    # Erstelle ein Mapping von Valve-IDs zu Node- und Edge-Dateien
    valve_files = {}
    for node_file in node_files:
        # Extrahiere die Valve-ID aus dem Dateinamen
        base_name = os.path.basename(node_file)
        valve_id = base_name.split('_Node.csv')[0]
        corresponding_edge_file = os.path.join(new_data_dir, f"{valve_id}_Pipes.csv")
        if corresponding_edge_file in edge_files:
            valve_files[valve_id] = (node_file, corresponding_edge_file)
        else:
            logger.warning(f"Keine entsprechende Pipes-Datei für {node_file} gefunden.")

    if not valve_files:
        logger.error("Keine passenden Node- und Pipes-Dateien gefunden. Beende das Programm.")
        return

    # Verwende das erste Valve-Paar, um die Anzahl der Features zu bestimmen
    first_valve_id, (first_node_file, first_edge_file) = next(iter(valve_files.items()))
    nodes_df = pd.read_csv(first_node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(first_edge_file, delimiter=';', decimal='.')

    # Vorverarbeite die Daten, um die Feature-Anzahl zu bestimmen
    data, _ = process_new_data(
        nodes_df, edges_df, included_nodes, zfluss_wl_nodes,
        physical_scaler, geo_scaler, edge_scaler,
        adjusted_physical_columns, additional_physical_columns, geo_columns
    )

    num_node_features = data.x.shape[1]
    num_edge_features = data.edge_attr.shape[1]

    # Modell initialisieren
    model = EdgeGAT(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=64,
        dropout=0.15
    ).to(device)

    # Modellzustand laden
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Modell erfolgreich geladen.")
    except FileNotFoundError:
        logger.error(f"Modell-Datei nicht gefunden: {model_path}")
        return
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        return

    # Iteriere über alle Valve-Dateipaaren und mache Vorhersagen
    for valve_id, (node_file, edge_file) in valve_files.items():
        logger.info(f"Verarbeite Valve: {valve_id}")
        try:
            # Lade die neuen Daten
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')

            # Vorverarbeite die neuen Daten
            data, processed_edges_df = process_new_data(
                nodes_df, edges_df, included_nodes, zfluss_wl_nodes,
                physical_scaler, geo_scaler, edge_scaler,
                adjusted_physical_columns, additional_physical_columns, geo_columns
            )

            # Überprüfe, ob die Feature-Anzahl übereinstimmt
            if data.x.shape[1] != num_node_features or data.edge_attr.shape[1] != num_edge_features:
                logger.error(f"Feature-Anzahl stimmt nicht überein für Valve {valve_id}. Überspringe.")
                continue

            # Erstelle das Data-Objekt und verschiebe es auf das Gerät
            data = data.to(device)

            # Führe die Vorhersage durch
            with torch.no_grad():
                logits = model(data)
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).cpu().numpy()

            # Füge Vorhersagen zu den Kantendaten hinzu
            processed_edges_df['prediction'] = predictions
            processed_edges_df['probability'] = probs.cpu().numpy()

            # Definiere den Ausgabepfad
            folder_path_data_results = config['paths']['folder_path_data_results']
            output_file = os.path.join(folder_path_data_results, f"{valve_id}_predictions.csv")

            # Speichere die Vorhersagen
            processed_edges_df.to_csv(output_file, sep=';', index=False)
            logger.info(f"Vorhersagen für Valve {valve_id} gespeichert unter {output_file}")

            # Optional: Ergebnisse anzeigen
            print(processed_edges_df[['ANFNAM', 'ENDNAM', 'prediction', 'probability']])

        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von Valve {valve_id}: {e}")
            continue

    logger.info("Alle Vorhersagen abgeschlossen.")


if __name__ == "__main__":
    main()
