import os
import pandas as pd
import torch
import numpy as np
import yaml
import joblib
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import logging
import glob
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

# ----------------------- Logging-Konfiguration -----------------------
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Konsole Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Datei Handler
    fh = logging.FileHandler('predictions.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

setup_logging()
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------

# --------------------------- DataModule ------------------------------
class DataModule:
    def __init__(self, directory, included_nodes, zfluss_wl_nodes, num_valves=100,
                 physical_scaler=None, geo_scaler=None, edge_scaler=None):
        self.directory = directory
        self.included_nodes = included_nodes
        self.zfluss_wl_nodes = zfluss_wl_nodes
        self.num_valves = num_valves
        self.physical_scaler = physical_scaler if physical_scaler is not None else StandardScaler()
        self.geo_scaler = geo_scaler if geo_scaler is not None else MinMaxScaler()
        self.edge_scaler = edge_scaler if edge_scaler is not None else StandardScaler()
        self.datasets = []
        self.geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
        self.adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
        self.additional_physical_columns = ['ZUFLUSS_WL']
        self.all_physical_columns = self.adjusted_physical_columns + self.additional_physical_columns

    def add_positional_encoding(self, df, columns, max_value=10000):
        for col in columns:
            df[f'{col}_sin'] = np.sin(df[col] * (2 * np.pi / max_value))
            df[f'{col}_cos'] = np.cos(df[col] * (2 * np.pi / max_value))
        return df

    def graph_based_imputation(self, df, edge_index, feature_name):
        node_values = df[feature_name].values
        missing_mask = np.isnan(node_values)
        adjacency = {i: [] for i in range(len(df))}
        for src, dst in edge_index.T:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
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

    def load_data(self, node_file, edge_file):
        try:
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
            logger.debug(f"Dateien geladen: {node_file}, {edge_file}.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Dateien {node_file} oder {edge_file}: {e}")
            raise e

        required_node_columns = ['KNAM', 'XRECHTS', 'YHOCH', 'GEOH'] + self.adjusted_physical_columns + ['ZUFLUSS_WL']
        for col in required_node_columns:
            if col not in nodes_df.columns:
                logger.debug(f"Spalte {col} fehlt in {node_file}.")
                raise ValueError(f"Spalte {col} fehlt in {node_file}.")

        required_edge_columns = ['ANFNAM', 'ENDNAM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'ROHRTYP', 'RORL', 'DM', 'RAISE']
        for col in required_edge_columns:
            if col not in edges_df.columns:
                logger.debug(f"Spalte {col} fehlt in {edge_file}.")
                raise ValueError(f"Spalte {col} fehlt in {edge_file}.")

        if 'RAU' in edges_df.columns:
            edges_df = edges_df.drop(columns=['RAU'])
            logger.debug("'RAU' Spalte gefunden und aus den Kantendaten entfernt.")

        nodes_df['KNAM'] = nodes_df['KNAM'].astype(str).str.strip().str.lower()
        edges_df['ANFNAM'] = edges_df['ANFNAM'].astype(str).str.strip().str.lower()
        edges_df['ENDNAM'] = edges_df['ENDNAM'].astype(str).str.strip().str.lower()

        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        nodes_df['node_idx'] = nodes_df['KNAM'].map(node_mapping)
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

        missing_anfnr = edges_df['ANFNR'].isnull()
        missing_endnr = edges_df['ENDNR'].isnull()

        if missing_anfnr.any() or missing_endnr.any():
            missing_anfnam = edges_df.loc[missing_anfnr, 'ANFNAM'].unique()
            missing_endnam = edges_df.loc[missing_endnr, 'ENDNAM'].unique()
            logger.error(f"Fehlende Knotenindizes für ANFNAMs: {missing_anfnam}, ENDNAMs: {missing_endnam}")
            raise ValueError("Kantendaten enthalten Knoten, die in den Knotendaten nicht gefunden wurden.")

        edges_df['ANFNR'] = edges_df['ANFNR'].astype(int)
        edges_df['ENDNR'] = edges_df['ENDNR'].astype(int)

        edges_df['edge_id'] = edges_df['ANFNAM'] + '_' + edges_df['ENDNAM']

        edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP', dtype=float)
        logger.debug("One-Hot-Encoding für 'ROHRTYP' durchgeführt.")

        edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

        edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']].astype(float)
        logger.debug("Relevante Kantenspalten in float konvertiert.")

        epsilon = 1e-6
        target_condition = (
            (edges_df['FLUSS_WL'].abs() < epsilon) |
            (edges_df['FLUSS_WOL'].abs() < epsilon) |
            (edges_df['VM_WL'].abs() < epsilon) |
            (edges_df['VM_WOL'].abs() < epsilon)
        )
        y = torch.tensor(target_condition.astype(float).values, dtype=torch.float)
        logger.debug("Zielvariable basierend auf Flussbedingungen erstellt.")

        nodes_df['Included'] = nodes_df['KNAM'].isin([n.lower() for n in self.included_nodes])
        for col in self.adjusted_physical_columns:
            nodes_df.loc[~nodes_df['Included'], col] = np.nan
            logger.debug(f"{col} auf NaN gesetzt für Knoten, die nicht enthalten sind.")

        nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in self.zfluss_wl_nodes] else np.nan,
            axis=1
        )
        logger.debug("'ZUFLUSS_WL' für bestimmte Knoten behandelt.")

        for col in self.all_physical_columns:
            nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)
            logger.debug(f"Fehlender Indikator für {col} hinzugefügt.")

        nodes_df = self.graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

        imputer = KNNImputer(n_neighbors=5)
        nodes_df[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df[self.adjusted_physical_columns])
        logger.debug("KNN-Imputation für angepasste physikalische Spalten durchgeführt.")

        nodes_df = nodes_df.drop(columns=['Included'])
        logger.debug("Hilfsspalte 'Included' aus Knotendaten entfernt.")

        nodes_df[self.all_physical_columns] = self.physical_scaler.transform(nodes_df[self.all_physical_columns])
        nodes_df[self.geo_columns] = self.geo_scaler.transform(nodes_df[self.geo_columns])
        logger.debug("Skalierung auf physikalische und geografische Spalten angewendet.")

        nodes_df = self.add_positional_encoding(nodes_df, self.geo_columns)
        logger.debug("Positionscodierung zu geografischen Spalten hinzugefügt.")

        node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values
        logger.debug("Knotenfeatures erstellt.")

        continuous_edge_columns = ['RORL', 'DM', 'RAISE']
        one_hot_edge_columns = list(edges_df.filter(like='ROHRTYP').columns)
        edge_columns = continuous_edge_columns + one_hot_edge_columns

        edges_df[edge_columns] = edges_df[edge_columns].apply(pd.to_numeric, errors='coerce')
        edges_df[one_hot_edge_columns] = edges_df[one_hot_edge_columns].fillna(0)
        edges_df[continuous_edge_columns] = self.edge_scaler.transform(edges_df[continuous_edge_columns])
        logger.debug("Skalierung auf kontinuierliche Kantenattribute angewendet.")

        edge_attributes = edges_df[edge_columns].values
        logger.debug("Kantenattribute kombiniert.")

        try:
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        except Exception as e:
            logger.error(f"Fehler beim Konvertieren zu Tensoren: {e}")
            logger.debug(f"node_features dtype: {node_features.dtype}")
            logger.debug(f"edge_index dtype: {edge_index.dtype}")
            logger.debug(f"edge_attr dtype: {edge_attributes.dtype}")
            raise e

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        logger.debug("PyTorch Geometric Data Objekt erstellt.")

        data.edge_ids = edges_df['edge_id'].values  # Dies ist ein NumPy-Array

        return data, edges_df

    def load_all_data(self):
        # Implementieren Sie das Laden aller Daten, falls erforderlich
        pass

    def get_loaders(self, val_size=0.25, test_size=0.2, random_state=42):
        if not self.datasets:
            logger.error("Keine Datensätze verfügbar.")
            return None, None, None

        # Aufteilen der Daten in Trainings-, Validierungs- und Testmengen
        train_val_data, test_data = train_test_split(
            self.datasets, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)  # Anpassung des Validierungsverhältnisses
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_ratio, random_state=random_state
        )
        logger.info("Daten in Trainings-, Validierungs- und Testmengen aufgeteilt.")

        # Überprüfen auf positive Beispiele
        train_positive = sum([data.y.sum().item() for data in train_data])
        val_positive = sum([data.y.sum().item() for data in val_data])
        test_positive = sum([data.y.sum().item() for data in test_data])

        if train_positive == 0 or val_positive == 0 or test_positive == 0:
            logger.warning("Keine positiven Beispiele in einem der Datensätze.")

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        logger.info("DataLoader für Trainings-, Validierungs- und Testmengen erstellt.")

        return train_loader, val_loader, test_loader
# ---------------------------------------------------------------------

# --------------------------- EdgeGAT Modell --------------------------
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
        logger.debug("Berechnete edge_logits.")

        return edge_logits  # Gibt Logits zurück
# ---------------------------------------------------------------------

# ----------------------- Weitere Funktionen --------------------------
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
        logger.error(f"Fehlende Knotenindizes für ANFNAMs: {missing_anfnam}, ENDNAMs: {missing_endnam}")
        raise ValueError("Kantendaten enthalten Knoten, die in den Knotendaten nicht gefunden wurden.")

    # Konvertiere Indizes zu Integer
    edges_df['ANFNR'] = edges_df['ANFNR'].astype(int)
    edges_df['ENDNR'] = edges_df['ENDNR'].astype(int)

    # Erstellen eines eindeutigen Kantenidentifikators
    edges_df['edge_id'] = edges_df['ANFNAM'] + '_' + edges_df['ENDNAM']

    # One-Hot-Encoding für 'ROHRTYP' mit dtype=float
    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP', dtype=float)
    logger.debug("One-Hot-Encoding für 'ROHRTYP' durchgeführt.")

    edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

    # Sicherstellen, dass relevante Spalten numerisch sind
    edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']].astype(float)
    logger.debug("Relevante Kantenspalten in float konvertiert.")

    # Zielvariable erstellen
    # Hier geht es um Klassifikation: ob FLUSS_WL, FLUSS_WOL, VM_WL oder VM_WOL nahe Null sind
    epsilon = 1e-6
    target_condition = (
        (edges_df['FLUSS_WL'].abs() < epsilon) |
        (edges_df['FLUSS_WOL'].abs() < epsilon) |
        (edges_df['VM_WL'].abs() < epsilon) |
        (edges_df['VM_WOL'].abs() < epsilon)
    )
    y = torch.tensor(target_condition.astype(float).values, dtype=torch.float)
    logger.debug("Zielvariable basierend auf Flussbedingungen erstellt.")

    # Knotenattribute anpassen
    # Werte der adjusted_physical_columns auf NaN setzen für Knoten, die nicht in included_nodes sind
    nodes_df['Included'] = nodes_df['KNAM'].isin([n.lower() for n in included_nodes])
    for col in adjusted_physical_columns:
        nodes_df.loc[~nodes_df['Included'], col] = np.nan
        logger.debug(f"{col} auf NaN gesetzt für Knoten, die nicht enthalten sind.")

    # ZUFLUSS_WL nur für bestimmte Knoten behandeln
    nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
        lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in zfluss_wl_nodes] else np.nan,
        axis=1
    )
    logger.debug("'ZUFLUSS_WL' für bestimmte Knoten behandelt.")

    # Indikatorspalten für fehlende Werte hinzufügen
    for col in adjusted_physical_columns + additional_physical_columns:
        nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)
        logger.debug(f"Fehlender Indikator für {col} hinzugefügt.")

    # Graph-basierte Imputation für fehlende ZUFLUSS_WL-Werte
    data_module_instance = DataModule('', included_nodes, zfluss_wl_nodes)  # Temporäre Instanz für die Methode
    nodes_df = data_module_instance.graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

    # Fehlende Werte für andere physikalische Spalten mit KNN Imputer behandeln
    imputer = KNNImputer(n_neighbors=5)
    nodes_df[adjusted_physical_columns] = imputer.fit_transform(nodes_df[adjusted_physical_columns])
    logger.debug("KNN-Imputation für angepasste physikalische Spalten durchgeführt.")

    # Hilfsspalte entfernen
    nodes_df = nodes_df.drop(columns=['Included'])
    logger.debug("Hilfsspalte 'Included' aus Knotendaten entfernt.")

    # Skalierung auf Knoteneigenschaften anwenden
    nodes_df[adjusted_physical_columns + additional_physical_columns] = physical_scaler.transform(
        nodes_df[adjusted_physical_columns + additional_physical_columns]
    )
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])
    logger.debug("Skalierung auf physikalische und geografische Spalten angewendet.")

    # Positionscodierung hinzufügen
    nodes_df = data_module_instance.add_positional_encoding(nodes_df, geo_columns)
    logger.debug("Positionscodierung zu geografischen Spalten hinzugefügt.")

    # Knoteneigenschaften erstellen
    node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values
    logger.debug("Knotenfeatures erstellt.")

    # Aktualisiere edge_columns nach One-Hot-Encoding
    continuous_edge_columns = ['RORL', 'DM', 'RAISE']
    one_hot_edge_columns = list(edges_df.filter(like='ROHRTYP').columns)
    edge_columns = continuous_edge_columns + one_hot_edge_columns

    # Sicherstellen, dass alle Kantenattribute numerisch sind
    edges_df[edge_columns] = edges_df[edge_columns].apply(pd.to_numeric, errors='coerce')

    # Fehlende Werte in One-Hot-encodierten Spalten füllen (falls vorhanden)
    edges_df[one_hot_edge_columns] = edges_df[one_hot_edge_columns].fillna(0)

    # Skalierung nur auf kontinuierliche Kantenattribute anwenden
    edge_attributes_continuous = edges_df[continuous_edge_columns].values
    edge_attributes_continuous_scaled = edge_scaler.transform(edge_attributes_continuous)
    edges_df[continuous_edge_columns] = edge_attributes_continuous_scaled
    logger.debug("Skalierung auf kontinuierliche Kantenattribute angewendet.")

    # Kombiniere skalierte kontinuierliche Attribute und unskalierte One-Hot-encodierte Attribute
    edge_attributes = edges_df[edge_columns].values
    logger.debug("Kantenattribute kombiniert.")

    # Konvertiere zu Tensoren
    try:
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    except Exception as e:
        logger.error(f"Fehler beim Konvertieren zu Tensoren: {e}")
        logger.debug(f"node_features dtype: {node_features.dtype}")
        logger.debug(f"edge_index dtype: {edge_index.dtype}")
        logger.debug(f"edge_attr dtype: {edge_attributes.dtype}")
        raise e

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    logger.debug("PyTorch Geometric Data Objekt erstellt.")

    # Speichern der Kantenidentifikatoren im Data-Objekt
    data.edge_ids = edges_df['edge_id'].values  # Dies ist ein NumPy-Array

    return data, edges_df

def load_threshold(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return float(data['optimal_threshold'])

# ---------------------------------------------------------------------

# --------------------------- Main Funktion ---------------------------
def main():

    # Pfad zur Konfigurationsdatei relativ zum Projektstammverzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # Zwei Ebenen hoch
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

    directory = config['paths']['folder_path_data']

    # Liste der Knoten mit verfügbaren Messdaten
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

    # Initialisiere DataModule (ohne Skalierer-Anpassung)
    data_module = DataModule(directory, included_nodes, zfluss_wl_nodes)
    # Skalierer sollten bereits angepasst und geladen werden

    # Laden der Skalierer
    models_dir = os.path.join(project_root, 'results', 'models')
    physical_scaler_path = os.path.join(models_dir, 'physical_scaler.pkl')
    geo_scaler_path = os.path.join(models_dir, 'geo_scaler.pkl')
    edge_scaler_path = os.path.join(models_dir, 'edge_scaler.pkl')
    threshold_path = os.path.join(models_dir, 'optimal_threshold.json')

    # Laden der Skalierer
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

    # Initialisiere DataModule mit den angepassten Skalierern
    data_module = DataModule(
        directory, included_nodes, zfluss_wl_nodes,
        physical_scaler=physical_scaler,
        geo_scaler=geo_scaler,
        edge_scaler=edge_scaler
    )

    # Laden des optimalen Schwellenwerts
    if os.path.exists(threshold_path):
        optimal_threshold = load_threshold(threshold_path)
        logger.info(f"Geladener optimaler Schwellenwert: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5  # Fallback
        logger.warning(f"Schwellenwert-Datei nicht gefunden. Verwende Standard-Schwellenwert: {optimal_threshold}")

    # Definiere die Anzahl der Features
    try:
        data, _ = data_module.load_data(first_node_file, first_edge_file)
        num_node_features = data.x.shape[1]
        num_edge_features = data.edge_attr.shape[1]
    except Exception as e:
        logger.error(f"Fehler beim Laden der Features aus {first_valve_id}: {e}")
        return

    # Gerät einstellen (CPU oder GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Verwende Gerät: {device}")

    # Modell initialisieren
    model = EdgeGAT(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=64,
        dropout=0.15
    ).to(device)

    # Modellzustand laden
    model_path = os.path.join(models_dir, 'edge_gat_model_classification.pth')
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
                predictions = (probs >= optimal_threshold).cpu().numpy()

            # Füge Vorhersagen zu den Kantendaten hinzu
            processed_edges_df['prediction'] = predictions
            processed_edges_df['probability'] = probs.cpu().numpy()

            # Definiere den Ausgabepfad
            folder_path_data_results = config['paths']['folder_path_data_results']
            if not os.path.exists(folder_path_data_results):
                os.makedirs(folder_path_data_results)
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
# ---------------------------------------------------------------------
