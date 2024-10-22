import os
import pandas as pd
import torch
import numpy as np
import yaml
import joblib
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, auc
import logging
import csv
import json

# Initialisiere das Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DataModule Class
class DataModule:
    def __init__(self, directory, included_nodes, zfluss_wl_nodes, num_valves=100):
        self.directory = directory
        self.included_nodes = included_nodes
        self.zfluss_wl_nodes = zfluss_wl_nodes
        self.num_valves = num_valves
        self.physical_scaler = StandardScaler()
        self.geo_scaler = MinMaxScaler()
        self.edge_scaler = StandardScaler()
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
        # Create adjacency list
        adjacency = {i: [] for i in range(len(df))}
        for src, dst in edge_index.T:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
        # Iterate over missing values
        for idx in np.where(missing_mask)[0]:
            neighbors = adjacency[idx]
            neighbor_values = [node_values[n] for n in neighbors if not np.isnan(node_values[n])]
            if neighbor_values:
                node_values[idx] = np.mean(neighbor_values)
            else:
                node_values[idx] = np.nanmean(node_values)
        df[feature_name] = node_values
        logger.debug(f"Graph-basierte Imputation für Merkmal '{feature_name}' durchgeführt.")
        return df

    def load_data(self, node_file, edge_file):
        try:
            # Daten laden
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
            logger.debug(f"Dateien geladen: {node_file}, {edge_file}.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Dateien {node_file} oder {edge_file}: {e}")
            raise e

        # Überprüfen, ob erforderliche Spalten vorhanden sind
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

        # 'RAU' Spalte entfernen, falls vorhanden
        if 'RAU' in edges_df.columns:
            edges_df = edges_df.drop(columns=['RAU'])
            logger.debug("'RAU' Spalte gefunden und aus den Kantendaten entfernt.")

        # Knoten- und Kantenbezeichnungen bereinigen
        nodes_df['KNAM'] = nodes_df['KNAM'].astype(str).str.strip().str.lower()
        edges_df['ANFNAM'] = edges_df['ANFNAM'].astype(str).str.strip().str.lower()
        edges_df['ENDNAM'] = edges_df['ENDNAM'].astype(str).str.strip().str.lower()

        # Knoten zu Indizes zuordnen
        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        nodes_df['node_idx'] = nodes_df['KNAM'].map(node_mapping)
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

        # Überprüfen auf fehlende Knotenindizes in Kanten
        missing_anfnr = edges_df['ANFNR'].isnull()
        missing_endnr = edges_df['ENDNR'].isnull()

        if missing_anfnr.any() or missing_endnr.any():
            missing_anfnam = edges_df.loc[missing_anfnr, 'ANFNAM'].unique()
            missing_endnam = edges_df.loc[missing_endnr, 'ENDNAM'].unique()
            logger.error(f"Fehlende Knotenindizes für ANFNAMs: {missing_anfnam}, ENDNAMs: {missing_endnam}")
            raise ValueError("Kantendaten enthalten Knoten, die in den Knotendaten nicht gefunden wurden.")

        # Indizes zu Ganzzahlen konvertieren
        edges_df['ANFNR'] = edges_df['ANFNR'].astype(int)
        edges_df['ENDNR'] = edges_df['ENDNR'].astype(int)

        # Erstellen eines eindeutigen Kantenidentifikators
        edges_df['edge_id'] = edges_df['ANFNAM'] + '_' + edges_df['ENDNAM']

        # One-Hot-Encoding für 'ROHRTYP'
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
        nodes_df['Included'] = nodes_df['KNAM'].isin([n.lower() for n in self.included_nodes])
        for col in self.adjusted_physical_columns:
            nodes_df.loc[~nodes_df['Included'], col] = np.nan
            logger.debug(f"{col} auf NaN gesetzt für Knoten, die nicht enthalten sind.")

        # ZUFLUSS_WL nur für bestimmte Knoten behandeln
        nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in self.zfluss_wl_nodes] else np.nan,
            axis=1
        )
        logger.debug("'ZUFLUSS_WL' für bestimmte Knoten behandelt.")

        # Indikatorspalten für fehlende Werte hinzufügen
        for col in self.all_physical_columns:
            nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)
            logger.debug(f"Fehlender Indikator für {col} hinzugefügt.")

        # Graph-basierte Imputation für fehlende ZUFLUSS_WL-Werte
        nodes_df = self.graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

        # Fehlende Werte für andere physikalische Spalten mit KNN Imputer behandeln
        imputer = KNNImputer(n_neighbors=5)
        nodes_df[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df[self.adjusted_physical_columns])
        logger.debug("KNN-Imputation für angepasste physikalische Spalten durchgeführt.")

        # Hilfsspalte entfernen
        nodes_df = nodes_df.drop(columns=['Included'])
        logger.debug("Hilfsspalte 'Included' aus Knotendaten entfernt.")

        # Skalierung auf Knoteneigenschaften anwenden
        nodes_df[self.all_physical_columns] = self.physical_scaler.transform(nodes_df[self.all_physical_columns])
        nodes_df[self.geo_columns] = self.geo_scaler.transform(nodes_df[self.geo_columns])
        logger.debug("Skalierung auf physikalische und geografische Spalten angewendet.")

        # Positionscodierung hinzufügen
        nodes_df = self.add_positional_encoding(nodes_df, self.geo_columns)
        logger.debug("Positionscodierung zu geografischen Spalten hinzugefügt.")

        # Knoteneigenschaften erstellen
        node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values

        # Aktualisiere edge_columns nach One-Hot-Encoding
        continuous_edge_columns = ['RORL', 'DM', 'RAISE']
        one_hot_edge_columns = list(edges_df.filter(like='ROHRTYP').columns)
        edge_columns = continuous_edge_columns + one_hot_edge_columns

        # Sicherstellen, dass alle Kantenattribute numerisch sind
        edges_df[edge_columns] = edges_df[edge_columns].apply(pd.to_numeric, errors='coerce')

        # Fehlende Werte in One-Hot-encodierten Spalten füllen (falls vorhanden)
        edges_df[one_hot_edge_columns] = edges_df[one_hot_edge_columns].fillna(0)

        # Skalierung nur auf kontinuierliche Kantenattribute anwenden
        edges_df[continuous_edge_columns] = self.edge_scaler.transform(edges_df[continuous_edge_columns])
        logger.debug("Skalierung auf kontinuierliche Kantenattribute angewendet.")

        # Kombiniere skalierte kontinuierliche Attribute und unskalierte One-Hot-encodierte Attribute
        edge_attributes = edges_df[edge_columns].values

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

        # Speichern der Kantenidentifikatoren im Data-Objekt
        data.edge_ids = edges_df['edge_id'].values  # Dies ist ein NumPy-Array

        logger.debug("PyTorch Geometric Data Objekt erstellt.")
        return data

    def fit_scalers(self):
        # Hier nimmst du den ersten Datensatz zum Anpassen der Skalierer
        node_file_first = os.path.join(self.directory, 'SyntheticData-Spechbach_Valve_1_combined_Node.csv')
        edge_file_first = os.path.join(self.directory, 'SyntheticData-Spechbach_Valve_1_combined_Pipes.csv')

        try:
            nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
            edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')
            logger.debug(f"Dateien zum Anpassen der Skalierer geladen: {node_file_first}, {edge_file_first}.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Dateien {node_file_first} oder {edge_file_first} zum Anpassen der Skalierer: {e}")
            raise e

        # 'RAU' Spalte entfernen, falls vorhanden
        if 'RAU' in edges_df_first.columns:
            edges_df_first = edges_df_first.drop(columns=['RAU'])
            logger.debug("'RAU' Spalte gefunden und aus Kantendaten entfernt.")

        # Knoten- und Kantenbezeichnungen bereinigen
        nodes_df_first['KNAM'] = nodes_df_first['KNAM'].astype(str).str.strip().str.lower()
        edges_df_first['ANFNAM'] = edges_df_first['ANFNAM'].astype(str).str.strip().str.lower()
        edges_df_first['ENDNAM'] = edges_df_first['ENDNAM'].astype(str).str.strip().str.lower()

        # Setze adjusted_physical_columns auf NaN für Knoten, die nicht in included_nodes sind
        nodes_df_first['Included'] = nodes_df_first['KNAM'].isin([n.lower() for n in self.included_nodes])
        for col in self.adjusted_physical_columns:
            nodes_df_first.loc[~nodes_df_first['Included'], col] = np.nan
            logger.debug(f"{col} auf NaN gesetzt für Knoten, die nicht enthalten sind (Skalierer-Anpassung).")

        # ZUFLUSS_WL nur für bestimmte Knoten behandeln
        nodes_df_first['ZUFLUSS_WL'] = nodes_df_first.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in self.zfluss_wl_nodes] else np.nan,
            axis=1
        )
        logger.debug("'ZUFLUSS_WL' für bestimmte Knoten behandelt (Skalierer-Anpassung).")

        # Indikatorspalten für fehlende Werte hinzufügen
        for col in self.all_physical_columns:
            nodes_df_first[f'{col}_missing'] = nodes_df_first[col].isna().astype(float)
            logger.debug(f"Fehlender Indikator für {col} hinzugefügt (Skalierer-Anpassung).")

        # Graph-basierte Imputation für fehlende ZUFLUSS_WL-Werte
        node_mapping_first = {name: idx for idx, name in enumerate(nodes_df_first['KNAM'])}
        edges_df_first['ANFNR'] = edges_df_first['ANFNAM'].map(node_mapping_first)
        edges_df_first['ENDNR'] = edges_df_first['ENDNAM'].map(node_mapping_first)
        edge_index_first = edges_df_first[['ANFNR', 'ENDNR']].values.T

        nodes_df_first = self.graph_based_imputation(nodes_df_first, edge_index_first, 'ZUFLUSS_WL')

        # Fehlende Werte für andere physikalische Spalten mit KNN Imputer behandeln
        imputer = KNNImputer(n_neighbors=5)
        nodes_df_first[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df_first[self.adjusted_physical_columns])
        logger.debug("KNN-Imputation für angepasste physikalische Spalten durchgeführt (Skalierer-Anpassung).")

        # Hilfsspalte entfernen
        nodes_df_first = nodes_df_first.drop(columns=['Included'])
        logger.debug("Hilfsspalte 'Included' aus Knotendaten entfernt (Skalierer-Anpassung).")

        # Skalierer an Knoteneigenschaften anpassen
        self.physical_scaler.fit(nodes_df_first[self.all_physical_columns])
        self.geo_scaler.fit(nodes_df_first[self.geo_columns])
        logger.debug("Physikalische und geografische Skalierer angepasst.")

        # One-Hot-Encoding für 'ROHRTYP' durchführen
        edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP', dtype=float)
        logger.debug("One-Hot-Encoding für 'ROHRTYP' durchgeführt (Skalierer-Anpassung).")

        # Aktualisiere edge_columns nach One-Hot-Encoding
        continuous_edge_columns = ['RORL', 'DM', 'RAISE']
        one_hot_edge_columns = list(edges_df_first.filter(like='ROHRTYP').columns)
        edge_columns = continuous_edge_columns + one_hot_edge_columns

        # Sicherstellen, dass alle Kantenattribute numerisch sind
        edges_df_first[edge_columns] = edges_df_first[edge_columns].apply(pd.to_numeric, errors='coerce')

        # Fehlende Werte in One-Hot-encodierten Spalten füllen (falls vorhanden)
        edges_df_first[one_hot_edge_columns] = edges_df_first[one_hot_edge_columns].fillna(0)

        # Skalierer an kontinuierliche Kantenattribute anpassen
        self.edge_scaler.fit(edges_df_first[continuous_edge_columns])
        logger.debug("Skalierer an kontinuierliche Kantenattribute angepasst.")

    def load_all_data(self):
        self.fit_scalers()
        logger.info("Beginne mit dem Laden aller Datensätze.")
        for i in range(1, self.num_valves + 1):
            node_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Valve_{i}_combined_Node.csv')
            edge_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Valve_{i}_combined_Pipes.csv')
            try:
                data = self.load_data(node_file, edge_file)
                self.datasets.append(data)
                logger.debug(f'Datensatz {i} erfolgreich geladen.')
            except Exception as e:
                logger.error(f'Fehler beim Laden des Datensatzes {i}: {e}')
                continue
        logger.info("Alle Datensätze geladen.")

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

# GAT Model with Edge Prediction for Binary Classification
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

# Trainer Class
class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, num_epochs=300, patience=10, results_dir='results'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.results_dir = results_dir
        self.best_model_state = None
        self.logger = logging.getLogger(__name__)  # Eigener Logger für die Klasse

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        self.logger.debug(f"Training epoch loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        self.logger.debug(f"Validation epoch loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_monitoring_dataset(self, monitoring_loader, epoch):
        self.model.eval()
        total_loss = 0
        y_true_list = []
        y_pred_list = []
        edge_id_list = []

        with torch.no_grad():
            for batch in monitoring_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                y_true_list.extend(batch.y.cpu().numpy().flatten())
                y_pred_list.extend(probs.cpu().numpy().flatten())
                edge_id_list.extend(batch.edge_ids)

        avg_loss = total_loss / len(monitoring_loader)
        y_true = np.array(y_true_list).reshape(-1)
        y_pred = np.array(y_pred_list).reshape(-1)
        edge_ids_all = np.array(edge_id_list).reshape(-1)

        # Berechne Metriken
        accuracy = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc_score = 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_true': y_true,           # Hinzugefügt
            'y_pred_prob': y_pred        # Hinzugefügt
        }

        # Speichern der Vorhersagen pro Kante in einer CSV-Datei
        df_predictions = pd.DataFrame({
            'edge_id': edge_ids_all,
            'y_true': y_true,
            'y_pred_prob': y_pred,
            'y_pred_label': (y_pred >= 0.5).astype(int)
        })

        # CSV-Datei speichern
        csv_filename = f'monitoring_predictions_epoch_{epoch}.csv'
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        df_predictions.to_csv(csv_filepath, index=False)
        self.logger.info(f'Per-Kante-Vorhersagen für Epoche {epoch} gespeichert unter {csv_filepath}')

        return metrics

    def train_model(self, train_loader, val_loader, monitoring_loader=None):
        best_val_loss = float('inf')
        patience_counter = 0

        self.monitoring_results = []

        self.logger.info("Starte den Trainingsprozess.")
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            # Evaluierung auf dem Überwachungsdatensatz
            if monitoring_loader is not None:
                monitoring_metrics = self.evaluate_monitoring_dataset(monitoring_loader, epoch)
                self.monitoring_results.append((epoch, monitoring_metrics))
                self.logger.info(
                    f'Epoche {epoch:03d}, Überwachung - Verlust: {monitoring_metrics["loss"]:.4f}, '
                    f'Genauigkeit: {monitoring_metrics["accuracy"]:.4f}, AUC: {monitoring_metrics["auc_score"]:.4f}'
                )

            self.scheduler.step(val_loss)

            if epoch % 10 == 0:
                self.logger.info(
                    f'Epoche {epoch:03d}, Trainingsverlust: {train_loss:.4f}, Validierungsverlust: {val_loss:.4f}'
                )

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
                self.logger.debug(f"Epoche {epoch}: Verbesserter Validierungsverlust auf {val_loss:.4f}.")
            else:
                patience_counter += 1
                self.logger.debug(f"Epoche {epoch}: Keine Verbesserung des Validierungsverlusts.")

            if patience_counter >= self.patience:
                self.logger.info(f"Frühes Stoppen in Epoche {epoch}.")
                break

        # Lade das beste Modell
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Bestes Modell basierend auf dem Validierungsverlust geladen.")

        # Speichere den optimalen Schwellenwert basierend auf der letzten Überwachung
        if self.monitoring_results:
            last_epoch, last_metrics = self.monitoring_results[-1]
            optimal_threshold = self.find_optimal_threshold(last_metrics['y_true'], last_metrics['y_pred_prob'])
            threshold_path = os.path.join(self.results_dir, 'optimal_threshold.json')
            self.save_threshold(optimal_threshold, threshold_path)
            self.logger.info(f"Optimaler Schwellenwert {optimal_threshold:.4f} gespeichert unter {threshold_path}.")

    def find_optimal_threshold(self, y_true, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        self.logger.info(f"Optimaler Schwellenwert basierend auf ROC: {optimal_threshold:.4f}")
        return optimal_threshold

    def save_threshold(self, optimal_threshold, path):
        try:
            with open(path, 'w') as f:
                json.dump({'optimal_threshold': float(optimal_threshold)}, f)
            self.logger.info(f"Optimaler Schwellenwert {optimal_threshold:.4f} in {path} gespeichert.")
        except TypeError as e:
            self.logger.error(f"Fehler beim Speichern des Schwellenwerts: {e}")
            raise

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'Modell gespeichert unter: {path}')

# Evaluator Class
class Evaluator:
    def __init__(self, model, device, results_dir):
        self.model = model
        self.device = device
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)

    def test_model(self, loader):
        y_true, y_pred = self._test(loader)
        return y_true, y_pred

    def _test(self, loader):
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                y_true_list.append(batch.y.cpu().numpy())
                y_pred_list.append(probs.cpu().numpy())
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        logger.info("Modelltest abgeschlossen.")
        return y_true, y_pred

    def calculate_metrics(self, y_true, y_pred):
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            self.logger.warning("y_true enthält nur eine Klasse. ROC AUC Score kann nicht berechnet werden.")
            auc_score = None
            optimal_threshold = 0.5  # Fallback
        else:
            auc_score = roc_auc_score(y_true, y_pred)
            self.logger.info(f'ROC AUC Score: {auc_score:.4f}')
            optimal_threshold = self.find_optimal_threshold(y_true, y_pred)

        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)
        self.logger.info(f'Genauigkeit: {accuracy:.4f}')

        return accuracy, auc_score, y_pred_binary, optimal_threshold

    def find_optimal_threshold(self, y_true, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"Optimaler Schwellenwert basierend auf ROC: {optimal_threshold:.4f}")
        return optimal_threshold

    def save_threshold(self, optimal_threshold, path):
        with open(path, 'w') as f:
            json.dump({'optimal_threshold': optimal_threshold}, f)
        logger.info(f"Optimaler Schwellenwert {optimal_threshold:.4f} in {path} gespeichert.")

    def plot_metrics(self, y_true, y_pred):
        self.plot_roc_curve(y_true, y_pred)
        self.plot_precision_recall(y_true, y_pred)
        self.plot_confusion_matrix(y_true, (y_pred >= 0.5).astype(int))  # Optional anpassen

    def plot_roc_curve(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Kurve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
        logger.debug("ROC Kurve geplottet.")

    def plot_precision_recall(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_pred)
        plt.figure()
        plt.plot(recall, precision, label=f'Precision-Recall Kurve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Kurve')
        plt.legend(loc="lower left")
        plt.show()
        logger.debug("Precision-Recall Kurve geplottet.")

    def plot_confusion_matrix(self, y_true, y_pred_binary):
        cm = confusion_matrix(y_true, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Konfusionsmatrix')
        plt.show()
        logger.debug("Konfusionsmatrix geplottet.")

# Main Function
def main():
    # Pfad zur Konfigurationsdatei relativ zum Projektstammverzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Zwei Ebenen hoch
    config_file = os.path.join(project_root, 'config', 'config.yaml')  # Standardpfad zur config.yaml

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

    # Verwenden Sie den globalen logger, keine lokale Zuweisung
    # logger = logging.getLogger(__name__)  # Entfernen Sie diese Zeile

    # Liste der Knoten mit verfügbaren Messdaten
    included_nodes = config['nodes']['included_nodes']
    zfluss_wl_nodes = config['nodes']['zfluss_wl_nodes']

    # Initialisiere DataModule
    data_module = DataModule(directory, included_nodes, zfluss_wl_nodes, num_valves=3800)
    data_module.load_all_data()
    train_loader, val_loader, test_loader = data_module.get_loaders()

    if not train_loader or not val_loader or not test_loader:
        logger.error("DataLoader konnten nicht erstellt werden. Beende das Programm.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Verwende Gerät: {device}")

    model = EdgeGAT(
        num_node_features=data_module.datasets[0].x.shape[1],
        num_edge_features=data_module.datasets[0].edge_attr.shape[1],
        hidden_dim=64,
        dropout=0.15
    ).to(device)
    logger.info("EdgeGAT-Modell initialisiert.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    logger.info("AdamW-Optimierer initialisiert.")

    # Verlustfunktion mit Klassengewichtung basierend auf dem Trainingsdatensatz
    positive_samples = sum([data.y.sum().item() for data in train_loader.dataset])
    total_samples = sum([len(data.y) for data in train_loader.dataset])
    negative_samples = total_samples - positive_samples

    if positive_samples == 0:
        pos_weight_value = 1.0
        logger.warning("Keine positiven Beispiele im Trainingsdatensatz gefunden. Setze pos_weight auf 1.0.")
    else:
        pos_weight_value = negative_samples / positive_samples
        logger.info(f"Berechneter pos_weight: {pos_weight_value:.4f}")

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("BCEWithLogitsLoss mit pos_weight initialisiert.")

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    logger.info("ReduceLROnPlateau-Scheduler initialisiert.")

    # Definiere das Ergebnisverzeichnis für den Trainer
    results_dir = os.path.join(project_root, 'results', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialisiere Trainer
    trainer = Trainer(
        model, optimizer, criterion, scheduler, device,
        num_epochs=300, patience=10, results_dir=results_dir  # Übergebe results_dir
    )
    logger.info("Trainer initialisiert.")

    # Laden des Überwachungsdatensatzes
    monitoring_node_file = os.path.join(directory, 'SyntheticData-Spechbach_Valve_0_combined_Node.csv')
    monitoring_edge_file = os.path.join(directory, 'SyntheticData-Spechbach_Valve_0_combined_Pipes.csv')

    try:
        monitoring_data = data_module.load_data(monitoring_node_file, monitoring_edge_file)
        monitoring_loader = DataLoader([monitoring_data], batch_size=1, shuffle=False)
        logger.info("Überwachungsdatensatz erfolgreich geladen.")
    except Exception as e:
        logger.error(f'Fehler beim Laden des Überwachungsdatensatzes: {e}')
        monitoring_loader = None

    # Start des Trainings mit Übergabe des monitoring_loader
    trainer.train_model(train_loader, val_loader, monitoring_loader=monitoring_loader)

    # Speichere das Modell
    models_dir = os.path.join(project_root, 'results', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, 'edge_gat_model_classification.pth')
    trainer.save_model(model_path)

    logger.info("Programm erfolgreich abgeschlossen.")

if __name__ == "__main__":
    main()
