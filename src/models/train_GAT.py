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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import logging
import csv

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
        self.rau_scaler = StandardScaler()  # Hinzugefügt
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

        required_edge_columns = ['ANFNAM', 'ENDNAM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'ROHRTYP', 'RORL', 'DM', 'RAISE', 'RAU']
        for col in required_edge_columns:
            if col not in edges_df.columns:
                logger.debug(f"Spalte {col} fehlt in {edge_file}.")
                raise ValueError(f"Spalte {col} fehlt in {edge_file}.")

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
        edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'RAU']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'RAU']].astype(float)
        logger.debug("Relevante Kantenspalten in float konvertiert.")

        # Zielvariable erstellen und skalieren (verwende den bereits angepassten Skalierer)
        y = edges_df['RAU'].values.reshape(-1, 1)
        y_scaled = self.rau_scaler.transform(y).flatten()
        y = torch.tensor(y_scaled, dtype=torch.float)
        logger.debug("Skalierte Zielvariable 'RAU' erstellt.")

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

        logger.debug("PyTorch Geometric Data-Objekt erstellt.")
        return data

    def fit_scalers(self):
        # Liste aller Datensätze zum Anpassen der Skalierer
        all_nodes_dfs = []
        all_edges_dfs = []

        for i in range(1, self.num_valves + 1):
            node_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Roughness_{i}_combined_Node.csv')
            edge_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Roughness_{i}_combined_Pipes.csv')

            try:
                nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
                edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
                all_nodes_dfs.append(nodes_df)
                all_edges_dfs.append(edges_df)
                logger.debug(f"Dateien zum Anpassen der Skalierer geladen: {node_file}, {edge_file}.")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Dateien {node_file} oder {edge_file} zum Anpassen der Skalierer: {e}")
                continue

        # Kombiniere alle DataFrames
        nodes_df_all = pd.concat(all_nodes_dfs, ignore_index=True)
        edges_df_all = pd.concat(all_edges_dfs, ignore_index=True)

        # Knoten- und Kantenbezeichnungen bereinigen
        nodes_df_all['KNAM'] = nodes_df_all['KNAM'].astype(str).str.strip().str.lower()
        edges_df_all['ANFNAM'] = edges_df_all['ANFNAM'].astype(str).str.strip().str.lower()
        edges_df_all['ENDNAM'] = edges_df_all['ENDNAM'].astype(str).str.strip().str.lower()

        # Setze adjusted_physical_columns auf NaN für Knoten, die nicht in included_nodes sind
        nodes_df_all['Included'] = nodes_df_all['KNAM'].isin([n.lower() for n in self.included_nodes])
        for col in self.adjusted_physical_columns:
            nodes_df_all.loc[~nodes_df_all['Included'], col] = np.nan
            logger.debug(f"{col} auf NaN gesetzt für Knoten, die nicht enthalten sind (Skalierer-Anpassung).")

        # ZUFLUSS_WL nur für bestimmte Knoten behandeln
        nodes_df_all['ZUFLUSS_WL'] = nodes_df_all.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in self.zfluss_wl_nodes] else np.nan,
            axis=1
        )
        logger.debug("'ZUFLUSS_WL' für bestimmte Knoten behandelt (Skalierer-Anpassung).")

        # Indikatorspalten für fehlende Werte hinzufügen
        for col in self.all_physical_columns:
            nodes_df_all[f'{col}_missing'] = nodes_df_all[col].isna().astype(float)
            logger.debug(f"Fehlender Indikator für {col} hinzugefügt (Skalierer-Anpassung).")

        # KNN-Imputation (falls erforderlich)
        imputer = KNNImputer(n_neighbors=5)
        nodes_df_all[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df_all[self.adjusted_physical_columns])
        logger.debug("KNN-Imputation für angepasste physikalische Spalten durchgeführt (Skalierer-Anpassung).")

        # Skalierer an Knoteneigenschaften anpassen
        self.physical_scaler.fit(nodes_df_all[self.all_physical_columns])
        self.geo_scaler.fit(nodes_df_all[self.geo_columns])
        self.rau_scaler.fit(edges_df_all[['RAU']])  # Skalieren der Zielvariable
        logger.debug("Physikalische, geografische und RAU Skalierer angepasst.")

        # One-Hot-Encoding für 'ROHRTYP' durchführen
        edges_df_all = pd.get_dummies(edges_df_all, columns=['ROHRTYP'], prefix='ROHRTYP', dtype=float)
        logger.debug("One-Hot-Encoding für 'ROHRTYP' durchgeführt (Skalierer-Anpassung).")

        # Aktualisiere edge_columns nach One-Hot-Encoding
        continuous_edge_columns = ['RORL', 'DM', 'RAISE']
        one_hot_edge_columns = list(edges_df_all.filter(like='ROHRTYP').columns)
        edge_columns = continuous_edge_columns + one_hot_edge_columns

        # Sicherstellen, dass alle Kantenattribute numerisch sind
        edges_df_all[edge_columns] = edges_df_all[edge_columns].apply(pd.to_numeric, errors='coerce')

        # Fehlende Werte in One-Hot-encodierten Spalten füllen (falls vorhanden)
        edges_df_all[one_hot_edge_columns] = edges_df_all[one_hot_edge_columns].fillna(0)

        # Skalierer an kontinuierliche Kantenattribute anpassen
        self.edge_scaler.fit(edges_df_all[continuous_edge_columns])
        logger.debug("Skalierer an kontinuierliche Kantenattribute angepasst.")

    def load_all_data(self):
        self.fit_scalers()
        logger.info("Beginne mit dem Laden aller Datensätze.")
        for i in range(1, self.num_valves + 1):
            node_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Roughness_{i}_combined_Node.csv')
            edge_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Roughness_{i}_combined_Pipes.csv')
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

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  # Angepasste Batchgröße
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        logger.info("DataLoader für Trainings-, Validierungs- und Testmengen erstellt.")

        return train_loader, val_loader, test_loader

# GAT Model with Edge Prediction for Regression
class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, dropout=0.15):
        super(EdgeGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * 8)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 8)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim * 8)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim * 8),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 8, hidden_dim * 8)
        )
        self.fc_edge = torch.nn.Linear(2 * hidden_dim * 8 + hidden_dim * 8, 1)  # Output ist ein Skalar
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
        x = F.elu(x)
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass durch conv3.")

        edge_features = self.edge_mlp(edge_attr)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_logits = self.fc_edge(edge_embeddings).squeeze()
        logger.debug("Berechnete edge_logits.")

        return edge_logits  # Gibt rohe Werte für Regression zurück

# Trainer Class
class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, num_epochs=300, patience=20, results_dir='results'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.results_dir = results_dir  # Hinzugefügt
        self.best_model_state = None
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(batch)
            loss = self.criterion(preds, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.debug(f"Training epoch loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model(batch)
                loss = self.criterion(preds, batch.y)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.debug(f"Validation epoch loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_monitoring_dataset(self, monitoring_loader, epoch, data_module):
        self.model.eval()
        total_loss = 0
        y_true_list = []
        y_pred_list = []
        edge_id_list = []

        with torch.no_grad():
            for batch in monitoring_loader:
                batch = batch.to(self.device)
                preds_scaled = self.model(batch)
                loss = self.criterion(preds_scaled, batch.y)
                total_loss += loss.item()
                y_true_list.extend(batch.y.cpu().numpy().flatten())
                y_pred_list.extend(preds_scaled.cpu().numpy().flatten())
                edge_id_list.extend(batch.edge_ids)

        avg_loss = total_loss / len(monitoring_loader)
        y_true_scaled = np.array(y_true_list).reshape(-1)
        y_pred_scaled = np.array(y_pred_list).reshape(-1)
        edge_ids_all = np.array(edge_id_list).reshape(-1)

        # Inverse Skalierung der Vorhersagen und der tatsächlichen Werte
        y_true = data_module.rau_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = data_module.rau_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Berechne Metriken auf den invers skalierten Werten
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'r2_score': r2
        }

        # Speichern der Vorhersagen pro Kante in einer CSV-Datei
        df_predictions = pd.DataFrame({
            'edge_id': edge_ids_all,
            'y_true': y_true,
            'y_pred': y_pred
        })

        # CSV-Datei speichern
        csv_filename = f'monitoring_predictions_epoch_{epoch}.csv'
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        df_predictions.to_csv(csv_filepath, index=False)
        logger.info(f'Per-Kante-Vorhersagen für Epoche {epoch} gespeichert unter {csv_filepath}')

        return metrics

    def train_model(self, train_loader, val_loader, monitoring_loader=None, data_module=None):
        best_val_loss = float('inf')
        patience_counter = 0

        self.monitoring_results = []

        logger.info("Starte den Trainingsprozess.")
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            # Evaluierung auf dem Überwachungsdatensatz
            if monitoring_loader is not None and data_module is not None:
                monitoring_metrics = self.evaluate_monitoring_dataset(monitoring_loader, epoch, data_module)
                self.monitoring_results.append((epoch, monitoring_metrics))
                logger.info(
                    f'Epoche {epoch:03d}, Überwachung - Verlust: {monitoring_metrics["loss"]:.4f}, '
                    f'MSE: {monitoring_metrics["mse"]:.4f}, MAE: {monitoring_metrics["mae"]:.4f}, R²: {monitoring_metrics["r2_score"]:.4f}'
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
                logger.debug(f"Epoche {epoch}: Verbesserter Validierungsverlust auf {val_loss:.4f}.")
            else:
                patience_counter += 1
                logger.debug(f"Epoche {epoch}: Keine Verbesserung des Validierungsverlusts.")

            if patience_counter >= self.patience:
                self.logger.info(f"Frühes Stoppen in Epoche {epoch}.")
                break

        # Lade das beste Modell
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Bestes Modell basierend auf dem Validierungsverlust geladen.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'Modell gespeichert unter: {path}')

# Evaluator Class
class Evaluator:
    def __init__(self, model, device, target_scaler):
        self.model = model
        self.device = device
        self.rau_scaler = target_scaler  # Skalierer für RAU
        self.logger = logging.getLogger(__name__)

    def test_model(self, loader):
        y_true_scaled, y_pred_scaled = self._test(loader)
        y_true = self.rau_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = self.rau_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_true, y_pred

    def _test(self, loader):
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds_scaled = self.model(batch)
                y_true_list.append(batch.y.cpu().numpy())
                y_pred_list.append(preds_scaled.cpu().numpy())
        y_true_scaled = np.concatenate(y_true_list)
        y_pred_scaled = np.concatenate(y_pred_list)
        logger.info("Modelltest abgeschlossen.")
        return y_true_scaled, y_pred_scaled

    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self.logger.info(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}')
        return mse, mae, r2

    def plot_metrics(self, y_true, y_pred):
        self.plot_predictions(y_true, y_pred)
        self.plot_residuals(y_true, y_pred)

    def plot_predictions(self, y_true, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Tatsächliche RAU-Werte')
        plt.ylabel('Vorhergesagte RAU-Werte')
        plt.title('Tatsächliche vs. Vorhergesagte RAU-Werte')
        plt.legend(['Ideal'])
        plt.grid(True)
        plt.show()
        logger.debug("Plot der tatsächlichen vs. vorhergesagten Werte erstellt.")

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
        plt.xlabel('Vorhergesagte RAU-Werte')
        plt.ylabel('Residuen')
        plt.title('Residuenplot')
        plt.grid(True)
        plt.show()
        logger.debug("Residuenplot erstellt.")

# Main Function
def main():

    # Pfad zur Konfigurationsdatei relativ zum Projektstammverzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # Zwei Ebenen hoch
    config_file = os.path.join(project_root, 'config', 'config.yaml')  # Standardpfad zur config.yaml

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    directory = config['paths']['folder_path_data']

    logger = logging.getLogger(__name__)

    # Liste der Knoten mit verfügbaren Messdaten
    included_nodes = config['nodes']['included_nodes']
    zfluss_wl_nodes = config['nodes']['zfluss_wl_nodes']

    # Initialisiere DataModule
    data_module = DataModule(directory, included_nodes, zfluss_wl_nodes, num_valves=4000)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    logger.info("AdamW-Optimierer initialisiert.")

    # Verlustfunktion für Regression
    criterion = torch.nn.MSELoss()
    logger.info("MSELoss initialisiert.")

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    logger.info("ReduceLROnPlateau-Scheduler initialisiert.")

    # Definiere das Ergebnisverzeichnis für den Trainer
    results_dir = os.path.join(project_root, 'results', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialisiere Trainer
    trainer = Trainer(
        model, optimizer, criterion, scheduler, device,
        num_epochs=500, patience=20, results_dir=results_dir  # Übergebe results_dir
    )
    logger.info("Trainer initialisiert.")

    # Laden des Überwachungsdatensatzes
    monitoring_node_file = os.path.join(directory, 'SyntheticData-Spechbach_Roughness_0_combined_Node.csv')
    monitoring_edge_file = os.path.join(directory, 'SyntheticData-Spechbach_Roughness_0_combined_Pipes.csv')

    try:
        monitoring_data = data_module.load_data(monitoring_node_file, monitoring_edge_file)
        monitoring_loader = DataLoader([monitoring_data], batch_size=16, shuffle=False)  # Angepasste Batchgröße
        logger.info("Überwachungsdatensatz erfolgreich geladen.")
    except Exception as e:
        logger.error(f'Fehler beim Laden des Überwachungsdatensatzes: {e}')
        monitoring_loader = None

    # Start des Trainings mit Übergabe des monitoring_loader und data_module
    trainer.train_model(train_loader, val_loader, monitoring_loader=monitoring_loader, data_module=data_module)

    # Initialisiere Evaluator mit dem Skalierer
    evaluator = Evaluator(model, device, data_module.rau_scaler)
    logger.info("Evaluator initialisiert.")

    # Teste das Modell auf dem Testdatensatz
    y_true, y_pred = evaluator.test_model(test_loader)

    # Berechne Metriken
    mse, mae, r2 = evaluator.calculate_metrics(y_true, y_pred)

    # Generiere Plots
    evaluator.plot_metrics(y_true, y_pred)

    # Speichere das Modell
    models_dir = os.path.join(project_root, 'results', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, 'edge_gat_model_regression.pth')
    trainer.save_model(model_path)

    # Speichere die Überwachungsergebnisse im gleichen Ordner wie die visuellen Ausgaben
    # Bereits erledigt durch die Trainer-Klasse

    logger.info("Programm erfolgreich abgeschlossen.")

if __name__ == "__main__":
    main()
