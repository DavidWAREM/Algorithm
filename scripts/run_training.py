import logging
import os
import torch
import yaml
from torch_geometric.loader import DataLoader  # Aktualisierter Import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, accuracy_score

from src.logging_config import setup_logging
from src.data.data_loader import load_data, graph_based_imputation
from src.data.data_preparation import prepare_first_dataset, scale_data
from src.models.model_setup import initialize_model, initialize_optimizer
from src.train.training_process import train_model
from src.evaluation.evaluation import evaluate_model
from src.utils.config_loader import load_config

def main():

    setup_logging()
    logger = logging.getLogger(__name__)

    # Bestimme das Projektverzeichnis und den Pfad zur Konfigurationsdatei
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    # Lade die Konfigurationsdatei
    config = load_config(config_path, logger)
    if config is None:
        return

    # Greife auf den Datenpfad aus der Konfiguration zu
    try:
        directory = config['paths']['folder_path_data']
    except KeyError:
        logger.error("Der Schlüssel 'paths.folder_path_data' fehlt in der Konfigurationsdatei.")
        return

    # Erhalte Hyperparameter aus der Konfiguration und konvertiere sie in die richtigen Typen
    try:
        batch_size = int(config['hyperparameters']['batch_size'])
        hidden_dim = int(config['hyperparameters']['hidden_dim'])
        dropout = float(config['hyperparameters']['dropout'])
        learning_rate = float(config['hyperparameters']['learning_rate'])
        weight_decay = float(config['hyperparameters']['weight_decay'])
        num_epochs = int(config['hyperparameters']['num_epochs'])
        patience = int(config['hyperparameters']['patience'])
        lr_scheduler_factor = float(config['hyperparameters']['lr_scheduler_factor'])
        lr_scheduler_patience = int(config['hyperparameters']['lr_scheduler_patience'])
    except KeyError as e:
        logger.error(f"Fehlender Hyperparameter: {e}")
        return
    except ValueError as e:
        logger.error(f"Fehler bei der Konvertierung der Hyperparameter: {e}")
        return

    # Lade die Listen aus der Konfiguration
    try:
        included_nodes = config['nodes']['included_nodes']
        zfluss_wl_nodes = config['nodes']['zfluss_wl_nodes']
    except KeyError as e:
        logger.error(f"Fehlender Knoten-Parameter in der Konfigurationsdatei: {e}")
        return

    # Geografische und physikalische Spalten
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
    adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    additional_physical_columns = ['ZUFLUSS_WL']
    all_physical_columns = adjusted_physical_columns + additional_physical_columns

    # Datenvorbereitung
    nodes_df_first, edges_df_first = prepare_first_dataset(
        directory, all_physical_columns, geo_columns, included_nodes, zfluss_wl_nodes
    )

    # One-Hot-Encoding für 'ROHRTYP'
    edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP')

    # Entfernen des Features 'RAU' aus edges_df_first, falls vorhanden
    if 'RAU' in edges_df_first.columns:
        edges_df_first = edges_df_first.drop(columns=['RAU'])

    # Aktualisieren der edge_columns nach One-Hot-Encoding
    edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df_first.filter(like='ROHRTYP').columns)

    # Skalieren der Daten
    physical_scaler, geo_scaler, edge_scaler = scale_data(
        nodes_df_first, edges_df_first, all_physical_columns, geo_columns, edge_columns
    )

    # Laden aller Datasets mit Skalierung und Positionskodierung
    datasets = []
    for i in range(1, 109):
        node_file = os.path.join(directory, f'SyntheticData-Spechbach_Valve_{i}_combined_Node.csv')
        edge_file = os.path.join(directory, f'SyntheticData-Spechbach_Valve_{i}_combined_Pipes.csv')
        try:
            data = load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, included_nodes, zfluss_wl_nodes)
            datasets.append(data)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten für Valve {i}: {e}")
            continue

    # Überprüfen, ob mindestens ein gültiger Datensatz vorhanden ist
    if not datasets:
        logger.error("Keine gültigen Datensätze verfügbar. Bitte überprüfen Sie die Daten.")
        return

    # Aufteilen in Trainings- und Validierungsdaten
    train_data, val_data = train_test_split(datasets, test_size=0.2, random_state=42)

    # Überprüfen, ob positive Beispiele in Trainings- und Validierungsdaten vorhanden sind
    train_positive = sum([data.y.sum().item() for data in train_data])
    val_positive = sum([data.y.sum().item() for data in val_data])

    if train_positive == 0 or val_positive == 0:
        logger.warning("Keine positiven Beispiele im Training oder in der Validierung. Bitte überprüfen Sie die Datenaufteilung.")
        return

    # Initialisiere das Modell, den Optimierer und die Verlustfunktion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(
        num_node_features=datasets[0].x.shape[1],
        num_edge_features=datasets[0].edge_attr.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device
    )
    optimizer = initialize_optimizer(model, learning_rate, weight_decay)

    # Verlustfunktion mit Klassengewichtung
    pos_weight = torch.tensor([len(train_data[0].edge_index[0]) - 1], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Lernraten-Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=True
    )

    # Erstelle DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Trainiere das Modell
    model = train_model(
        train_loader, val_loader, model, optimizer, criterion, scheduler, device, num_epochs, patience, logger
    )

    # Teste das Modell
    test_loader = DataLoader(datasets, batch_size=batch_size, shuffle=False)
    auc_score, accuracy, model = evaluate_model(test_loader=test_loader, model=model, device=device, logger=logger)

    # Modell speichern
    model_path = os.path.join(directory, 'edge_gat_model_classification.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Modell wurde gespeichert unter: {model_path}')

if __name__ == "__main__":
    main()
