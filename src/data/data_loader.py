import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from math import pi

def add_positional_encoding(df, columns, max_value=10000):
    """
    Fügt Positionskodierung mittels Sinus und Kosinus für geografische Koordinaten hinzu.
    """
    for col in columns:
        df[f'{col}_sin'] = np.sin(df[col] * (2 * pi / max_value))
        df[f'{col}_cos'] = np.cos(df[col] * (2 * pi / max_value))
    return df

def graph_based_imputation(df, edge_index, feature_name):
    """
    Führt eine graph-basierte Imputation für fehlende Werte in der angegebenen Feature-Spalte durch.
    """
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

def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, included_nodes, zfluss_wl_nodes):
    """
    Lädt Daten aus Knoten- und Kanten-Dateien und bereitet sie für das GAT-Modell vor.
    """
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

    # Graph-basierte Imputation für fehlende ZUFLUSS_WL-Werte
    nodes_df = graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

    # Fehlende Werte für andere physikalische Spalten mit KNN-Imputation behandeln
    imputer = KNNImputer(n_neighbors=5)
    nodes_df[adjusted_physical_columns] = imputer.fit_transform(nodes_df[adjusted_physical_columns])

    # Entfernen der Hilfsspalte
    nodes_df = nodes_df.drop(columns=['Included'])

    # Anwenden der Skalierung
    nodes_df[all_physical_columns] = physical_scaler.transform(nodes_df[all_physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])

    # Positionskodierung für geografische Spalten hinzufügen
    nodes_df = add_positional_encoding(nodes_df, geo_columns)

    # Knotenfeatures erstellen
    node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values

    # Entfernen des Features 'RAU' aus den Edge-Features
    edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)
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
