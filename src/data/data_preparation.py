import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from src.data.data_loader import load_data, graph_based_imputation


def prepare_first_dataset(directory, all_physical_columns, geo_columns, included_nodes, zfluss_wl_nodes):
    node_file_first = os.path.join(directory, 'SyntheticData-Spechbach_Valve_1_combined_Node.csv')
    edge_file_first = os.path.join(directory, 'SyntheticData-Spechbach_Valve_1_combined_Pipes.csv')

    nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
    edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')

    # Für Knoten, die nicht in included_nodes sind, die Werte der adjusted_physical_columns auf NaN setzen
    nodes_df_first['Included'] = nodes_df_first['KNAM'].isin(included_nodes)
    for col in all_physical_columns:
        if col != 'ZUFLUSS_WL':  # Handle 'ZUFLUSS_WL' separat
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
    adjusted_physical_columns = [col for col in all_physical_columns if col != 'ZUFLUSS_WL']
    nodes_df_first[adjusted_physical_columns] = imputer.fit_transform(nodes_df_first[adjusted_physical_columns])

    # Graph-basierte Imputation für ZUFLUSS_WL im ersten Datensatz
    node_mapping_first = {name: idx for idx, name in enumerate(nodes_df_first['KNAM'])}
    edges_df_first['ANFNR'] = edges_df_first['ANFNAM'].map(node_mapping_first)
    edges_df_first['ENDNR'] = edges_df_first['ENDNAM'].map(node_mapping_first)
    edge_index_first = edges_df_first[['ANFNR', 'ENDNR']].values.T

    nodes_df_first = graph_based_imputation(nodes_df_first, edge_index_first, 'ZUFLUSS_WL')

    # Entfernen der Hilfsspalte
    nodes_df_first = nodes_df_first.drop(columns=['Included'])

    return nodes_df_first, edges_df_first


def scale_data(nodes_df, edges_df, all_physical_columns, geo_columns, edge_columns):
    physical_scaler = StandardScaler()
    geo_scaler = MinMaxScaler()
    edge_scaler = StandardScaler()

    physical_scaler.fit(nodes_df[all_physical_columns])
    geo_scaler.fit(nodes_df[geo_columns])
    edge_scaler.fit(edges_df[edge_columns])

    nodes_df[all_physical_columns] = physical_scaler.transform(nodes_df[all_physical_columns])
    nodes_df[geo_columns] = geo_scaler.transform(nodes_df[geo_columns])
    edges_df[edge_columns] = edge_scaler.transform(edges_df[edge_columns])

    return physical_scaler, geo_scaler, edge_scaler
