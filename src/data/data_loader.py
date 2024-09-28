# src/data/data_loader.py
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.impute import KNNImputer
import logging

def graph_based_imputation(nodes_df, edge_index, column_name):
    """
    Performs graph-based imputation on a specified column.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node data.
        edge_index (np.ndarray): Array representing graph edges.
        column_name (str): Name of the column to impute.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    logger = logging.getLogger(__name__)
    try:
        values = nodes_df[column_name].values
        mask = np.isnan(values)
        for i in np.where(mask)[0]:
            # Find connected nodes
            connected_indices = edge_index[:, edge_index[0] == i][1]
            connected_values = values[connected_indices]
            connected_values = connected_values[~np.isnan(connected_values)]
            if len(connected_values) > 0:
                values[i] = np.mean(connected_values)
            else:
                logger.warning(f"No connected values found for node {i}; cannot impute.")
        nodes_df[column_name] = values
        logger.info(f"Graph-based imputation completed for column '{column_name}'.")
        return nodes_df
    except Exception as e:
        logger.error(f"Error during graph-based imputation: {e}")
        raise

def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, included_nodes, zfluss_wl_nodes):
    """
    Loads and processes node and edge data from CSV files.

    Args:
        node_file (str): Path to the node CSV file.
        edge_file (str): Path to the edge CSV file.
        physical_scaler (StandardScaler): Scaler for physical features.
        geo_scaler (MinMaxScaler): Scaler for geographical features.
        edge_scaler (StandardScaler): Scaler for edge features.
        included_nodes (list): List of node names to include.
        zfluss_wl_nodes (list): List of node names for 'ZUFLUSS_WL' processing.

    Returns:
        torch_geometric.data.Data: Processed graph data.
    """
    logger = logging.getLogger(__name__)
    try:
        # Load CSV files
        nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
        edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
        logger.info(f"Loaded node data from {node_file} and edge data from {edge_file}.")

        # Preprocessing nodes
        nodes_df['Included'] = nodes_df['KNAM'].isin(included_nodes)
        for col in physical_scaler.feature_names_in_:
            if col != 'ZUFLUSS_WL':
                nodes_df.loc[~nodes_df['Included'], col] = np.nan

        # Handle 'ZUFLUSS_WL' only for specified nodes
        nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in zfluss_wl_nodes else np.nan,
            axis=1
        )

        # Add missing indicators
        for col in physical_scaler.feature_names_in_:
            nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)

        # KNN Imputation for physical columns
        imputer = KNNImputer(n_neighbors=5)
        adjusted_physical_columns = [col for col in physical_scaler.feature_names_in_ if col != 'ZUFLUSS_WL']
        nodes_df[adjusted_physical_columns] = imputer.fit_transform(nodes_df[adjusted_physical_columns])
        logger.info("KNN imputation completed for physical features.")

        # Graph-based Imputation for 'ZUFLUSS_WL'
        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)
        edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

        nodes_df = graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

        # Drop the helper column
        nodes_df = nodes_df.drop(columns=['Included'])

        # Scaling
        nodes_df[physical_scaler.feature_names_in_] = physical_scaler.transform(nodes_df[physical_scaler.feature_names_in_])
        nodes_df[['XRECHTS', 'YHOCH', 'GEOH']] = geo_scaler.transform(nodes_df[['XRECHTS', 'YHOCH', 'GEOH']])
        edges_df[edge_scaler.feature_names_in_] = edge_scaler.transform(edges_df[edge_scaler.feature_names_in_])
        logger.info("Scaling of features completed.")

        # Convert to PyTorch Geometric Data
        x = torch.tensor(nodes_df[physical_scaler.feature_names_in_].values, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edges_df[edge_scaler.feature_names_in_].values, dtype=torch.float)
        y = torch.tensor(nodes_df['ZUFLUSS_WL'].notna().astype(float).values, dtype=torch.float).unsqueeze(1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        logger.info("Data conversion to PyTorch Geometric format completed.")
        return data

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error.filename}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("No data: One of the CSV files is empty.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise
