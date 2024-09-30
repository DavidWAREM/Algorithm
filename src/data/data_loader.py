# src/data/data_loader.py
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.impute import KNNImputer
import logging

def encode_rohrtyp(edges_df, all_rohrtyp_columns):
    """
    Ensures that all ROHRTYP One-Hot Encoded columns are present in the edges_df.
    Missing columns are filled with zeros.

    Args:
        edges_df (pd.DataFrame): Edges DataFrame after One-Hot Encoding.
        all_rohrtyp_columns (list): Complete list of ROHRTYP One-Hot Encoded column names.

    Returns:
        pd.DataFrame: Updated edges_df with all ROHRTYP columns.
    """
    missing_rohrtyp_columns = [col for col in all_rohrtyp_columns if col not in edges_df.columns]
    if missing_rohrtyp_columns:
        edges_df = edges_df.reindex(
            columns=edges_df.columns.tolist() + missing_rohrtyp_columns,
            fill_value=0
        )
    return edges_df

def graph_based_imputation(nodes_df, edge_index, column):
    """
    Performs graph-based imputation on the specified column using KNNImputer.

    Args:
        nodes_df (pd.DataFrame): Nodes DataFrame with missing values.
        edge_index (np.ndarray): Edge indices for graph connectivity.
        column (str): Column name to impute.

    Returns:
        pd.DataFrame: Updated nodes_df with imputed values.
    """
    logger = logging.getLogger(__name__)
    try:
        # Extract the column to impute
        data = nodes_df[[column]].copy()

        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=5)

        # Perform imputation
        imputed_data = imputer.fit_transform(data)

        # Update the DataFrame
        nodes_df[column] = imputed_data

        logger.info(f"Graph-based Imputation für '{column}' abgeschlossen.")
        return nodes_df
    except Exception as e:
        logger.error(f"Fehler bei der Graph-basierten Imputation für '{column}': {e}")
        raise

def load_data(node_file, edge_file, physical_scaler, geo_scaler, edge_scaler, included_nodes, zfluss_wl_nodes, all_rohrtyp_columns):
    """
    Loads and preprocesses node and edge data, applies scaling, and creates PyTorch Geometric Data objects.

    Args:
        node_file (str): Path to the node CSV file.
        edge_file (str): Path to the edge CSV file.
        physical_scaler (StandardScaler): Fitted scaler for physical features.
        geo_scaler (MinMaxScaler): Fitted scaler for geographical features.
        edge_scaler (StandardScaler): Fitted scaler for edge features.
        included_nodes (list): List of node names to include.
        zfluss_wl_nodes (list): List of node names for 'ZUFLUSS_WL' processing.
        all_rohrtyp_columns (list): Complete list of ROHRTYP One-Hot Encoded column names.

    Returns:
        torch_geometric.data.Data: PyTorch Geometric Data object.
    """
    logger = logging.getLogger(__name__)

    # Load CSV files
    nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
    edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
    logger.info(f"Loaded node data from {node_file} and edge data from {edge_file}.")

    # Standardize node names
    nodes_df['KNAM'] = nodes_df['KNAM'].str.strip().str.upper()
    edges_df['ANFNAM'] = edges_df['ANFNAM'].str.strip().str.upper()
    edges_df['ENDNAM'] = edges_df['ENDNAM'].str.strip().str.upper()

    # Create node mapping before any modifications
    node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
    edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
    edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

    # Check for unmapped edges
    unmapped_edges = edges_df[edges_df['ANFNR'].isnull() | edges_df['ENDNR'].isnull()]
    if not unmapped_edges.empty:
        logger.error("Some edges konnten nicht zugeordnet werden zu Knoten-Indizes.")
        logger.debug(f"Unzugeordnete Kanten:\n{unmapped_edges}")
        # Entfernen Sie diese Kanten oder behandeln Sie sie entsprechend
        edges_df = edges_df.dropna(subset=['ANFNR', 'ENDNR'])

    # Proceed with node modifications
    nodes_df['Included'] = nodes_df['KNAM'].isin(included_nodes)
    for col in physical_scaler.feature_names_in_:
        if col != 'ZUFLUSS_WL':
            nodes_df.loc[~nodes_df['Included'], col] = np.nan

    # Handle 'ZUFLUSS_WL'
    nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
        lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in zfluss_wl_nodes else np.nan,
        axis=1
    )

    # Add missing indicators
    for col in physical_scaler.feature_names_in_:
        nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)

    # Graph-based imputation
    edge_index = edges_df[['ANFNR', 'ENDNR']].dropna().values.T.astype(int)
    nodes_df = graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

    # Drop helper column
    nodes_df = nodes_df.drop(columns=['Included'])

    # One-Hot Encoding für 'ROHRTYP'
    edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')
    logger.info("One-Hot Encoding für 'ROHRTYP' abgeschlossen.")

    # Reindex to include all possible 'ROHRTYP' columns
    edges_df = encode_rohrtyp(edges_df, all_rohrtyp_columns)
    logger.info("Reindexed edges_df, um alle ROHRTYP Spalten zu inkludieren.")

    # Drop 'RAU' if it exists
    if 'RAU' in edges_df.columns:
        edges_df = edges_df.drop(columns=['RAU'])
        logger.info("Spalte 'RAU' aus edges_df entfernt.")

    # Define edge columns
    edge_columns = ['RORL', 'DM', 'RAISE'] + all_rohrtyp_columns

    # Ensure edge_columns are present
    for col in edge_columns:
        if col not in edges_df.columns:
            edges_df[col] = 0
            logger.debug(f"Fehlende Kanten-Spalte hinzugefügt: {col}")

    # Scale edge features
    edges_df[edge_columns] = edge_scaler.transform(edges_df[edge_columns])

    # Prepare node features
    node_features = nodes_df.drop(columns=['KNAM']).values
    x = torch.tensor(node_features, dtype=torch.float)

    # Prepare edge_index
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    # Prepare edge attributes
    edge_attributes = edges_df[edge_columns].values
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    # Prepare target variable
    # Hier definieren Sie 'y' basierend auf Ihren Anforderungen
    # Beispiel: Binäre Klassifikation basierend auf 'FLUSS_WL'
    epsilon = 1e-6
    required_columns = ['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']
    missing_required = [col for col in required_columns if col not in edges_df.columns]
    if missing_required:
        logger.error(f"Fehlende erforderliche Spalten für Zielvariable: {missing_required}")
        # Definieren Sie 'y' als Null oder behandeln Sie dies entsprechend
        y = torch.zeros(edges_df.shape[0], dtype=torch.float)
    else:
        target_condition = (edges_df['FLUSS_WL'].abs() < epsilon) | \
                           (edges_df['FLUSS_WOL'].abs() < epsilon) | \
                           (edges_df['VM_WL'].abs() < epsilon) | \
                           (edges_df['VM_WOL'].abs() < epsilon)
        y = torch.tensor(target_condition.astype(float).values, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr, y=y)
    return data