import os
import glob
import pandas as pd
import torch
import numpy as np
import yaml
import joblib
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Updated import to resolve deprecation warning
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import logging

# Initialize logging with INFO level to capture essential information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataModule:
    """
    DataModule is responsible for loading, preprocessing, and preparing the dataset
    for training, validation, and testing. It handles node and edge data, applies
    scaling, imputes missing values, and adds positional encoding to geographic features.
    """

    def __init__(self, directory, included_nodes, zfluss_wl_nodes):
        """
        Initializes the DataModule with the specified directory and node configurations.

        Args:
            directory (str): Path to the directory containing node and edge CSV files.
            included_nodes (list): List of node names to be included with available measurement data.
            zfluss_wl_nodes (list): List of node names for which 'ZUFLUSS_WL' is applicable.
        """
        self.directory = directory
        self.included_nodes = included_nodes
        self.zfluss_wl_nodes = zfluss_wl_nodes

        # Initialize scalers for different types of features
        self.physical_scaler = StandardScaler()
        self.geo_scaler = MinMaxScaler()
        self.edge_scaler = StandardScaler()
        self.rau_scaler = StandardScaler()  # For scaling the target variable 'RAU'

        self.datasets = []

        # Define column names for geographic and physical attributes
        self.geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
        self.adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
        self.additional_physical_columns = ['ZUFLUSS_WL']
        self.all_physical_columns = self.adjusted_physical_columns + self.additional_physical_columns

        # Define node feature columns (12 features)
        self.node_feature_columns = [
            'PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp',
            'ZUFLUSS_WL',
            'XRECHTS_sin', 'XRECHTS_cos',
            'YHOCH_sin', 'YHOCH_cos',
            'GEOH_sin', 'GEOH_cos'
        ]

        # Define edge feature columns (3 features)
        self.edge_feature_columns = ['RORL', 'DM', 'RAISE']

    def add_positional_encoding(self, df, columns, max_value=10000):
        """
        Adds sinusoidal positional encoding to the specified columns to capture spatial information.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            columns (list): List of column names to which positional encoding will be added.
            max_value (int, optional): Maximum value used to scale the positional encoding. Defaults to 10000.

        Returns:
            pd.DataFrame: DataFrame with added positional encoding columns.
        """
        for col in columns:
            df[f'{col}_sin'] = np.sin(df[col] * (2 * np.pi / max_value))
            df[f'{col}_cos'] = np.cos(df[col] * (2 * np.pi / max_value))
        return df

    def graph_based_imputation(self, df, edge_index, feature_name):
        """
        Performs graph-based imputation for missing values in a specified feature using neighboring nodes.

        Args:
            df (pd.DataFrame): DataFrame containing node data.
            edge_index (np.ndarray): Array containing edge indices.
            feature_name (str): Name of the feature to impute.

        Returns:
            pd.DataFrame: DataFrame with imputed values for the specified feature.
        """
        node_values = df[feature_name].values
        missing_mask = np.isnan(node_values)

        # Create adjacency list from edge indices
        adjacency = {i: [] for i in range(len(df))}
        for src, dst in edge_index.T:
            adjacency[src].append(dst)
            adjacency[dst].append(src)

        # Iterate over missing values and impute based on neighbors
        for idx in np.where(missing_mask)[0]:
            neighbors = adjacency[idx]
            neighbor_values = [node_values[n] for n in neighbors if not np.isnan(node_values[n])]
            if neighbor_values:
                node_values[idx] = np.mean(neighbor_values)
            else:
                node_values[idx] = np.nanmean(node_values)

        df[feature_name] = node_values
        logger.debug(f"Graph-based imputation completed for feature '{feature_name}'.")
        return df

    def load_data(self, node_file, edge_file):
        """
        Loads and preprocesses node and edge data from CSV files, applies scaling,
        imputes missing values, and constructs a PyTorch Geometric Data object.

        Args:
            node_file (str): Path to the node CSV file.
            edge_file (str): Path to the edge CSV file.

        Returns:
            Data: PyTorch Geometric Data object containing processed node and edge data.
        """
        try:
            # Load node and edge data separately
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
            logger.debug(f"Loaded files: {node_file}, {edge_file}.")
        except Exception as e:
            logger.error(f"Error loading files {node_file} or {edge_file}: {e}")
            raise e

        # Check for required node columns
        required_node_columns = ['KNAM', 'XRECHTS', 'YHOCH', 'GEOH'] + self.adjusted_physical_columns + ['ZUFLUSS_WL']
        for col in required_node_columns:
            if col not in nodes_df.columns:
                logger.debug(f"Column {col} is missing in {node_file}.")
                raise ValueError(f"Column {col} is missing in {node_file}.")

        # Check for required edge columns
        required_edge_columns = [
            'ANFNAM', 'ENDNAM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL',
            'RORL', 'DM', 'RAISE', 'RAU'
        ]
        for col in required_edge_columns:
            if col not in edges_df.columns:
                logger.debug(f"Column {col} is missing in {edge_file}.")
                raise ValueError(f"Column {col} is missing in {edge_file}.")

        # Clean node and edge names by stripping whitespace and converting to lowercase
        nodes_df['KNAM'] = nodes_df['KNAM'].astype(str).str.strip().str.lower()
        edges_df['ANFNAM'] = edges_df['ANFNAM'].astype(str).str.strip().str.lower()
        edges_df['ENDNAM'] = edges_df['ENDNAM'].astype(str).str.strip().str.lower()

        # Map node names to unique indices for graph representation
        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        nodes_df['node_idx'] = nodes_df['KNAM'].map(node_mapping)
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

        # Check for any missing node indices in edges
        missing_anfnr = edges_df['ANFNR'].isnull()
        missing_endnr = edges_df['ENDNR'].isnull()

        if missing_anfnr.any() or missing_endnr.any():
            missing_anfnam = edges_df.loc[missing_anfnr, 'ANFNAM'].unique()
            missing_endnam = edges_df.loc[missing_endnr, 'ENDNAM'].unique()
            logger.error(f"Missing node indices for ANFNAMs: {missing_anfnam}, ENDNAMs: {missing_endnam}")
            raise ValueError("Edge data contains nodes that are not found in node data.")

        # Convert node indices to integers
        edges_df['ANFNR'] = edges_df['ANFNR'].astype(int)
        edges_df['ENDNR'] = edges_df['ENDNR'].astype(int)

        # Create a unique identifier for each edge by concatenating node names
        edges_df['edge_id'] = edges_df['ANFNAM'] + '_' + edges_df['ENDNAM']

        # Extract edge indices for graph representation
        edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

        # Ensure relevant edge columns are numeric
        edge_features_columns = self.edge_feature_columns  # ['RORL', 'DM', 'RAISE']
        edges_df[edge_features_columns] = edges_df[edge_features_columns].astype(float)
        logger.debug("Converted relevant edge columns to float.")

        # Create and scale the target variable 'RAU'
        y_df = edges_df[['RAU']].copy()
        y_df_scaled = pd.DataFrame(
            self.rau_scaler.transform(y_df),
            columns=['RAU'],
            index=y_df.index
        )
        y = torch.tensor(y_df_scaled['RAU'].values, dtype=torch.float)
        logger.debug("Scaled target variable 'RAU'.")

        # Adjust node attributes by setting non-included nodes' physical columns to NaN
        nodes_df['Included'] = nodes_df['KNAM'].isin([n.lower() for n in self.included_nodes])
        for col in self.adjusted_physical_columns:
            nodes_df.loc[~nodes_df['Included'], col] = np.nan
            logger.debug(f"Set {col} to NaN for nodes not included.")

        # Handle 'ZUFLUSS_WL' only for specific nodes
        nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in self.zfluss_wl_nodes] else np.nan,
            axis=1
        )
        logger.debug("Handled 'ZUFLUSS_WL' for specific nodes.")

        # Perform graph-based imputation for missing 'ZUFLUSS_WL' values
        nodes_df = self.graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

        # Handle missing values for other physical columns using KNN Imputer
        imputer = KNNImputer(n_neighbors=5)
        nodes_df[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df[self.adjusted_physical_columns])
        logger.debug("Performed KNN imputation for adjusted physical columns.")

        # Remove the helper 'Included' column as it's no longer needed
        nodes_df = nodes_df.drop(columns=['Included'])
        logger.debug("Removed helper column 'Included' from node data.")

        # Apply scaling to node attributes
        nodes_df[self.all_physical_columns] = pd.DataFrame(
            self.physical_scaler.transform(nodes_df[self.all_physical_columns]),
            columns=self.all_physical_columns,
            index=nodes_df.index
        )
        nodes_df[self.geo_columns] = pd.DataFrame(
            self.geo_scaler.transform(nodes_df[self.geo_columns]),
            columns=self.geo_columns,
            index=nodes_df.index
        )
        logger.debug("Applied scaling to physical and geographic node columns.")

        # Add positional encoding to geographic columns to capture spatial relationships
        nodes_df = self.add_positional_encoding(nodes_df, self.geo_columns)
        logger.debug("Added positional encoding to geographic columns.")

        # Create node features by selecting the defined columns
        node_features = nodes_df[self.node_feature_columns].values

        # Ensure all edge attributes are numeric (already handled above)
        # Apply scaling to edge attributes
        edges_df[self.edge_feature_columns] = pd.DataFrame(
            self.edge_scaler.transform(edges_df[self.edge_feature_columns]),
            columns=self.edge_feature_columns,
            index=edges_df.index
        )
        logger.debug("Applied scaling to edge attributes.")

        # Combine scaled edge attributes
        edge_attributes = edges_df[self.edge_feature_columns].values

        # Convert node features, edge indices, and edge attributes to PyTorch tensors
        try:
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        except Exception as e:
            logger.error(f"Error converting to tensors: {e}")
            logger.debug(f"node_features dtype: {node_features.dtype}")
            logger.debug(f"edge_index dtype: {edge_index.dtype}")
            logger.debug(f"edge_attr dtype: {edge_attributes.dtype}")
            raise e

        # Create a PyTorch Geometric Data object with node features, edge indices, edge attributes, and target variable
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Save edge identifiers in the Data object for reference
        data.edge_ids = edges_df['edge_id'].values  # This is a NumPy array

        logger.debug("Created PyTorch Geometric Data object.")
        return data

    def fit_scalers(self):
        """
        Fits the scalers for physical, geographic, and edge attributes using all training data.
        This ensures that scaling parameters are learned from the training set and can be consistently applied.
        """
        # Lists to hold all node and edge DataFrames for fitting scalers
        all_nodes_dfs = []
        all_edges_dfs = []

        # Use glob to find matching node and edge files based on patterns
        node_pattern = os.path.join(self.directory, '*_Roughness_*_combined_Node.csv')
        edge_pattern = os.path.join(self.directory, '*_Roughness_*_combined_Pipes.csv')

        node_files = glob.glob(node_pattern)
        edge_files = glob.glob(edge_pattern)

        # Sort files for consistency in processing
        node_files.sort()
        edge_files.sort()

        # Log the found files for scaler fitting
        logger.info(f"Found node files for scaler fitting: {node_files}")
        logger.info(f"Found edge files for scaler fitting: {edge_files}")

        # Define patterns to identify and exclude monitoring files (typically used for validation or testing)
        monitoring_node_pattern = os.path.join(self.directory, '*_Roughness_0_combined_Node.csv')
        monitoring_edge_pattern = os.path.join(self.directory, '*_Roughness_0_combined_Pipes.csv')

        monitoring_node_files = glob.glob(monitoring_node_pattern)
        monitoring_edge_files = glob.glob(monitoring_edge_pattern)

        # Remove monitoring files from the main lists to ensure only training data is used for scaler fitting
        for monitoring_node_file in monitoring_node_files:
            if monitoring_node_file in node_files:
                node_files.remove(monitoring_node_file)
                logger.debug(f"Removed monitoring node file: {monitoring_node_file}")
        for monitoring_edge_file in monitoring_edge_files:
            if monitoring_edge_file in edge_files:
                edge_files.remove(monitoring_edge_file)
                logger.debug(f"Removed monitoring edge file: {monitoring_edge_file}")

        # Check if there are any training files left after removing monitoring files
        if not node_files or not edge_files:
            logger.error("No training files found. Please check the data directory and file names.")
            raise ValueError("No training data found.")

        # Iterate over training files and load them into DataFrames
        for node_file, edge_file in zip(node_files, edge_files):
            try:
                nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
                edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
                all_nodes_dfs.append(nodes_df)
                all_edges_dfs.append(edges_df)
                logger.debug(f"Loaded files for scaler fitting: {node_file}, {edge_file}.")
            except Exception as e:
                logger.error(f"Error loading files {node_file} or {edge_file} for scaler fitting: {e}")
                continue

        # Log the number of loaded DataFrames
        logger.info(f"Number of node files loaded for scaler fitting: {len(all_nodes_dfs)}")
        logger.info(f"Number of edge files loaded for scaler fitting: {len(all_edges_dfs)}")

        # Check if any DataFrames were loaded
        if not all_nodes_dfs or not all_edges_dfs:
            logger.error("No node or edge data found for scaler fitting.")
            raise ValueError("No data available for scaler fitting.")

        # Combine all DataFrames into single DataFrames for nodes and edges
        nodes_df_all = pd.concat(all_nodes_dfs, ignore_index=True)
        edges_df_all = pd.concat(all_edges_dfs, ignore_index=True)

        # Clean node and edge names by stripping whitespace and converting to lowercase
        nodes_df_all['KNAM'] = nodes_df_all['KNAM'].astype(str).str.strip().str.lower()
        edges_df_all['ANFNAM'] = edges_df_all['ANFNAM'].astype(str).str.strip().str.lower()
        edges_df_all['ENDNAM'] = edges_df_all['ENDNAM'].astype(str).str.strip().str.lower()

        # Adjust node attributes by setting non-included nodes' physical columns to NaN
        nodes_df_all['Included'] = nodes_df_all['KNAM'].isin([n.lower() for n in self.included_nodes])
        for col in self.adjusted_physical_columns:
            nodes_df_all.loc[~nodes_df_all['Included'], col] = np.nan
            logger.debug(f"Set {col} to NaN for nodes not included (scaler fitting).")

        # Handle 'ZUFLUSS_WL' only for specific nodes
        nodes_df_all['ZUFLUSS_WL'] = nodes_df_all.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in [n.lower() for n in self.zfluss_wl_nodes] else np.nan,
            axis=1
        )
        logger.debug("Handled 'ZUFLUSS_WL' for specific nodes (scaler fitting).")

        # Perform graph-based imputation for missing 'ZUFLUSS_WL' values
        # Note: Edge index is not available here; assuming no imputation during scaler fitting

        # Handle missing values for other physical columns using KNN Imputer
        imputer = KNNImputer(n_neighbors=5)
        nodes_df_all[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df_all[self.adjusted_physical_columns])
        logger.debug("Performed KNN imputation for adjusted physical columns (scaler fitting).")

        # Remove the helper 'Included' column as it's no longer needed
        nodes_df_all = nodes_df_all.drop(columns=['Included'])
        logger.debug("Removed helper column 'Included' from node data (scaler fitting).")

        # Fit scalers using the combined training data
        self.physical_scaler.fit(nodes_df_all[self.all_physical_columns])
        self.geo_scaler.fit(nodes_df_all[self.geo_columns])
        self.rau_scaler.fit(edges_df_all[['RAU']])  # Scale target variable

        # Fit the edge scaler on edge features
        self.edge_scaler.fit(edges_df_all[self.edge_feature_columns])
        logger.debug("Fitted physical, geographic, edge, and RAU scalers.")

    def load_all_data(self):
        """
        Loads all datasets by fitting scalers first and then loading each dataset.
        Excludes monitoring files during loading.
        """
        self.fit_scalers()
        logger.info("Started loading all datasets.")

        # Use glob to find matching node and edge files based on patterns
        node_pattern = os.path.join(self.directory, '*_Roughness_*_combined_Node.csv')
        edge_pattern = os.path.join(self.directory, '*_Roughness_*_combined_Pipes.csv')

        node_files = glob.glob(node_pattern)
        edge_files = glob.glob(edge_pattern)

        # Sort files for consistency in processing
        node_files.sort()
        edge_files.sort()

        # Log the found files for data loading
        logger.info(f"Found node files for loading: {node_files}")
        logger.info(f"Found edge files for loading: {edge_files}")

        # Define patterns to identify and exclude monitoring files (typically used for validation or testing)
        monitoring_node_pattern = os.path.join(self.directory, '*_Roughness_0_combined_Node.csv')
        monitoring_edge_pattern = os.path.join(self.directory, '*_Roughness_0_combined_Pipes.csv')

        monitoring_node_files = glob.glob(monitoring_node_pattern)
        monitoring_edge_files = glob.glob(monitoring_edge_pattern)

        # Remove monitoring files from the main lists to ensure only training data is loaded
        for monitoring_node_file in monitoring_node_files:
            if monitoring_node_file in node_files:
                node_files.remove(monitoring_node_file)
                logger.debug(f"Removed monitoring node file: {monitoring_node_file}")
        for monitoring_edge_file in monitoring_edge_files:
            if monitoring_edge_file in edge_files:
                edge_files.remove(monitoring_edge_file)
                logger.debug(f"Removed monitoring edge file: {monitoring_edge_file}")

        # Check if there are any training files left after removing monitoring files
        if not node_files or not edge_files:
            logger.error("No training files found. Please check the data directory and file names.")
            raise ValueError("No training data found.")

        # Iterate over training files and load them into datasets
        for node_file, edge_file in zip(node_files, edge_files):
            try:
                data = self.load_data(node_file, edge_file)
                self.datasets.append(data)
                logger.debug(f"Successfully loaded dataset: {node_file}, {edge_file}")
            except Exception as e:
                logger.error(f"Error loading files {node_file} or {edge_file}: {e}")
                continue
        logger.info("All datasets loaded successfully.")

    def get_loaders(self, val_size=0.25, test_size=0.2, random_state=42):
        """
        Splits the loaded datasets into training, validation, and test sets and creates DataLoaders.

        Args:
            val_size (float, optional): Proportion of data to use for validation. Defaults to 0.25.
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            tuple: DataLoaders for training, validation, and testing datasets.
        """
        if not self.datasets:
            logger.error("No datasets available.")
            return None, None, None

        # Split data into training+validation and test sets
        train_val_data, test_data = train_test_split(
            self.datasets, test_size=test_size, random_state=random_state
        )
        # Adjust validation ratio relative to the remaining training data
        val_ratio = val_size / (1 - test_size)  # e.g., 0.25 / 0.8 = 0.3125
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_ratio, random_state=random_state
        )
        logger.info("Split data into training, validation, and test sets.")

        # Create DataLoaders with a batch size of 16
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        logger.info("Created DataLoaders for training, validation, and test sets.")

        # Log the feature columns used for training based on a sample dataset
        if len(self.datasets) > 0:
            sample_data = self.datasets[0]
            logger.info(f"Used node feature columns for training: {self.node_feature_columns}")
            logger.info(f"Used edge feature columns for training: {self.edge_feature_columns}")

        return train_loader, val_loader, test_loader


class EdgeGAT(torch.nn.Module):
    """
    EdgeGAT is a Graph Attention Network (GAT) model tailored for edge-level regression tasks.
    It processes node and edge features to predict target values associated with each edge.
    """

    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, dropout=0.15):
        """
        Initializes the EdgeGAT model with specified parameters.

        Args:
            num_node_features (int): Number of features per node.
            num_edge_features (int): Number of features per edge.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.15.
        """
        super(EdgeGAT, self).__init__()
        # Define three Graph Attention Convolutional layers with 8 attention heads each
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout)

        # Define Batch Normalization layers after each convolutional layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * 8)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 8)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim * 8)

        # Define an MLP for processing edge features
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim * 8),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 8, hidden_dim * 8)
        )

        # Define a fully connected layer to produce the final edge output
        self.fc_edge = torch.nn.Linear(2 * hidden_dim * 8 + hidden_dim * 8, 1)  # Output is a scalar

        self.dropout = dropout

    def forward(self, data):
        """
        Defines the forward pass of the EdgeGAT model.

        Args:
            data (Data): PyTorch Geometric Data object containing node features, edge indices, edge attributes, and target.

        Returns:
            torch.Tensor: Predicted values for each edge.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply the first Graph Attention Convolutional layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass through conv1.")

        # Apply the second Graph Attention Convolutional layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass through conv2.")

        # Apply the third Graph Attention Convolutional layer
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass through conv3.")

        # Process edge attributes through the MLP
        edge_features = self.edge_mlp(edge_attr)

        # Concatenate source node features, target node features, and edge features
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)

        # Pass the concatenated embeddings through the fully connected layer to get edge predictions
        edge_logits = self.fc_edge(edge_embeddings).squeeze()
        logger.debug("Computed edge logits.")

        return edge_logits  # Returns raw values for regression


class Trainer:
    """
    Trainer handles the training loop, including training and validation epochs,
    monitoring performance on a separate dataset, implementing early stopping,
    and saving the best model based on validation loss.
    """

    def __init__(self, model, optimizer, criterion, scheduler, device, num_epochs=500, patience=20,
                 results_dir='results'):
        """
        Initializes the Trainer with the specified model, optimizer, loss function, scheduler, and training parameters.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            criterion (torch.nn.Module): Loss function to be minimized.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            device (torch.device): Device to run the training on (CPU or GPU).
            num_epochs (int, optional): Maximum number of training epochs. Defaults to 500.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 20.
            results_dir (str, optional): Directory to save training results. Defaults to 'results'.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.results_dir = results_dir  # Directory to save results
        self.best_model_state = None
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, loader):
        """
        Executes a single training epoch.

        Args:
            loader (DataLoader): DataLoader for the training data.

        Returns:
            float: Average training loss for the epoch.
        """
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
        """
        Executes a single validation epoch.

        Args:
            loader (DataLoader): DataLoader for the validation data.

        Returns:
            float: Average validation loss for the epoch.
        """
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
        """
        Evaluates the model on a monitoring dataset, calculates metrics, and saves predictions.

        Args:
            monitoring_loader (DataLoader): DataLoader for the monitoring data.
            epoch (int): Current epoch number.
            data_module (DataModule): Instance of DataModule for inverse scaling.

        Returns:
            dict: Dictionary containing loss, MSE, MAE, and R² metrics.
        """
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

        # Inverse scaling of predictions and true values
        y_true = data_module.rau_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = data_module.rau_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Calculate regression metrics on inversely scaled values
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'r2_score': r2
        }

        # Save predictions per edge to a CSV file for analysis
        df_predictions = pd.DataFrame({
            'edge_id': edge_ids_all,
            'y_true': y_true,
            'y_pred': y_pred
        })

        # Define the filename and filepath for saving predictions
        csv_filename = f'monitoring_predictions_epoch_{epoch}.csv'
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        df_predictions.to_csv(csv_filepath, index=False)
        logger.info(f'Saved edge-wise predictions for epoch {epoch} at {csv_filepath}')

        return metrics

    def train_model(self, train_loader, val_loader, monitoring_loader=None, data_module=None):
        """
        Trains the model over multiple epochs, evaluates on validation and monitoring datasets,
        implements early stopping, and retains the best model based on validation loss.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            monitoring_loader (DataLoader, optional): DataLoader for the monitoring dataset. Defaults to None.
            data_module (DataModule, optional): Instance of DataModule for inverse scaling. Defaults to None.
        """
        best_val_loss = float('inf')
        patience_counter = 0

        self.monitoring_results = []

        logger.info("Starting the training process.")
        for epoch in range(1, self.num_epochs + 1):
            # Perform training and validation for the current epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            # Evaluate on the monitoring dataset if provided
            if monitoring_loader is not None and data_module is not None:
                monitoring_metrics = self.evaluate_monitoring_dataset(monitoring_loader, epoch, data_module)
                self.monitoring_results.append((epoch, monitoring_metrics))
                logger.info(
                    f'Epoch {epoch:03d}, Monitoring - Loss: {monitoring_metrics["loss"]:.4f}, '
                    f'MSE: {monitoring_metrics["mse"]:.4f}, MAE: {monitoring_metrics["mae"]:.4f}, R²: {monitoring_metrics["r2_score"]:.4f}'
                )

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Log training and validation loss every 10 epochs
            if epoch % 10 == 0:
                self.logger.info(
                    f'Epoch {epoch:03d}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}'
                )

            # Implement Early Stopping based on validation loss improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
                logger.debug(f"Epoch {epoch}: Improved validation loss to {val_loss:.4f}.")
            else:
                patience_counter += 1
                logger.debug(f"Epoch {epoch}: No improvement in validation loss.")

            # Check if patience threshold is reached for early stopping
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Load the best model state based on validation loss
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded the best model based on validation loss.")

    def save_model(self, path):
        """
        Saves the model's state dictionary to the specified path.

        Args:
            path (str): File path where the model will be saved.
        """
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'Model saved at: {path}')


class Evaluator:
    """
    Evaluator handles the evaluation of the trained model on test data,
    calculating performance metrics, and generating visualization plots.
    """

    def __init__(self, model, device, target_scaler):
        """
        Initializes the Evaluator with the trained model, device, and scaler for the target variable.

        Args:
            model (torch.nn.Module): The trained model to evaluate.
            device (torch.device): Device to run the evaluation on (CPU or GPU).
            target_scaler (StandardScaler): Scaler used for inverse transforming the target variable.
        """
        self.model = model
        self.device = device
        self.rau_scaler = target_scaler  # Scaler for the target variable 'RAU'
        self.logger = logging.getLogger(__name__)

    def test_model(self, loader):
        """
        Tests the model on the provided test dataset and returns the true and predicted values.

        Args:
            loader (DataLoader): DataLoader for the test data.

        Returns:
            tuple: Arrays of true and predicted target values.
        """
        y_true_scaled, y_pred_scaled = self._test(loader)
        y_true = self.rau_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = self.rau_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_true, y_pred

    def _test(self, loader):
        """
        Internal method to perform the testing loop.

        Args:
            loader (DataLoader): DataLoader for the test data.

        Returns:
            tuple: Arrays of scaled true and predicted target values.
        """
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
        logger.info("Model testing completed.")
        return y_true_scaled, y_pred_scaled

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates regression metrics (MSE, MAE, R²) between true and predicted values.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            tuple: Mean Squared Error, Mean Absolute Error, and R² score.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self.logger.info(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}')
        return mse, mae, r2

    def plot_metrics(self, y_true, y_pred):
        """
        Generates and displays plots for predicted vs. true values and residuals.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        self.plot_predictions(y_true, y_pred)
        self.plot_residuals(y_true, y_pred)

    def plot_predictions(self, y_true, y_pred):
        """
        Plots predicted values against true values to visualize model performance.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel('True RAU Values')
        plt.ylabel('Predicted RAU Values')
        plt.title('True vs. Predicted RAU Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        logger.debug("Generated plot for true vs. predicted RAU values.")

    def plot_residuals(self, y_true, y_pred):
        """
        Plots residuals (differences between true and predicted values) to assess model errors.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
        plt.xlabel('Predicted RAU Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True)
        plt.show()
        logger.debug("Generated residuals plot.")


def main():
    """
    The main function orchestrates the entire workflow:
    - Loads configuration.
    - Initializes the DataModule and loads all data.
    - Splits data into training, validation, and test sets.
    - Initializes the EdgeGAT model, optimizer, loss function, and scheduler.
    - Sets up the Trainer and Evaluator.
    - Loads the monitoring dataset.
    - Trains the model with early stopping.
    - Evaluates the model on the test dataset.
    - Generates evaluation metrics and plots.
    - Saves the trained model.
    """
    # Path to the configuration file relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # Two levels up
    config_file = os.path.join(project_root, 'config', 'config.yaml')  # Default path to config.yaml

    # Check if config file exists
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found at: {config_file}")
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Directory containing data files
    directory = config['paths']['folder_path_data']

    logger = logging.getLogger(__name__)

    # List of nodes with available measurement data
    included_nodes = config['nodes']['included_nodes']
    zfluss_wl_nodes = config['nodes']['zfluss_wl_nodes']

    # Initialize DataModule with specified directory and node configurations
    data_module = DataModule(directory, included_nodes, zfluss_wl_nodes)
    data_module.load_all_data()
    train_loader, val_loader, test_loader = data_module.get_loaders()

    # Check if DataLoaders were successfully created
    if not train_loader or not val_loader or not test_loader:
        logger.error("DataLoaders could not be created. Exiting the program.")
        return

    # Determine the device to run the training on (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Ensure there is at least one dataset loaded for training
    if len(data_module.datasets) == 0:
        logger.error("No datasets found for training the model.")
        raise ValueError("No datasets found for training the model.")

    # Get the number of node and edge features from the DataModule
    num_node_features = len(data_module.node_feature_columns)
    num_edge_features = len(data_module.edge_feature_columns)

    # Initialize the EdgeGAT model with the determined number of node and edge features
    model = EdgeGAT(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=64,
        dropout=0.15
    ).to(device)
    logger.info("Initialized EdgeGAT model.")

    # Initialize the AdamW optimizer with learning rate and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    logger.info("Initialized AdamW optimizer.")

    # Define the loss function for regression tasks
    criterion = torch.nn.MSELoss()
    logger.info("Initialized MSELoss as the loss function.")

    # Initialize the learning rate scheduler to reduce LR on plateau of validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    logger.info("Initialized ReduceLROnPlateau scheduler.")

    # Define the results directory for storing training outputs
    results_dir = os.path.join(project_root, 'results', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.debug(f"Created results directory at: {results_dir}")

    # Initialize the Trainer with the model, optimizer, loss function, scheduler, and training parameters
    trainer = Trainer(
        model, optimizer, criterion, scheduler, device,
        num_epochs=500, patience=20, results_dir=results_dir  # Pass results_dir
    )
    logger.info("Initialized Trainer.")

    # Define patterns to locate the monitoring dataset files
    monitoring_node_pattern = os.path.join(directory, '*_Roughness_0_combined_Node.csv')
    monitoring_edge_pattern = os.path.join(directory, '*_Roughness_0_combined_Pipes.csv')

    try:
        # Use glob to find monitoring node and edge files
        monitoring_node_files = glob.glob(monitoring_node_pattern)
        monitoring_edge_files = glob.glob(monitoring_edge_pattern)

        # Check if monitoring files are found
        if monitoring_node_files and monitoring_edge_files:
            monitoring_node_file = monitoring_node_files[0]
            monitoring_edge_file = monitoring_edge_files[0]
            monitoring_data = data_module.load_data(monitoring_node_file, monitoring_edge_file)
            monitoring_loader = DataLoader([monitoring_data], batch_size=16, shuffle=False)
            logger.info("Successfully loaded monitoring dataset.")
        else:
            logger.error("Monitoring dataset not found.")
            monitoring_loader = None
    except Exception as e:
        logger.error(f"Error loading monitoring dataset: {e}")
        monitoring_loader = None

    # Start training the model with the training and validation DataLoaders
    # If a monitoring dataset is available, it will be used for additional evaluation
    trainer.train_model(train_loader, val_loader, monitoring_loader=monitoring_loader, data_module=data_module)

    # Initialize the Evaluator with the trained model, device, and target scaler
    evaluator = Evaluator(model, device, data_module.rau_scaler)
    logger.info("Initialized Evaluator.")

    # Test the model on the test dataset to obtain true and predicted values
    y_true, y_pred = evaluator.test_model(test_loader)

    # Calculate and log regression metrics
    mse, mae, r2 = evaluator.calculate_metrics(y_true, y_pred)

    # Generate and display evaluation plots
    evaluator.plot_metrics(y_true, y_pred)

    # Define the directory to save the trained model
    models_dir = os.path.join(project_root, 'results', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.debug(f"Created models directory at: {models_dir}")

    # Define the path to save the trained model's state dictionary
    model_path = os.path.join(models_dir, 'edge_gat_model_regression.pth')
    trainer.save_model(model_path)

    logger.info("Program completed successfully.")


if __name__ == "__main__":
    main()
