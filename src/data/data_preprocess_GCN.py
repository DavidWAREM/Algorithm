import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

class GCNDataPreprocessor:
    def preprocess_data(self, nodes_df, edges_df):
        logging.debug("Starting to preprocess data...")

        # Create node-to-index mapping
        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

        # Scaling node features
        geographical_features = nodes_df[['XRECHTS', 'YHOCH']].values
        other_features = nodes_df.drop(columns=['KNAM', 'XRECHTS', 'YHOCH']).values

        geo_scaler = MinMaxScaler()
        scaled_geo_features = geo_scaler.fit_transform(geographical_features)

        other_scaler = StandardScaler()
        scaled_other_features = other_scaler.fit_transform(other_features)

        scaled_node_features = torch.tensor(
            np.hstack([scaled_geo_features, scaled_other_features]), dtype=torch.float
        )

        # Edge features including RAU (target)
        edge_features = edges_df[['RAU', 'RORL', 'DM', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE']].values

        # Extract RAU as the target (y)
        target_y = torch.tensor(edge_features[:, 0], dtype=torch.float)

        # Use the rest as edge features
        edge_features_tensor = torch.tensor(edge_features[:, 1:], dtype=torch.float)

        # Prepare edge_index
        edge_index = torch.tensor(edges_df[['ANFNR', 'ENDNR']].values.T, dtype=torch.long)

        # Create Data object
        data = Data(x=scaled_node_features, edge_index=edge_index, edge_attr=edge_features_tensor)

        logging.info("Data preprocessing completed successfully.")
        return data, target_y

    def split_data(self, datasets, test_size=0.2):
        # Preprocess each dataset
        processed_datasets = [(self.preprocess_data(nodes_df, edges_df)) for nodes_df, edges_df in datasets]
        # Split into training and testing
        return train_test_split(processed_datasets, test_size=test_size, random_state=42)