# src/data/data_preparation.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import logging

from src.data.data_loader import graph_based_imputation, load_data

class DataModule:
    """
    DataModule handles data loading, preprocessing, scaling, and splitting.
    """
    def __init__(self, directory, all_physical_columns, geo_columns, included_nodes, zfluss_wl_nodes):
        """
        Initializes the DataModule with necessary parameters.

        Args:
            directory (str): Path to the data directory.
            all_physical_columns (list): List of all physical feature column names.
            geo_columns (list): List of geographical feature column names.
            included_nodes (list): List of node names to include.
            zfluss_wl_nodes (list): List of node names for 'ZUFLUSS_WL' processing.
        """
        self.directory = directory
        self.all_physical_columns = all_physical_columns
        self.geo_columns = geo_columns
        self.included_nodes = included_nodes
        self.zfluss_wl_nodes = zfluss_wl_nodes
        self.physical_scaler = StandardScaler()
        self.geo_scaler = MinMaxScaler()
        self.edge_scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def prepare_first_dataset(self):
        """
        Loads and prepares the first dataset to fit scalers.

        Returns:
            tuple: DataFrames for nodes and edges.
        """
        try:
            node_file_first = os.path.join(self.directory, 'SyntheticData-Spechbach_Valve_1_combined_Node.csv')
            edge_file_first = os.path.join(self.directory, 'SyntheticData-Spechbach_Valve_1_combined_Pipes.csv')

            nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
            edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')
            self.logger.info("First dataset loaded successfully.")

            # Mark included nodes
            nodes_df_first['Included'] = nodes_df_first['KNAM'].isin(self.included_nodes)
            for col in self.all_physical_columns:
                if col != 'ZUFLUSS_WL':
                    nodes_df_first.loc[~nodes_df_first['Included'], col] = np.nan

            # Handle 'ZUFLUSS_WL' only for specified nodes
            nodes_df_first['ZUFLUSS_WL'] = nodes_df_first.apply(
                lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in self.zfluss_wl_nodes else np.nan,
                axis=1
            )

            # Add missing indicators
            for col in self.all_physical_columns:
                nodes_df_first[f'{col}_missing'] = nodes_df_first[col].isna().astype(float)

            # KNN Imputation
            imputer = KNNImputer(n_neighbors=5)
            adjusted_physical_columns = [col for col in self.all_physical_columns if col != 'ZUFLUSS_WL']
            nodes_df_first[adjusted_physical_columns] = imputer.fit_transform(nodes_df_first[adjusted_physical_columns])
            self.logger.info("KNN imputation completed for the first dataset.")

            # Graph-based Imputation for 'ZUFLUSS_WL'
            node_mapping_first = {name: idx for idx, name in enumerate(nodes_df_first['KNAM'])}
            edges_df_first['ANFNR'] = edges_df_first['ANFNAM'].map(node_mapping_first)
            edges_df_first['ENDNR'] = edges_df_first['ENDNAM'].map(node_mapping_first)
            edge_index_first = edges_df_first[['ANFNR', 'ENDNR']].values.T

            nodes_df_first = graph_based_imputation(nodes_df_first, edge_index_first, 'ZUFLUSS_WL')

            # Drop helper column
            nodes_df_first = nodes_df_first.drop(columns=['Included'])
            self.logger.info("Graph-based imputation completed for 'ZUFLUSS_WL'.")

            return nodes_df_first, edges_df_first

        except FileNotFoundError as fnf_error:
            self.logger.error(f"File not found: {fnf_error.filename}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("No data: The first CSV file is empty.")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during preparing the first dataset: {e}")
            raise

    def scale_data(self, nodes_df, edges_df, edge_columns):
        """
        Fits scalers on the first dataset and scales all datasets.

        Args:
            nodes_df (pd.DataFrame): Nodes DataFrame.
            edges_df (pd.DataFrame): Edges DataFrame.
            edge_columns (list): List of edge feature column names.

        Returns:
            tuple: Scaled nodes and edges DataFrames.
        """
        try:
            # Fit scalers
            self.physical_scaler.fit(nodes_df[self.all_physical_columns])
            self.geo_scaler.fit(nodes_df[self.geo_columns])
            self.edge_scaler.fit(edges_df[edge_columns])
            self.logger.info("Scalers fitted on the first dataset.")

            # Scale data
            nodes_df[self.all_physical_columns] = self.physical_scaler.transform(nodes_df[self.all_physical_columns])
            nodes_df[self.geo_columns] = self.geo_scaler.transform(nodes_df[self.geo_columns])
            edges_df[edge_columns] = self.edge_scaler.transform(edges_df[edge_columns])
            self.logger.info("Scaling of features completed.")

            return nodes_df, edges_df

        except Exception as e:
            self.logger.error(f"An error occurred during scaling data: {e}")
            raise

    def load_datasets(self, num_valves=108):
        """
        Loads all datasets applying scaling and encoding.

        Args:
            num_valves (int, optional): Number of valve datasets to load. Defaults to 108.

        Returns:
            list: List of PyTorch Geometric Data objects.
        """
        datasets = []
        for i in range(1, num_valves + 1):
            node_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Valve_{i}_combined_Node.csv')
            edge_file = os.path.join(self.directory, f'SyntheticData-Spechbach_Valve_{i}_combined_Pipes.csv')
            try:
                data = load_data(
                    node_file, edge_file,
                    self.physical_scaler, self.geo_scaler, self.edge_scaler,
                    self.included_nodes, self.zfluss_wl_nodes
                )
                datasets.append(data)
                self.logger.info(f"Valve {i} data loaded successfully.")
            except Exception as e:
                self.logger.error(f"Error loading data for Valve {i}: {e}")
                continue
        return datasets

    def prepare(self):
        """
        Prepares the data by loading, preprocessing, and scaling the first dataset.

        Returns:
            tuple: Scaled nodes DataFrame, edges DataFrame, and edge feature columns.
        """
        try:
            nodes_df_first, edges_df_first = self.prepare_first_dataset()
            # One-Hot Encoding for 'ROHRTYP'
            edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP')
            self.logger.info("One-Hot Encoding for 'ROHRTYP' completed.")

            # Drop 'RAU' if it exists
            if 'RAU' in edges_df_first.columns:
                edges_df_first = edges_df_first.drop(columns=['RAU'])
                self.logger.info("Dropped 'RAU' column from edges.")

            # Update edge columns after One-Hot Encoding
            edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df_first.filter(like='ROHRTYP').columns)

            # Scale data
            nodes_df_first, edges_df_first = self.scale_data(nodes_df_first, edges_df_first, edge_columns)

            return nodes_df_first, edges_df_first, edge_columns

        except Exception as e:
            self.logger.error(f"An error occurred during data preparation: {e}")
            raise
