import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import PolynomialFeatures
import logging

# Setup logging to create a log file named 'create_dataset.log'
logging.basicConfig(
    filename='create_dataset.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite the log file each time
)

# Constants for column names
KNOTEN_NUMERICAL_COLUMNS = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP', 'XRECHTS', 'YHOCH', 'GEOH']
LEITUNGEN_NUMERICAL_COLUMNS = ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE']

class GraphDataset:
    """
    This class handles loading and processing of node and edge data
    from CSV files to create PyTorch Geometric Data objects.
    """

    def __init__(self, folder_path, save_path):
        self.folder_path = folder_path
        self.save_path = save_path
        self.data_list = []
        self.load_datasets()
        self.save_datasets()

    @staticmethod
    def load_data_from_csv(file_path):
        """
        Load data from a CSV file.

        Parameters:
        - file_path: Path to the CSV file.

        Returns:
        - DataFrame containing the loaded data.
        """
        logging.debug(f"Loading data from {file_path}")
        try:
            data = pd.read_csv(file_path, sep=';', encoding='latin1')  # Adjust encoding as needed
            logging.debug(f"Successfully loaded data from {file_path} with shape {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {file_path}: {e}")
            raise

    def get_matching_files(self):
        """
        Find and pair matching node and edge CSV files in the specified folder.

        Returns:
        - List of tuples containing matching pairs of node and edge filenames.
        """
        knoten_files = [f for f in os.listdir(self.folder_path) if f.endswith('_Node.csv')]
        leitungen_files = [f for f in os.listdir(self.folder_path) if f.endswith('_Pipes.csv')]

        matching_pairs = []
        for knoten_file in knoten_files:
            base_name = knoten_file.replace('_Node.csv', '')
            matching_leitungen_file = f"{base_name}_Pipes.csv".replace('_Node', '')
            if matching_leitungen_file in leitungen_files:
                matching_pairs.append((knoten_file, matching_leitungen_file))

        logging.debug(f"Found {len(matching_pairs)} matching pairs of Knoten and Leitungen files")
        return matching_pairs

    def create_data_object(self, knoten_data, leitungen_data):
        """
        Create a PyTorch Geometric Data object from node and edge data.

        Parameters:
        - knoten_data: DataFrame containing node data.
        - leitungen_data: DataFrame containing edge data.

        Returns:
        - PyTorch Geometric Data object.
        """
        X_knoten = self.process_numerical_data(knoten_data, KNOTEN_NUMERICAL_COLUMNS)
        X_leitungen = self.process_numerical_data(leitungen_data, LEITUNGEN_NUMERICAL_COLUMNS)

        # Combine node features using PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_knoten_poly = poly.fit_transform(X_knoten)

        # Convert node features to a tensor
        X_knoten_tensor = torch.tensor(X_knoten_poly, dtype=torch.float)

        # Create edge index based on edge data
        edge_index, missing_nodes, self_connections, duplicate_edges = self.create_edge_index(knoten_data, leitungen_data)

        # Extract target variable (assume 'RAU' as the target variable, adjust as necessary)
        y = torch.tensor(knoten_data['PRECH'].values, dtype=torch.float).view(-1, 1)

        # Create PyTorch Geometric Data object
        graph_data = Data(x=X_knoten_tensor, edge_index=edge_index, y=y)

        # Add the positional data as attributes to the Data object
        graph_data.pos = torch.tensor(knoten_data[['XRECHTS', 'YHOCH', 'GEOH']].values, dtype=torch.float)

        logging.debug(
            f"Created Data object with {len(edge_index[0]) // 2} edges, {len(missing_nodes)} missing nodes, {len(self_connections)} self-connections, and {len(duplicate_edges)} duplicate edges.")
        return graph_data

    def process_numerical_data(self, data, numerical_columns):
        """
        Process numerical data by selecting relevant columns and handling missing values.

        Parameters:
        - data: DataFrame containing the data.
        - numerical_columns: List of numerical columns to select.

        Returns:
        - Processed DataFrame with numerical data.
        """
        X = data[numerical_columns]
        # Handle missing values by filling with zeros or another strategy
        X = X.replace('?', 0).astype(float)
        return X

    def create_edge_index(self, knoten_data, leitungen_data):
        """
        Create edge index tensor from node and edge data.

        Parameters:
        - knoten_data: DataFrame containing node data.
        - leitungen_data: DataFrame containing edge data.

        Returns:
        - edge_index: Tensor containing edge indices.
        - missing_nodes: Set of missing node IDs.
        - self_connections: Set of nodes with self-connections.
        - duplicate_edges: Set of duplicate edges.
        """
        edges = []
        missing_nodes = set()
        duplicate_edges = set()
        self_connections = set()

        for _, row in leitungen_data.iterrows():
            start_node = row['ANFNAM']
            end_node = row['ENDNAM']

            # Check if both start and end nodes exist in the node data
            if start_node not in knoten_data['KNAM'].values:
                missing_nodes.add(start_node)
                logging.warning(f"Missing start node: {start_node}")
                continue
            if end_node not in knoten_data['KNAM'].values:
                missing_nodes.add(end_node)
                logging.warning(f"Missing end node: {end_node}")
                continue

            if start_node == end_node:
                self_connections.add(start_node)
                logging.warning(f"Self-connection found at node: {start_node}")
                continue

            start_index = knoten_data.index[knoten_data['KNAM'] == start_node].tolist()[0]
            end_index = knoten_data.index[knoten_data['KNAM'] == end_node].tolist()[0]
            edge = (start_index, end_index)

            if edge in duplicate_edges or (end_index, start_index) in duplicate_edges:
                logging.warning(f"Duplicate edge found between nodes: {start_node} and {end_node}")
                continue

            edges.append([start_index, end_index])
            edges.append([end_index, start_index])  # Assuming undirected graph
            duplicate_edges.add(edge)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index, missing_nodes, self_connections, duplicate_edges

    def load_datasets(self):
        """
        Load and process datasets, creating PyTorch Geometric Data objects.
        """
        matching_pairs = self.get_matching_files()

        for knoten_file, leitungen_file in matching_pairs:
            logging.debug(f"Processing pair: {knoten_file}, {leitungen_file}")
            try:
                # Construct paths to node and pipe files
                knoten_file_path = os.path.join(self.folder_path, knoten_file)
                leitungen_file_path = os.path.join(self.folder_path, leitungen_file)

                # Load data from CSV files
                knoten_data = self.load_data_from_csv(knoten_file_path)
                leitungen_data = self.load_data_from_csv(leitungen_file_path)

                # Create Data object
                graph_data = self.create_data_object(knoten_data, leitungen_data)

                # Append the Data object to the list
                self.data_list.append(graph_data)
                logging.debug(f"Successfully created Data object for {knoten_file} and {leitungen_file}")

            except Exception as e:
                logging.error(f"Failed to process pair: {knoten_file}, {leitungen_file}: {e}")

    def save_datasets(self):
        """
        Save the processed datasets to files for future use.
        """
        os.makedirs(self.save_path, exist_ok=True)
        for i, data in enumerate(self.data_list):
            file_path = os.path.join(self.save_path, f"data_{i}.pt")
            torch.save(data, file_path)
            logging.debug(f"Saved Data object to {file_path}")

class GraphDataLoader:
    """
    This class handles creating and managing the DataLoader for PyTorch Geometric Data objects.
    """

    def __init__(self, data_list, batch_size=1, shuffle=True):
        self.dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
        self.log_dataset_info(data_list)

    @staticmethod
    def log_dataset_info(data_list):
        """
        Log information about the datasets.

        Parameters:
        - data_list: List of PyTorch Geometric Data objects.
        """
        logging.debug(f"Number of datasets: {len(data_list)}")
        for i, data in enumerate(data_list):
            logging.debug(f"Dataset {i + 1} - Data object: {data}")

    def get_dataloader(self):
        """
        Get the DataLoader.

        Returns:
        - DataLoader object.
        """
        return self.dataloader

    @staticmethod
    def load_saved_datasets(save_path):
        """
        Load saved datasets from files.

        Parameters:
        - save_path: Path to the directory containing saved dataset files.

        Returns:
        - List of loaded PyTorch Geometric Data objects.
        """
        data_list = []
        for file_name in os.listdir(save_path):
            if file_name.endswith('.pt'):
                file_path = os.path.join(save_path, file_name)
                data = torch.load(file_path)
                data_list.append(data)
                logging.debug(f"Loaded Data object from {file_path}")
        return data_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Usage example
    folder_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher'
    save_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher\saved_data'

    # Create and save datasets
    graph_dataset = GraphDataset(folder_path, save_path)

    # Load datasets from saved files
    loaded_data_list = GraphDataLoader.load_saved_datasets(save_path)
    data_loader = GraphDataLoader(loaded_data_list)



    # Output to console as well
    print(f"Number of datasets: {len(graph_dataset.data_list)}")
    for i, data in enumerate(graph_dataset.data_list):
        print(f"Dataset {i + 1} - Data object: {data}")


