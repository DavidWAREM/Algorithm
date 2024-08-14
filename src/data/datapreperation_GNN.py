import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
import logging

# Constants for column names
KNOTEN_NUMERICAL_COLUMNS = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP', 'XRECHTS', 'YHOCH', 'GEOH']
LEITUNGEN_NUMERICAL_COLUMNS = ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE']

class GraphDataset:
    def __init__(self, folder_path, save_path, normalize=False, standardize=False):
        self.folder_path = folder_path
        self.save_path = save_path
        self.data_list = []
        self.normalize = normalize
        self.standardize = standardize
        self.scaler = None
        self.load_datasets()
        self.save_datasets()

    @staticmethod
    def load_data_from_csv(file_path):
        logging.debug(f"Loading data from {file_path}")
        try:
            data = pd.read_csv(file_path, sep=';', encoding='latin1')  # Adjust encoding as needed
            logging.debug(f"Successfully loaded data from {file_path} with shape {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {file_path}: {e}")
            raise

    def get_matching_files(self):
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
        X_knoten = self.process_numerical_data(knoten_data, KNOTEN_NUMERICAL_COLUMNS)
        X_leitungen = self.process_numerical_data(leitungen_data, LEITUNGEN_NUMERICAL_COLUMNS)

        # Normalize or standardize the data if specified
        if self.normalize:
            logging.debug("Applying MinMaxScaler to normalize the data.")
            scaler = MinMaxScaler()
            X_knoten = scaler.fit_transform(X_knoten)
            X_leitungen = scaler.fit_transform(X_leitungen)
        elif self.standardize:
            logging.debug("Applying StandardScaler to standardize the data.")
            scaler = StandardScaler()
            X_knoten = scaler.fit_transform(X_knoten)
            X_leitungen = scaler.fit_transform(X_leitungen)

        # Combine node features using PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_knoten_poly = poly.fit_transform(X_knoten)

        # Convert node features to a tensor
        X_knoten_tensor = torch.tensor(X_knoten_poly, dtype=torch.float)

        # Convert edge features to a tensor
        X_leitungen_tensor = torch.tensor(X_leitungen, dtype=torch.float)

        # Create directed edge index based on edge data
        edge_index, missing_nodes, self_connections, duplicate_edges = self.create_edge_index(knoten_data, leitungen_data)

        # Extract target variable (assume 'RAU' as the target variable, adjust as necessary)
        y = torch.tensor(leitungen_data['RAU'].values, dtype=torch.float).view(-1, 1)

        # Create PyTorch Geometric Data object with directed edges
        graph_data = Data(x=X_knoten_tensor, edge_index=edge_index, edge_attr=X_leitungen_tensor, y=y)

        # Add the positional data as attributes to the Data object
        graph_data.pos = torch.tensor(knoten_data[['XRECHTS', 'YHOCH', 'GEOH']].values, dtype=torch.float)

        logging.debug(
            f"Created directed Data object with {len(edge_index[0])} edges, {len(missing_nodes)} missing nodes, {len(self_connections)} self-connections, and {len(duplicate_edges)} duplicate edges.")
        return graph_data

    def process_numerical_data(self, data, numerical_columns):
        X = data[numerical_columns]
        X = X.replace('?', 0).astype(float)
        return X

    def create_edge_index(self, knoten_data, leitungen_data):
        edges = []
        missing_nodes = set()
        duplicate_edges = set()
        self_connections = set()

        for _, row in leitungen_data.iterrows():
            start_node = row['ANFNAM']
            end_node = row['ENDNAM']

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

            edges.append([start_index, end_index])  # Directed edge
            duplicate_edges.add(edge)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index, missing_nodes, self_connections, duplicate_edges

    def load_datasets(self):
        matching_pairs = self.get_matching_files()

        for knoten_file, leitungen_file in matching_pairs:
            logging.debug(f"Processing pair: {knoten_file}, {leitungen_file}")
            try:
                knoten_file_path = os.path.join(self.folder_path, knoten_file)
                leitungen_file_path = os.path.join(self.folder_path, leitungen_file)

                knoten_data = self.load_data_from_csv(knoten_file_path)
                leitungen_data = self.load_data_from_csv(leitungen_file_path)

                logging.debug(f"Loaded knoten_data shape: {knoten_data.shape}")
                logging.debug(f"knoten_data head: \n{knoten_data.head()}")
                logging.debug(f"Loaded leitungen_data shape: {leitungen_data.shape}")
                logging.debug(f"leitungen_data head: \n{leitungen_data.head()}")

                graph_data = self.create_data_object(knoten_data, leitungen_data)

                self.data_list.append(graph_data)
                logging.debug(f"Successfully created Data object for {knoten_file} and {leitungen_file}")

            except Exception as e:
                logging.error(f"Failed to process pair: {knoten_file}, {leitungen_file}: {e}")

    def save_datasets(self):
        os.makedirs(self.save_path, exist_ok=True)
        for i, data in enumerate(self.data_list):
            file_path = os.path.join(self.save_path, f"data_{i}.pt")
            torch.save(data, file_path)
            logging.debug(f"Saved Data object to {file_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    folder_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher'
    save_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher\saved_data'

    # Create and save datasets
    graph_dataset = GraphDataset(folder_path, save_path, normalize=False, standardize=False)

    print(f"Number of datasets: {len(graph_dataset.data_list)}")
    for i, data in enumerate(graph_dataset.data_list):
        print(f"Dataset {i + 1} - Data object: {data}")

    # Visualize the first Data object
    first_data = graph_dataset.data_list[0]

    # Print basic info about the first data object
    print("\nFirst Data Object:")
    print(f"Number of nodes: {first_data.num_nodes}")
    print(f"Number of edges: {first_data.num_edges}")
    print(f"Node feature dimension: {first_data.x.size(1)}")
    print(f"Edge feature dimension: {first_data.edge_attr.size(1)}")
    print(f"Number of targets: {first_data.y.size(0)}")

    # Visualize the edge index
    print("\nEdge Index:")
    print(first_data.edge_index)

    # Visualize the node features (first 10 nodes)
    print("\nNode Features (First 10 nodes):")
    print(first_data.x[:10])

    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx


    # Function to visualize the graph
    def visualize_graph(data, show_labels=True):
        # Convert PyTorch Geometric data to a NetworkX graph
        G = to_networkx(data, to_undirected=False, edge_attrs=['edge_attr'], node_attrs=['x'])

        # Set up the plot
        plt.figure(figsize=(10, 10))

        # Get node positions
        pos = {i: (data.pos[i, 0].item(), data.pos[i, 1].item()) for i in range(data.num_nodes)}

        # Draw the graph
        nx.draw(G, pos, with_labels=show_labels, node_size=500, node_color='lightblue', font_size=10,
                font_weight='bold')

        # Draw node labels for easier interpretation
        if show_labels:
            labels = {i: f"Node {i}" for i in range(data.num_nodes)}
            nx.draw_networkx_labels(G, pos, labels, font_size=12)

        # Draw edge labels if any (e.g., edge attributes)
        edge_labels = {(u, v): f"{data.edge_attr[idx].item():.2f}" for idx, (u, v) in enumerate(G.edges())}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        plt.title("Graph Network Visualization")
        plt.show()


    # Assuming you have already created your GraphDataset instance and loaded datasets
    # Visualize the first data object in the list
    visualize_graph(first_data)
