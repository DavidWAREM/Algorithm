import os
import pandas as pd
import torch
from torch_geometric.data import Data
import logging
from create_dataset import GraphDataLoader

# Setup logging
logging.basicConfig(
    filename='verify_graph.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite the log file each time
)

# Log start of the script
logging.info("Starting the graph verification script")

# Constants for column names
KNOTEN_NUMERICAL_COLUMNS = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP', 'XRECHTS', 'YHOCH', 'GEOH']
LEITUNGEN_NUMERICAL_COLUMNS = ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DP', 'DPREL', 'RAISE']

def load_csv_data(file_path):
    try:
        data = pd.read_csv(file_path, sep=';', encoding='latin1')
        logging.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def verify_node_attributes(graph_data, knoten_data):
    X_knoten = knoten_data[KNOTEN_NUMERICAL_COLUMNS].replace('?', 0).astype(float)

    # Debug-Ausgabe der ersten paar Zeilen der XRECHTS-Spalte
    print("Erste paar Zeilen der XRECHTS-Spalte aus der CSV-Datei:")
    print(X_knoten['XRECHTS'].head())

    print("Erste paar Werte der XRECHTS-Spalte aus dem Data-Objekt:")
    print(graph_data.x[:, 5][:5])  # XRECHTS ist die 6. Spalte in KNOTEN_NUMERICAL_COLUMNS, daher Index 5

    # Check the features of the first node
    for i in range(len(KNOTEN_NUMERICAL_COLUMNS)):
        original_value = round(X_knoten.iloc[0, i], 5)
        graph_value = round(graph_data.x[0, i].item(), 5)
        if abs(original_value - graph_value) >= 1:
            print(
                f"Mismatch in node attribute {KNOTEN_NUMERICAL_COLUMNS[i]}: original={original_value}, graph={graph_value}")
        assert abs(original_value - graph_value) < 1, f"Mismatch in node attribute {KNOTEN_NUMERICAL_COLUMNS[i]}"
    logging.info("Node attributes are correct.")

def verify_edge_index(graph_data, knoten_data, leitungen_data):
    edges = []
    for _, row in leitungen_data.iterrows():
        start_node = row['ANFNAM']
        end_node = row['ENDNAM']

        if start_node in knoten_data['KNAM'].values and end_node in knoten_data['KNAM'].values:
            start_index = knoten_data.index[knoten_data['KNAM'] == start_node].tolist()[0]
            end_index = knoten_data.index[knoten_data['KNAM'] == end_node].tolist()[0]
            edges.append([start_index, end_index])
            edges.append([end_index, start_index])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    assert torch.equal(graph_data.edge_index, edge_index), "Mismatch in edge index"
    logging.info("Edge index is correct.")

def verify_graph_data(graph_data, knoten_file_path, leitungen_file_path):
    knoten_data = load_csv_data(knoten_file_path)
    leitungen_data = load_csv_data(leitungen_file_path)

    verify_node_attributes(graph_data, knoten_data)
    verify_edge_index(graph_data, knoten_data, leitungen_data)
    logging.info("Graph data verification completed successfully.")

# Example usage
folder_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher'
save_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher\saved_data'

# Load the saved datasets
loaded_data_list = GraphDataLoader.load_saved_datasets(save_path)
logging.info(f"Loaded {len(loaded_data_list)} datasets")

# Verify the first graph in the list
if loaded_data_list:
    knoten_file_path = os.path.join(folder_path, 'export_results_13_Spechbach_RNAB.TXT_0_Node.csv')
    leitungen_file_path = os.path.join(folder_path, 'export_results_13_Spechbach_RNAB.TXT_0_Pipes.csv')
    logging.info(f"Verifying graph data for {knoten_file_path} and {leitungen_file_path}")
    verify_graph_data(loaded_data_list[0], knoten_file_path, leitungen_file_path)

# Output to console
print(f"Number of datasets: {len(loaded_data_list)}")
for i, data in enumerate(loaded_data_list):
    print(f"Dataset {i + 1} - Data object: {data}")

logging.info("Script completed")
