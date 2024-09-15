import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

class GCNDataPreprocessor:
    def preprocess_data(self, nodes_df, edges_df):
        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

        edge_index = edges_df[['ANFNR', 'ENDNR']].values.T
        node_features = nodes_df.drop(columns=['KNAM']).values
        edge_labels = edges_df['RAU'].values

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(edge_labels, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        return data, y

    def split_data(self, datasets, test_size=0.2):
        # Preprocess each dataset
        processed_datasets = [(self.preprocess_data(nodes_df, edges_df)) for nodes_df, edges_df in datasets]
        # Split into training and testing
        return train_test_split(processed_datasets, test_size=test_size, random_state=42)
