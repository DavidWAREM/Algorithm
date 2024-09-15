import pandas as pd

class GCNDataLoader:
    def __init__(self, folder_path_data, num_datasets=10000):
        self.folder_path_data = folder_path_data
        self.num_datasets = num_datasets

    def load_all_data(self):
        datasets = []
        for i in range(1, self.num_datasets + 1):
            node_file = f'{self.folder_path_data}/SyntheticData-Spechbach_Roughness_{i}_Node.csv'
            edge_file = f'{self.folder_path_data}/SyntheticData-Spechbach_Roughness_{i}_Pipes.csv'
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
            datasets.append((nodes_df, edges_df))
        return datasets
