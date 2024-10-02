import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, auc
import logging

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG if you want to see debug logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data Preparation Class
class DataModule:
    def __init__(self, directory, included_nodes, zfluss_wl_nodes, num_valves=108):
        self.directory = directory
        self.included_nodes = included_nodes
        self.zfluss_wl_nodes = zfluss_wl_nodes
        self.num_valves = num_valves
        self.physical_scaler = StandardScaler()
        self.geo_scaler = MinMaxScaler()
        self.edge_scaler = StandardScaler()
        self.datasets = []
        self.geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
        self.adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
        self.additional_physical_columns = ['ZUFLUSS_WL']
        self.all_physical_columns = self.adjusted_physical_columns + self.additional_physical_columns

    def add_positional_encoding(self, df, columns, max_value=10000):
        for col in columns:
            df[f'{col}_sin'] = np.sin(df[col] * (2 * pi / max_value))
            df[f'{col}_cos'] = np.cos(df[col] * (2 * pi / max_value))
        return df

    def graph_based_imputation(self, df, edge_index, feature_name):
        node_values = df[feature_name].values
        missing_mask = np.isnan(node_values)
        # Create adjacency list
        adjacency = {i: [] for i in range(len(df))}
        for src, dst in edge_index.T:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
        # Iterate over missing values
        for idx in np.where(missing_mask)[0]:
            neighbors = adjacency[idx]
            neighbor_values = [node_values[n] for n in neighbors if not np.isnan(node_values[n])]
            if neighbor_values:
                node_values[idx] = np.mean(neighbor_values)
            else:
                node_values[idx] = np.nanmean(node_values)
        df[feature_name] = node_values
        logger.debug(f"Performed graph-based imputation for feature '{feature_name}'.")
        return df

    def load_data(self, node_file, edge_file):
        try:
            # Load data
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
            logger.debug(f"Loaded files: {node_file}, {edge_file}.")
        except Exception as e:
            logger.error(f"Error loading files {node_file} or {edge_file}: {e}")
            raise e

        # Check if required columns are present
        required_node_columns = ['KNAM', 'XRECHTS', 'YHOCH', 'GEOH'] + self.adjusted_physical_columns + ['ZUFLUSS_WL']
        for col in required_node_columns:
            if col not in nodes_df.columns:
                logger.debug(f"Column {col} is missing in {node_file}.")
                raise ValueError(f"Column {col} is missing in {node_file}.")

        required_edge_columns = ['ANFNAM', 'ENDNAM', 'FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL', 'ROHRTYP', 'RORL', 'DM', 'RAISE']
        for col in required_edge_columns:
            if col not in edges_df.columns:
                logger.debug(f"Column {col} is missing in {edge_file}.")
                raise ValueError(f"Column {col} is missing in {edge_file}.")

        # Remove 'RAU' feature directly after loading
        if 'RAU' in edges_df.columns:
            edges_df = edges_df.drop(columns=['RAU'])
            logger.debug("'RAU' column found and removed from edge data.")

        node_mapping = {name: idx for idx, name in enumerate(nodes_df['KNAM'])}
        nodes_df['node_idx'] = nodes_df['KNAM'].map(node_mapping)
        edges_df['ANFNR'] = edges_df['ANFNAM'].map(node_mapping)
        edges_df['ENDNR'] = edges_df['ENDNAM'].map(node_mapping)

        edges_df = pd.get_dummies(edges_df, columns=['ROHRTYP'], prefix='ROHRTYP')
        edge_index = edges_df[['ANFNR', 'ENDNR']].values.T

        # Ensure relevant columns are numeric
        edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']] = edges_df[['FLUSS_WL', 'FLUSS_WOL', 'VM_WL', 'VM_WOL']].astype(float)
        logger.debug("Converted relevant edge columns to float.")

        # Define a small epsilon for approximation to zero
        epsilon = 1e-6

        # Create target variable
        target_condition = (
            (edges_df['FLUSS_WL'].abs() < epsilon) |
            (edges_df['FLUSS_WOL'].abs() < epsilon) |
            (edges_df['VM_WL'].abs() < epsilon) |
            (edges_df['VM_WOL'].abs() < epsilon)
        )
        y = torch.tensor(target_condition.astype(float).values, dtype=torch.float)
        logger.debug("Created target variable based on flow conditions.")

        # Adjust node attributes
        # Set values of adjusted_physical_columns to NaN for nodes not in included_nodes
        nodes_df['Included'] = nodes_df['KNAM'].isin(self.included_nodes)
        for col in self.adjusted_physical_columns:
            nodes_df.loc[~nodes_df['Included'], col] = np.nan
            logger.debug(f"Set {col} to NaN for nodes not included.")

        # Handle ZUFLUSS_WL only for specific nodes
        nodes_df['ZUFLUSS_WL'] = nodes_df.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in self.zfluss_wl_nodes else np.nan,
            axis=1
        )
        logger.debug("Handled 'ZUFLUSS_WL' for specific nodes.")

        # Add indicator columns for missing values
        for col in self.all_physical_columns:
            nodes_df[f'{col}_missing'] = nodes_df[col].isna().astype(float)
            logger.debug(f"Added missing indicator for {col}.")

        # Graph-based imputation for missing ZUFLUSS_WL values
        nodes_df = self.graph_based_imputation(nodes_df, edge_index, 'ZUFLUSS_WL')

        # Handle missing values for other physical columns using KNN Imputer
        imputer = KNNImputer(n_neighbors=5)
        nodes_df[self.adjusted_physical_columns] = imputer.fit_transform(nodes_df[self.adjusted_physical_columns])
        logger.debug("Performed KNN imputation for adjusted physical columns.")

        # Remove the helper column
        nodes_df = nodes_df.drop(columns=['Included'])
        logger.debug("Dropped the 'Included' column from node data.")

        # Apply scaling
        nodes_df[self.all_physical_columns] = self.physical_scaler.transform(nodes_df[self.all_physical_columns])
        nodes_df[self.geo_columns] = self.geo_scaler.transform(nodes_df[self.geo_columns])
        logger.debug("Applied scaling to physical and geographic columns.")

        # Add positional encoding
        nodes_df = self.add_positional_encoding(nodes_df, self.geo_columns)
        logger.debug("Added positional encoding to geographic columns.")

        # Create node features
        node_features = nodes_df.drop(columns=['KNAM', 'node_idx']).values
        logger.debug("Created node features.")

        # Update edge_columns after One-Hot Encoding
        edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df.filter(like='ROHRTYP').columns)

        # Scale edge attributes
        edges_df[edge_columns] = self.edge_scaler.transform(edges_df[edge_columns])
        logger.debug("Scaled edge attributes.")

        edge_attributes = edges_df[edge_columns].values

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        logger.debug("Created PyTorch Geometric Data object.")
        return data

    def fit_scalers(self):
        node_file_first = f'{self.directory}SyntheticData-Spechbach_Valve_1_combined_Node.csv'
        edge_file_first = f'{self.directory}SyntheticData-Spechbach_Valve_1_combined_Pipes.csv'

        try:
            nodes_df_first = pd.read_csv(node_file_first, delimiter=';', decimal='.')
            edges_df_first = pd.read_csv(edge_file_first, delimiter=';', decimal='.')
            logger.debug(f"Loaded files for scaler fitting: {node_file_first}, {edge_file_first}.")
        except Exception as e:
            logger.error(f"Error loading files {node_file_first} or {edge_file_first} for scaler fitting: {e}")
            raise e

        # Remove 'RAU' feature if present
        if 'RAU' in edges_df_first.columns:
            edges_df_first = edges_df_first.drop(columns=['RAU'])
            logger.debug("'RAU' column found and removed from edge data during scaler fitting.")

        # Set adjusted_physical_columns to NaN for nodes not in included_nodes
        nodes_df_first['Included'] = nodes_df_first['KNAM'].isin(self.included_nodes)
        for col in self.adjusted_physical_columns:
            nodes_df_first.loc[~nodes_df_first['Included'], col] = np.nan
            logger.debug(f"Set {col} to NaN for nodes not included during scaler fitting.")

        # Handle ZUFLUSS_WL only for specific nodes
        nodes_df_first['ZUFLUSS_WL'] = nodes_df_first.apply(
            lambda row: row['ZUFLUSS_WL'] if row['KNAM'] in self.zfluss_wl_nodes else np.nan,
            axis=1
        )
        logger.debug("Handled 'ZUFLUSS_WL' for specific nodes during scaler fitting.")

        # Add indicator columns for missing values
        for col in self.all_physical_columns:
            nodes_df_first[f'{col}_missing'] = nodes_df_first[col].isna().astype(float)
            logger.debug(f"Added missing indicator for {col} during scaler fitting.")

        # Graph-based imputation for missing ZUFLUSS_WL values
        node_mapping_first = {name: idx for idx, name in enumerate(nodes_df_first['KNAM'])}
        edges_df_first['ANFNR'] = edges_df_first['ANFNAM'].map(node_mapping_first)
        edges_df_first['ENDNR'] = edges_df_first['ENDNAM'].map(node_mapping_first)
        edge_index_first = edges_df_first[['ANFNR', 'ENDNR']].values.T

        nodes_df_first = self.graph_based_imputation(nodes_df_first, edge_index_first, 'ZUFLUSS_WL')

        # Remove the helper column
        nodes_df_first = nodes_df_first.drop(columns=['Included'])
        logger.debug("Dropped the 'Included' column from node data during scaler fitting.")

        # Fit scalers
        self.physical_scaler.fit(nodes_df_first[self.all_physical_columns])
        self.geo_scaler.fit(nodes_df_first[self.geo_columns])
        logger.debug("Fitted physical and geographic scalers.")

        # One-Hot Encoding for 'ROHRTYP'
        edges_df_first = pd.get_dummies(edges_df_first, columns=['ROHRTYP'], prefix='ROHRTYP')
        logger.debug("Performed One-Hot Encoding for 'ROHRTYP' during scaler fitting.")

        # Update edge_columns after One-Hot Encoding
        edge_columns = ['RORL', 'DM', 'RAISE'] + list(edges_df_first.filter(like='ROHRTYP').columns)

        # Fit edge scaler
        self.edge_scaler.fit(edges_df_first[edge_columns])
        logger.debug("Fitted edge scaler.")

    def load_all_data(self):
        self.fit_scalers()
        logger.info("Started loading all datasets.")
        for i in range(1, self.num_valves + 1):
            node_file = f'{self.directory}SyntheticData-Spechbach_Valve_{i}_combined_Node.csv'
            edge_file = f'{self.directory}SyntheticData-Spechbach_Valve_{i}_combined_Pipes.csv'
            try:
                data = self.load_data(node_file, edge_file)
                self.datasets.append(data)
                logger.info(f'Dataset {i} loaded successfully.')
            except Exception as e:
                logger.error(f'Error loading Dataset {i}: {e}')
                continue
        logger.info("Completed loading all datasets.")

    def get_loaders(self, val_size=0.25, test_size=0.2, random_state=42):
        if not self.datasets:
            logger.error("No datasets available.")
            return None, None, None

        # Split data into training, validation, and test sets
        train_val_data, test_data = train_test_split(
            self.datasets, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)  # Adjust validation ratio
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_ratio, random_state=random_state
        )
        logger.info("Split data into training, validation, and test sets.")

        # Check for positive examples
        train_positive = sum([data.y.sum().item() for data in train_data])
        val_positive = sum([data.y.sum().item() for data in val_data])
        test_positive = sum([data.y.sum().item() for data in test_data])

        if train_positive == 0 or val_positive == 0 or test_positive == 0:
            logger.warning("No positive examples in one of the datasets.")

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        logger.info("Created DataLoaders for training, validation, and test sets.")

        return train_loader, val_loader, test_loader

# GAT Model with Edge Prediction for Binary Classification
class EdgeGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=16, dropout=0.2):
        super(EdgeGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 4)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_edge = torch.nn.Linear(2 * hidden_dim + hidden_dim, 1)  # Output is a logit
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass through conv1.")

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logger.debug("Forward pass through conv2.")

        x = self.conv3(x, edge_index)
        logger.debug("Forward pass through conv3.")

        edge_features = self.edge_mlp(edge_attr)
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_features], dim=1)
        edge_logits = self.fc_edge(edge_embeddings).squeeze()
        logger.debug("Computed edge logits.")

        return edge_logits  # Return logits

# Trainer Class
class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, num_epochs=300, patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.best_model_state = None
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.debug(f"Training epoch loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.debug(f"Validation epoch loss: {avg_loss:.4f}")
        return avg_loss

    def train_model(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting training process.")
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.scheduler.step(val_loss)

            if epoch % 10 == 0:
                self.logger.info(
                    f'Epoch {epoch:03d}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}'
                )

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
                logger.debug(f"Epoch {epoch}: Improved validation loss to {val_loss:.4f}.")
            else:
                patience_counter += 1
                logger.debug(f"Epoch {epoch}: No improvement in validation loss.")

            if patience_counter >= self.patience:
                self.logger.info(f"Early Stopping at Epoch {epoch}.")
                break

        # Load the best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded the best model based on validation loss.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'Model saved at: {path}')

# Evaluator Class
class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)

    def test_model(self, loader):
        y_true, y_pred = self._test(loader)
        return y_true, y_pred

    def _test(self, loader):
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                y_true_list.append(batch.y.cpu().numpy())
                y_pred_list.append(probs.cpu().numpy())
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        logger.info("Completed model testing.")
        return y_true, y_pred

    def calculate_metrics(self, y_true, y_pred):
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            self.logger.warning("y_true contains only one class. ROC AUC Score cannot be calculated.")
            auc_score = None
        else:
            auc_score = roc_auc_score(y_true, y_pred)
            self.logger.info(f'ROC AUC Score: {auc_score:.4f}')

        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)
        self.logger.info(f'Accuracy: {accuracy:.4f}')

        return accuracy, auc_score, y_pred_binary

    def plot_metrics(self, y_true, y_pred):
        self.plot_roc_curve(y_true, y_pred)
        self.plot_precision_recall(y_true, y_pred)
        self.plot_confusion_matrix(y_true, (y_pred >= 0.5).astype(int))

    def plot_roc_curve(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
        logger.debug("Plotted ROC curve.")

    def plot_precision_recall(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_pred)
        plt.figure()
        plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()
        logger.debug("Plotted Precision-Recall curve.")

    def plot_confusion_matrix(self, y_true, y_pred_binary):
        cm = confusion_matrix(y_true, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
        logger.debug("Plotted Confusion Matrix.")

# Main Function
def main():
    directory = r'C:/Users/D.Muehlfeld/Documents/Berechnungsdaten_Valve/Zwischenspeicher/'

    # List of nodes with available measurement data
    included_nodes = [
        'K0003', 'K0004', 'K0005', 'K0006', 'K0011', 'K0012', 'K0014', 'K0016',
        'K0017', 'K0020', 'K0021', 'K0028', 'K0032', 'K0033', 'K0034', 'K0061',
        'K0062', 'K0063', 'K0064', 'K0080', 'K0086', 'K0087', 'K0088', 'K0090',
        'K0091', 'K0093', 'K0094', 'K0115', 'K0119', 'K0121', 'K0125', 'K0135',
        'K0136', 'K0138', 'K0144', 'K0145', 'K0147', 'K0148', 'K0151', 'K0155',
        'K0156', 'K0163', 'K0164', 'K0165', 'K0166', 'K0173', 'K0174', 'K0176',
        'K0179', 'K0180', 'K0183', 'K0184', 'K0185', 'K0186', 'K0191', 'K0195',
        'K0201', 'K0207', 'K0208', 'K0209', 'K0210', 'K0211', 'K0213', 'K0214',
        'K0220', 'K0229', 'K0230', 'K0238', 'K0239', 'K0245', 'K0254', 'K0260',
        'K0261', 'K0294', 'K0300', 'K0301', 'K0305', 'K0311', 'K0314', 'K0315',
        'K0316', 'K0317', 'K0318', 'K0319', 'K0327', 'K0328', 'K0329', 'K0330',
        'K0333', 'K0334', 'K0335', 'K0336', 'K0340', 'K0347', 'K0357', 'K0215',
        'K0388'
    ]

    # List of nodes with available ZUFLUSS_WL data (sorted and without duplicates)
    zfluss_wl_nodes = [
        'K0003', 'K0004', 'K0005', 'K0006', 'K0007', 'K0008', 'K0011', 'K0012', 'K0013', 'K0014',
        'K0015', 'K0016', 'K0021', 'K0022', 'K0023', 'K0025', 'K0026', 'K0028', 'K0029', 'K0030',
        'K0031', 'K0033', 'K0034', 'K0036', 'K0037', 'K0038', 'K0040', 'K0041', 'K0042', 'K0045',
        'K0046', 'K0059', 'K0060', 'K0061', 'K0062', 'K0063', 'K0065', 'K0066', 'K0067', 'K0070',
        'K0071', 'K0073', 'K0074', 'K0076', 'K0077', 'K0078', 'K0079', 'K0080', 'K0081', 'K0082',
        'K0083', 'K0084', 'K0085', 'K0086', 'K0089', 'K0090', 'K0091', 'K0093', 'K0094', 'K0095',
        'K0101', 'K0106', 'K0108', 'K0109', 'K0110', 'K0111', 'K0112', 'K0113', 'K0115', 'K0118',
        'K0119', 'K0121', 'K0122', 'K0125', 'K0127', 'K0128', 'K0129', 'K0130', 'K0131', 'K0132',
        'K0133', 'K0135', 'K0136', 'K0137', 'K0138', 'K0140', 'K0141', 'K0142', 'K0147', 'K0148',
        'K0151', 'K0152', 'K0155', 'K0156', 'K0160', 'K0161', 'K0162', 'K0163', 'K0164', 'K0165',
        'K0166', 'K0168', 'K0169', 'K0170', 'K0171', 'K0172', 'K0173', 'K0174', 'K0176', 'K0177',
        'K0179', 'K0180', 'K0181', 'K0182', 'K0183', 'K0184', 'K0185', 'K0186', 'K0188', 'K0189',
        'K0190', 'K0191', 'K0193', 'K0195', 'K0196', 'K0197', 'K0198', 'K0199', 'K0200', 'K0201',
        'K0202', 'K0204', 'K0205', 'K0206', 'K0207', 'K0208', 'K0209', 'K0210', 'K0211', 'K0213',
        'K0214', 'K0215', 'K0219', 'K0220', 'K0222', 'K0223', 'K0226', 'K0229', 'K0230', 'K0232',
        'K0233', 'K0234', 'K0235', 'K0237', 'K0238', 'K0239', 'K0244', 'K0245', 'K0248', 'K0249',
        'K0250', 'K0251', 'K0252', 'K0255', 'K0256', 'K0260', 'K0261', 'K0262', 'K0264', 'K0265',
        'K0266', 'K0277', 'K0280', 'K0281', 'K0282', 'K0283', 'K0287', 'K0289', 'K0290', 'K0291',
        'K0292', 'K0295', 'K0296', 'K0298', 'K0299', 'K0300', 'K0301', 'K0304', 'K0305', 'K0306',
        'K0307', 'K0308', 'K0309', 'K0311', 'K0312', 'K0313', 'K0314', 'K0315', 'K0316', 'K0317',
        'K0318', 'K0319', 'K0320', 'K0321', 'K0322', 'K0323', 'K0324', 'K0325', 'K0327', 'K0328',
        'K0329', 'K0330', 'K0332', 'K0333', 'K0334', 'K0336', 'K0340', 'K0342', 'K0346', 'K0347',
        'K0357', 'K0364', 'K0365', 'K0388'
    ]

    # Initialize DataModule
    data_module = DataModule(directory, included_nodes, zfluss_wl_nodes, num_valves=108)
    data_module.load_all_data()
    train_loader, val_loader, test_loader = data_module.get_loaders()

    if not train_loader or not val_loader or not test_loader:
        logger.error("Data loaders could not be created. Exiting the program.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = EdgeGAT(
        num_node_features=data_module.datasets[0].x.shape[1],
        num_edge_features=data_module.datasets[0].edge_attr.shape[1],
        hidden_dim=64,
        dropout=0.15
    ).to(device)
    logger.info("Initialized EdgeGAT model.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    logger.info("Initialized AdamW optimizer.")

    # Loss function with class weighting based on the training dataset
    positive_samples = sum([data.y.sum().item() for data in train_loader.dataset])
    total_samples = sum([len(data.y) for data in train_loader.dataset])
    negative_samples = total_samples - positive_samples

    if positive_samples == 0:
        pos_weight_value = 1.0
        logger.warning("No positive samples found in the training dataset. Setting pos_weight to 1.0.")
    else:
        pos_weight_value = negative_samples / positive_samples
        logger.info(f"Calculated pos_weight: {pos_weight_value:.4f}")

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("Initialized BCEWithLogitsLoss with pos_weight.")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    logger.info("Initialized ReduceLROnPlateau scheduler.")

    # Initialize Trainer
    trainer = Trainer(model, optimizer, criterion, scheduler, device, num_epochs=300, patience=10)
    logger.info("Initialized Trainer.")

    # Start training
    trainer.train_model(train_loader, val_loader)

    # Initialize Evaluator
    evaluator = Evaluator(model, device)
    logger.info("Initialized Evaluator.")

    # Test the model on the test dataset
    y_true, y_pred = evaluator.test_model(test_loader)

    # Calculate metrics
    accuracy, auc_score, y_pred_binary = evaluator.calculate_metrics(y_true, y_pred)

    # Generate plots
    evaluator.plot_metrics(y_true, y_pred)

    # Save the model
    model_path = os.path.join(directory, 'edge_gat_model_classification.pth')
    trainer.save_model(model_path)

    logger.info("Program completed successfully.")

if __name__ == "__main__":
    main()
