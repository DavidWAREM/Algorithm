import os
import numpy as np
import torch
import logging
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.nn import Linear, ReLU, Dropout
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from src.data.datapreperation_GNN import GraphDataset

class GNNModel:
    def __init__(self, folder_path, save_path, epochs=5, batch_size=1, learning_rate=0.001):
        self.folder_path = folder_path
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Load the datasets
        self.graph_dataset = GraphDataset(folder_path, save_path=save_path)
        logging.debug("Graph dataset loaded successfully.")

        # Print the first 10 rows of the first dataset with headers
        if len(self.graph_dataset.data_list) > 0:
            first_data = self.graph_dataset.data_list[0]
            print("First 10 rows of the first dataset with headers:")

            # Define the column names (headers) for the dataset
            feature_names = ['ZUFLUSS', 'PMESS', 'PRECH', 'DP', 'HP', 'XRECHTS', 'YHOCH', 'GEOH']  # Adjust as necessary
            print(" | ".join(feature_names))  # Print headers
            for row in first_data.x[:10]:
                print(" | ".join(f"{value:.4f}" for value in row))  # Print each row with formatted values
        else:
            logging.error("No datasets found to display.")

        # Ensure the dataset is not empty
        if not self.graph_dataset.data_list:
            logging.error("No datasets loaded. Exiting.")
            raise ValueError("No datasets loaded.")

        # Split the data into training and testing datasets
        train_size = int(0.8 * len(self.graph_dataset.data_list))
        test_size = len(self.graph_dataset.data_list) - train_size
        self.train_data_list, self.test_data_list = torch.utils.data.random_split(
            self.graph_dataset.data_list, [train_size, test_size])
        logging.debug(f"Data split into {train_size} training samples and {test_size} test samples.")

        # Create DataLoader instances for training and testing
        self.train_loader = DataLoader(self.train_data_list, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data_list, batch_size=batch_size, shuffle=False)
        logging.debug("DataLoader instances created for training and testing.")

        # Extract input dimensions from the dataset
        self.input_dim = self.graph_dataset.data_list[0].num_features
        logging.debug(f"Input dimensions extracted: {self.input_dim} features.")

        # Instantiate the model, define the loss function and the optimizer
        self.model = self.ImprovedGNN(input_dim=self.input_dim)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss()
        logging.debug("Model, optimizer, and loss function initialized.")

    class ImprovedGNN(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv1 = GCNConv(input_dim, 32)
            self.bn1 = BatchNorm(32)
            self.conv2 = GCNConv(32, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = GCNConv(64, 32)
            self.bn3 = BatchNorm(32)
            self.fc1 = Linear(32, 16)
            self.fc2 = Linear(16, 1)
            self.relu = ReLU()
            self.dropout = Dropout(0.5)
            logging.debug("ImprovedGNN model architecture initialized.")

        def forward(self, data):
            logging.debug("Forward pass started.")
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x, edge_index)
            x = self.bn3(x)
            x = self.relu(x)
            x = global_mean_pool(x, data.batch)
            x = self.dropout(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            logging.debug("Forward pass completed.")
            return x

    def train(self):
        self.model.train()
        total_loss = 0
        logging.debug("Training started.")
        for data in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            target = data.y.mean(dim=0).view(1, -1)  # Average the target values
            if target.size() != output.size():
                logging.error(f"Target size {target.size()} does not match output size {output.size()}")
                continue
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        logging.debug(f"Training completed with average loss: {avg_loss}.")
        return avg_loss

    def test(self):
        self.model.eval()
        total_loss = 0
        logging.debug("Testing started.")
        with torch.no_grad():
            for data in self.test_loader:
                output = self.model(data)
                target = data.y.mean(dim=0).view(1, -1)  # Average the target values
                if target.size() != output.size():
                    logging.error(f"Target size {target.size()} does not match output size {output.size()}")
                    continue
                loss = self.criterion(output, target)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.test_loader)
        logging.debug(f"Testing completed with average loss: {avg_loss}.")
        return avg_loss

    def run_training(self):
        train_losses = []
        test_losses = []

        logging.debug("Training loop started.")
        for epoch in range(self.epochs):
            train_loss = self.train()
            test_loss = self.test()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            logging.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")
            logging.debug("Training loop completed.")

    def evaluate(self):
        # Predict on the test set
        self.model.eval()
        y_pred = []
        y_test_actual = []

        logging.debug("Evaluation started.")
        with torch.no_grad():
            for data in self.test_loader:
                output = self.model(data)
                y_pred.extend(output.cpu().numpy().flatten())
                y_test_actual.extend(data.y.mean(dim=0).cpu().numpy().flatten())

        # Ensure y_pred and y_test_actual are consistent in length
        if len(y_pred) != len(y_test_actual):
            logging.error(f"Mismatch in length of predictions and actual values: {len(y_pred)} vs {len(y_test_actual)}")
            raise ValueError(f"Mismatch in length of predictions and actual values: {len(y_pred)} vs {len(y_test_actual)}")

        # Convert to numpy arrays
        y_pred = np.array(y_pred)
        y_test_actual = np.array(y_test_actual)

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred)

        logging.info(f"GNN Model - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
        logging.debug("Evaluation metrics calculated.")

        # Print results to console
        print("Model Evaluation Results:")
        print(f"GNN Model - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Print the first 20 predictions along with the actual values
        print("\nFirst 20 Predictions vs Actual Values:")
        for i in range(min(20, len(y_pred))):
            print(f"Prediction {i + 1}: Predicted = {y_pred[i]:.4f}, Actual = {y_test_actual[i]:.4f}")

        # Plot the results of the model
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_actual, y_pred, color='black', label='Actual vs Predicted')
        plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], color='blue',
                 linewidth=2, label='Ideal Fit')
        plt.xlabel('Actual RAU')
        plt.ylabel('Predicted RAU')
        plt.title('Results for GNN Model')
        plt.legend()
        plt.show()

        # Residual Analysis
        residuals = y_test_actual - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_actual, residuals, color='red')
        plt.axhline(y=0, color='blue', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        plt.show()

        # Evaluate model performance on large values
        large_values_mask = y_test_actual > np.quantile(y_test_actual, 0.75)
        mse_large_values = mean_squared_error(y_test_actual[large_values_mask], y_pred[large_values_mask])
        rmse_large_values = np.sqrt(mse_large_values)

        logging.info(f"Performance on large values - MSE: {mse_large_values}, RMSE: {rmse_large_values}")
        print(f"Performance on large values - MSE: {mse_large_values}, RMSE: {rmse_large_values}")

    def save_model(self, file_name="gnn_model.pth"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, file_name)
        torch.save(self.model.state_dict(), file_path)
        logging.info(f"Model saved to {file_path}")

if __name__ == "__main__":
    # Setup logging only when running as the main module
    logging.basicConfig(
        filename='gnn_model.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    folder_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher'
    save_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\GNN Dataset'
    gnn_model = GNNModel(folder_path=folder_path, save_path=save_path)
    gnn_model.run_training()
    gnn_model.evaluate()
    gnn_model.save_model()
