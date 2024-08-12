import os
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.nn import Linear, ReLU, Dropout
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
from create_dataset import GraphDataset, GraphDataLoader

# Setup logging
logging.basicConfig(
    filename='gnn_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite the log file each time
)

# Load the datasets using GraphDataset
folder_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\Zwischenspeicher'
graph_dataset = GraphDataset(folder_path, save_path='graph_data.pt')

# Ensure the dataset is not empty
if not graph_dataset.data_list:
    logging.error("No datasets loaded. Exiting.")
    print("No datasets loaded. Exiting.")
    exit(1)

# Split the data into training and testing datasets
train_size = int(0.8 * len(graph_dataset.data_list))
test_size = len(graph_dataset.data_list) - train_size
train_data_list, test_data_list = torch.utils.data.random_split(graph_dataset.data_list, [train_size, test_size])

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)


class ImprovedGNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(ImprovedGNN, self).__init__()
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

    def forward(self, data):
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
        return x


# Extract input dimensions from the dataset
input_dim = graph_dataset.data_list[0].num_features

# Instantiate the model, define the loss function and the optimizer
model = ImprovedGNN(input_dim=input_dim)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.MSELoss()


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        target = data.y.mean(dim=0).view(1, -1)  # Average the target values
        if target.size() != output.size():
            logging.error(f"Target size {target.size()} does not match output size {output.size()}")
            continue
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            output = model(data)
            target = data.y.mean(dim=0).view(1, -1)  # Average the target values
            if target.size() != output.size():
                logging.error(f"Target size {target.size()} does not match output size {output.size()}")
                continue
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(loader)


# Training loop
epochs = 50
train_losses = []
test_losses = []

for epoch in range(epochs):
    train_loss = train()
    test_loss = test(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")

# Predict on the test set
model.eval()
y_pred = []
y_test_actual = []

with torch.no_grad():
    for data in test_loader:
        output = model(data)
        y_pred.extend(output.cpu().numpy().flatten())
        y_test_actual.extend(data.y.mean(dim=0).cpu().numpy().flatten())

# Ensure y_pred and y_test_actual are consistent in length
if len(y_pred) != len(y_test_actual):
    logging.error(f"Mismatch in length of predictions and actual values: {len(y_pred)} vs {len(y_test_actual)}")
    print(f"Mismatch in length of predictions and actual values: {len(y_pred)} vs {len(y_test_actual)}")
else:
    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_test_actual = np.array(y_test_actual)

    # Calculate metrics
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred)

    logging.info(f"GNN Model - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    # Print results to console
    print("Model Evaluation Results:")
    print(f"GNN Model - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

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

    # Display predicted and actual values for a dataset
    print("\nPredicted vs Actual values:")
    for i, (pred, actual) in enumerate(zip(y_pred, y_test_actual)):
        print(f"Sample {i + 1}: Predicted = {pred:.4f}, Actual = {actual:.4f}")


# test
