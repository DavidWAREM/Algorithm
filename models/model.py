import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Setup logging
logging.basicConfig(filename='model_comparison.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path, sep=';')
    logging.info(f"Loaded data from {file_path} with shape {data.shape}")
    return data

# Path to the folder with the CSV files
folder_path = r"C:\Users\d.muehlfeld\Berechnungsdaten\Trainingsdaten"

# List to store data from all CSV files
all_data = []

# Iterate through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        data = load_data_from_csv(file_path)

        # Log the columns of the file for debugging purposes
        logging.debug(f"Columns in {file_name}: {data.columns.tolist()}")

        # Check if required columns are present
        required_columns = ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']
        if not all(column in data.columns for column in required_columns):
            logging.warning(f"Required columns not found in file: {file_name}")
            continue

        all_data.append((file_name, data))

# Raise an error if no valid CSV files are found
if not all_data:
    logging.error("No valid CSV files found with the required columns.")
    raise ValueError("No valid CSV files found with the required columns.")

# Combine all data for feature engineering
combined_data = pd.concat([data for _, data in all_data])
logging.info(f"Combined data shape: {combined_data.shape}")

# Add a small constant to avoid log(0) issues
small_constant = 1e-6
features_to_transform = ['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']
combined_data[features_to_transform] = combined_data[features_to_transform] + small_constant

# Log-transform the relevant features and the target variable
combined_data[features_to_transform] = combined_data[features_to_transform].apply(lambda x: np.log1p(x))
combined_data['RAU'] = np.log1p(combined_data['RAU'])

# Check for and handle any NaN values resulting from the transformation
if combined_data.isnull().values.any():
    combined_data = combined_data.dropna()

# Feature Engineering: Adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_combined = combined_data[features_to_transform]
X_combined_poly = poly.fit_transform(X_combined)
y_combined = combined_data['RAU']
logging.info(f"Feature engineering complete with polynomial features of degree 2")

# Use a smaller sample of the data for training
X_combined_poly_sample, _, y_combined_sample, _ = train_test_split(X_combined_poly, y_combined, test_size=0.9, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_poly_sample, y_combined_sample, test_size=0.2, random_state=42)
logging.info(f"Data split into training and testing sets with test size 20%")

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info(f"Data scaling complete using StandardScaler")

# Handle outliers using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(X_train_scaled)
mask = outliers != -1
X_train_scaled, y_train = X_train_scaled[mask], y_train[mask]

# Build a simpler model architecture
def build_simple_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Build the model
model = build_simple_model(X_train_scaled.shape[1])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

# Predict on the test set
y_pred = model.predict(X_test_scaled).flatten()  # Ensure y_pred is 1-dimensional

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Evaluate model performance on large values
large_values_mask = y_test > y_test.quantile(0.75)
mse_large_values = mean_squared_error(y_test[large_values_mask], y_pred[large_values_mask])
rmse_large_values = np.sqrt(mse_large_values)
mae_large_values = mean_absolute_error(y_test[large_values_mask], y_pred[large_values_mask])

# Identify the data points with the largest residuals
residuals = y_test - y_pred
sorted_residuals = np.argsort(np.abs(residuals))[::-1]  # Indices of residuals sorted by magnitude
top_residual_indices = sorted_residuals[:2]  # Get the top 2 residuals

# Output the results in a detailed format
results = {
    "Overall Performance": {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    },
    "Performance on Large Values": {
        "MSE": mse_large_values,
        "RMSE": rmse_large_values,
        "MAE": mae_large_values
    }
}

# Print results
print("Model Evaluation Results:")
print(f"Overall Performance - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")
print(f"Performance on Large Values - MSE: {mse_large_values}, RMSE: {rmse_large_values}, MAE: {mae_large_values}")

# Print details of the data points with the largest residuals
print("\nData points with the largest residuals:")
for idx in top_residual_indices:
    print(f"Index: {idx}, Actual: {y_test.iloc[idx]}, Predicted: {y_pred[idx]}, Residual: {residuals.iloc[idx]}")

# Plot the results of the model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
plt.scatter(y_test.iloc[top_residual_indices], y_pred[top_residual_indices], color='red', label='Largest Residuals')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual RAU')
plt.ylabel('Predicted RAU')
plt.title('Results for ANN Model')
plt.legend()
plt.show()

# Residual Analysis
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='red')
plt.scatter(y_test.iloc[top_residual_indices], residuals.iloc[top_residual_indices], color='blue', label='Largest Residuals')
plt.axhline(y=0, color='blue', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.legend()
plt.show()

# Save results to a file for further analysis
with open("model_results.txt", "w") as file:
    file.write("Model Evaluation Results:\n")
    file.write(f"Overall Performance - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}\n")
    file.write(f"Performance on Large Values - MSE: {mse_large_values}, RMSE: {rmse_large_values}, MAE: {mae_large_values}\n")
    file.write("\nData points with the largest residuals:\n")
    for idx in top_residual_indices:
        file.write(f"Index: {idx}, Actual: {y_test.iloc[idx]}, Predicted: {y_pred[idx]}, Residual: {residuals.iloc[idx]}\n")

# Print detailed results for further improvement
import pprint
pprint.pprint(results)

# Additional plots for better evaluation
# Histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, color='grey', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# QQ-plot of residuals
import scipy.stats as stats
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ-Plot of Residuals')
plt.show()

# Scatter plot of residuals vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', edgecolor='black')
plt.scatter(y_pred[top_residual_indices], residuals.iloc[top_residual_indices], color='red', label='Largest Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.legend()
plt.show()
