import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import logging

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

# Feature Engineering: Adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_combined = combined_data[['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']]
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

# Define models to test
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

# Function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2

# Evaluate all models and store results
results = {}
for name, model in models.items():
    logging.info(f"Evaluating model: {name}")
    mse, rmse, r2 = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results[name] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}
    logging.info(f"{name} - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

# Find the best model
best_model_name = min(results, key=lambda x: results[x]['MSE'])
best_model = models[best_model_name]
logging.info(f"Best model: {best_model_name} with MSE: {results[best_model_name]['MSE']}")

# Plot the results of the best model
y_pred = best_model.predict(X_test_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual RAU')
plt.ylabel('Predicted RAU')
plt.title(f'Results for {best_model_name}')
plt.legend()
plt.show()

# Output the results
print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']}, RMSE={metrics['RMSE']}, R2={metrics['R2']}")

print(f"Best model: {best_model_name} with MSE: {results[best_model_name]['MSE']}")
