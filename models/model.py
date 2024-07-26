import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(filename='model_training.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
X_combined = combined_data[['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE']]
X_combined_poly = poly.fit_transform(X_combined)
y_combined = combined_data['DP']
logging.info(f"Feature engineering complete with polynomial features of degree 2")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_poly, y_combined, test_size=0.2, random_state=42)
logging.info(f"Data split into training and testing sets with test size 20%")

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info(f"Data scaling complete using StandardScaler")

# List of models to evaluate
models = {
    'SGDRegressor': SGDRegressor(max_iter=1000000, random_state=42),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    logging.info(f"Training {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    logging.info(f"{name} R^2 Score: {r2}")

# Identify the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
logging.info(f"Best model: {best_model_name} with R^2 Score: {results[best_model_name]}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)

# Check if the best model has intercept_ and coef_ attributes and log them if they exist
if hasattr(best_model, 'intercept_'):
    logging.info(f"Intercept: {best_model.intercept_}")

if hasattr(best_model, 'coef_'):
    unscaled_intercept = best_model.intercept_ - np.sum(best_model.coef_ * scaler.mean_ / scaler.scale_)
    unscaled_coefficients = best_model.coef_ / scaler.scale_
    logging.info(f"Scaled Intercept: {best_model.intercept_}")
    logging.info(f"Scaled Coefficients: {best_model.coef_}")
    logging.info(f"Unscaled Intercept: {unscaled_intercept}")
    logging.info(f"Unscaled Coefficients: {unscaled_coefficients}")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

logging.info(f"Mean Squared Error (MSE): {mse}")
logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
logging.info(f"R^2 Score: {r2}")

# Plot the collected results
plt.figure()
plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual DP')
plt.ylabel('Predicted DP')
plt.title(f'Combined Results for {best_model_name}')
plt.legend()
plt.show()

logging.info("Model evaluation and plotting complete.")
