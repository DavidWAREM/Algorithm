import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
import logging

# Setup logging
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Define the parameter grid for GridSearchCV with reduced options
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_features': [1.0, 'sqrt'],
    'max_depth': [5, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'subsample': [0.8],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Initialize GridSearchCV for each model
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='neg_mean_squared_error')
grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=3, scoring='neg_mean_squared_error')

# Train the models using GridSearchCV
logging.info(f"Starting GridSearchCV for RandomForestRegressor")
grid_search_rf.fit(X_train_scaled, y_train)
logging.info(f"Completed GridSearchCV for RandomForestRegressor with best parameters: {grid_search_rf.best_params_}")

logging.info(f"Starting GridSearchCV for GradientBoostingRegressor")
grid_search_gb.fit(X_train_scaled, y_train)
logging.info(f"Completed GridSearchCV for GradientBoostingRegressor with best parameters: {grid_search_gb.best_params_}")

# Output the best parameters
best_params_rf = grid_search_rf.best_params_
best_params_gb = grid_search_gb.best_params_

print(f"Best parameters for RandomForestRegressor: {best_params_rf}")
print(f"Best parameters for GradientBoostingRegressor: {best_params_gb}")

# Best models from GridSearchCV
best_model_rf = grid_search_rf.best_estimator_
best_model_gb = grid_search_gb.best_estimator_

# Evaluate the best models on the test set
y_pred_rf = best_model_rf.predict(X_test_scaled)
y_pred_gb = best_model_gb.predict(X_test_scaled)

# Calculate metrics for RandomForestRegressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

logging.info(f"RandomForestRegressor - Mean Squared Error (MSE): {mse_rf}")
logging.info(f"RandomForestRegressor - Root Mean Squared Error (RMSE): {rmse_rf}")
logging.info(f"RandomForestRegressor - R^2 Score: {r2_rf}")

# Calculate metrics for GradientBoostingRegressor
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

logging.info(f"GradientBoostingRegressor - Mean Squared Error (MSE): {mse_gb}")
logging.info(f"GradientBoostingRegressor - Root Mean Squared Error (RMSE): {rmse_gb}")
logging.info(f"GradientBoostingRegressor - R^2 Score: {r2_gb}")

# Plot the collected results for RandomForestRegressor
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, color='black', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual DP')
plt.ylabel('Predicted DP')
plt.title(f'Results for RandomForestRegressor')
plt.legend()

# Plot feature importances for RandomForestRegressor
if hasattr(best_model_rf, 'feature_importances_'):
    plt.subplot(1, 2, 2)
    feature_importances = best_model_rf.feature_importances_
    features = poly.get_feature_names_out(['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE'])
    sorted_idx = np.argsort(feature_importances)
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importances for RandomForestRegressor')

plt.tight_layout()
plt.show()

# Plot the collected results for GradientBoostingRegressor
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_gb, color='black', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual DP')
plt.ylabel('Predicted DP')
plt.title(f'Results for GradientBoostingRegressor')
plt.legend()

# Plot feature importances for GradientBoostingRegressor
if hasattr(best_model_gb, 'feature_importances_'):
    plt.subplot(1, 2, 2)
    feature_importances = best_model_gb.feature_importances_
    features = poly.get_feature_names_out(['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE'])
    sorted_idx = np.argsort(feature_importances)
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importances for GradientBoostingRegressor')

plt.tight_layout()
plt.show()

logging.info("Model evaluation and plotting complete.")
