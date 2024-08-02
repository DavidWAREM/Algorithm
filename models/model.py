import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
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
    if file_name.endswith('_Pipes.csv'):  # Only process files ending with '_Pipes.csv'
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

# Split the data into training and testing sets without sampling
X_train, X_test, y_train, y_test = train_test_split(X_combined_poly, y_combined, test_size=0.2, random_state=42)
logging.info(f"Data split into training and testing sets with test size 20%")

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info(f"Data scaling complete using StandardScaler")

# Define the best Gradient Boosting model with the found hyperparameters
best_params = {
    'learning_rate': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 300,
    'subsample': 1.0
}

# Use the best parameters to train the final model
best_gb = GradientBoostingRegressor(**best_params, random_state=42)
best_gb.fit(X_train_scaled, y_train)
y_pred = best_gb.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Perform cross-validation on the best Gradient Boosting model
cv_scores = cross_val_score(best_gb, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
cv_rmse = np.sqrt(cv_mse)

logging.info(f"Best Gradient Boosting Model - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
logging.info(f"Cross-Validation - MSE: {cv_mse}, RMSE: {cv_rmse}")

# Plot the results of the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual RAU')
plt.ylabel('Predicted RAU')
plt.title('Results for Best Gradient Boosting Model')
plt.legend()
plt.show()

# Output the results
print("Model Evaluation Results:")
print(f"Best Gradient Boosting Model - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
print(f"Cross-Validation - MSE: {cv_mse}, RMSE: {cv_rmse}")

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='red')
plt.axhline(y=0, color='blue', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()

# Evaluate model performance on large values
large_values_mask = y_test > y_test.quantile(0.75)
mse_large_values = mean_squared_error(y_test[large_values_mask], y_pred[large_values_mask])
rmse_large_values = np.sqrt(mse_large_values)
logging.info(f"Performance on large values - MSE: {mse_large_values}, RMSE: {rmse_large_values}")

print(f"Performance on large values - MSE: {mse_large_values}, RMSE: {rmse_large_values}")
