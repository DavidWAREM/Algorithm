import logging
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.metrics_impl import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import yaml  # Import YAML to read configuration files

class XGBoostModelEvaluator:
    @staticmethod
    def evaluate_and_visualize(X_test, y_test, model_file="xgboost_model.json"):
        """
        Evaluate and visualize the performance of an XGBoost model.

        Args:
            X_test (np.array): Features of the test dataset.
            y_test (np.array): True target values for the test dataset.
            model_file (str, optional): Filename of the saved XGBoost model. Defaults to "xgboost_model.json".

        This method:
        1. Loads the configuration file to extract 'folder_path_rawdata'.
        2. Extracts the folder name from 'folder_path_rawdata'.
        3. Loads the saved XGBoost model.
        4. Makes predictions on the test data.
        5. Calculates evaluation metrics (MSE, RMSE, and R² score).
        6. Visualizes the true vs predicted values with a scatter plot.
        7. Saves the plot in the 'results/results' directory with the metrics and folder name included.
        """
        # Initialize logger
        logger = logging.getLogger(__name__)

        # Define the path to the configuration file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Adjust as per your project structure
        config_file = os.path.join(project_root, 'config', 'config.yaml')  # Adjust if your config file is located elsewhere

        # Check if the config file exists
        if not os.path.exists(config_file):
            logger.error(f"Configuration file not found at: {config_file}")
            raise FileNotFoundError(f"Configuration file not found at: {config_file}")

        # Load configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Extract 'folder_path_rawdata' from the configuration
        folder_path_rawdata = config['paths']['folder_path_rawdata']

        # Extract the folder name from the path
        data_label = os.path.basename(os.path.normpath(folder_path_rawdata))

        # Load the saved XGBoost model from the file
        model_path = os.path.join(project_root, 'results', 'models', model_file)

        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.error(f"The specified model file '{model_path}' does not exist.")
            raise FileNotFoundError(f"The specified model file '{model_path}' does not exist.")

        # Load the XGBoost model
        model = xgb.Booster()
        model.load_model(model_path)  # Load the model from the full path
        logger.info(f"Loaded XGBoost model from {model_path}")

        # Create a DMatrix for the test data (required by XGBoost)
        dtest = xgb.DMatrix(X_test)

        # Make predictions on the test data
        y_pred = model.predict(dtest)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)  # R-squared (coefficient of determination)

        # Log the evaluation results
        logger.info(f"Model Evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Plot the true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

        # Include the data label in the title
        plt.title(f'XGBoost - True vs Predicted RAU Values\n{data_label}')
        plt.grid(False)

        # Add the metrics to the plot as text
        metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        # Generate the directory path for saving the plot
        results_dir = os.path.join(project_root, 'results', 'results')
        os.makedirs(results_dir, exist_ok=True)
        logger.debug(f"Ensured that results directory exists at: {results_dir}")

        # Create a filename with the current date and time, including the data label
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        file_name = f"XGB_{data_label}_{timestamp}.png"
        file_path = os.path.join(results_dir, file_name)

        # Save the plot
        plt.savefig(file_path)
        logger.info(f"Plot saved to {file_path}")

        # Show the plot
        plt.show()

        # Print the evaluation metrics to the console
        print(f"Model Evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
