import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import yaml  # Added to import the YAML library
from src.prediction.predictions_ANN import ANNPrediction

class ANNModelEvaluator:
    # Initialize a logger for this class
    logger = logging.getLogger(__name__)

    @staticmethod
    def evaluate_and_visualize(X_test, y_test):
        """
        Evaluate and visualize the performance of an Artificial Neural Network (ANN) model.

        Args:
            X_test (np.array): Features of the test dataset.
            y_test (np.array): True target values for the test dataset.

        This method:
        1. Initializes an instance of the `ANNPrediction` class to load the trained ANN model.
        2. Makes predictions on the test data.
        3. Calculates evaluation metrics (MSE, RMSE, and R² score) for the predicted vs actual target values.
        4. Visualizes the true vs predicted values with a scatter plot.
        5. Saves the plot in the 'results/results' directory with the metrics included in the plot.
        6. Includes the data label (extracted from the folder path) in the plot and filename.
        """
        # Load configuration from YAML file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        config_file = os.path.join(project_root, 'config', 'config.yaml')

        # Check if the config file exists
        if not os.path.exists(config_file):
            ANNModelEvaluator.logger.error(f"Configuration file not found at: {config_file}")
            raise FileNotFoundError(f"Configuration file not found at: {config_file}")

        # Load configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Extract 'folder_path_rawdata' from the configuration
        folder_path_rawdata = config['paths']['folder_path_rawdata']

        # Extract the folder name from the path
        data_label = os.path.basename(os.path.normpath(folder_path_rawdata))

        # Initialize the ANNPrediction class to make predictions using the trained ANN model
        ann_predict = ANNPrediction()

        # Make predictions using the test data
        y_pred = ann_predict.predict(X_test).flatten()  # Flatten to match the shape of the target values

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)  # R² score (coefficient of determination)

        # Log the evaluation results
        ANNModelEvaluator.logger.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot the true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

        # Include the data label in the title
        plt.title(f'ANN - True vs Predicted RAU Values\n{data_label}')

        # Add the metrics to the plot as text
        metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        # Generate the directory path for saving the plot
        results_dir = os.path.join(project_root, 'results', 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Create a filename with the current date and time, including the data label
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        file_name = f"ANN_{data_label}_{timestamp}.png"
        file_path = os.path.join(results_dir, file_name)

        # Save the plot
        plt.savefig(file_path)
        ANNModelEvaluator.logger.info(f"Plot saved to {file_path}")

        # Show the plot
        plt.show()

        # Print the evaluation metrics to the console
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
