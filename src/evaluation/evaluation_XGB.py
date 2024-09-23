import logging
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

class XGBoostModelEvaluator:
    @staticmethod
    def evaluate_and_visualize(X_test, y_test, model_file="xgboost_model.json"):
        # Initialize logger
        logger = logging.getLogger(__name__)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_path = os.path.join(project_root, 'results', 'models', model_file)

        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.error(f"The specified model file '{model_path}' does not exist.")
            return

        # Load the saved XGBoost model from the file
        model = xgb.Booster()
        model.load_model(model_path)  # Load the model from the full path

        # Create a DMatrix for the test data (required by XGBoost)
        dtest = xgb.DMatrix(X_test)

        # Make predictions on the test data
        y_pred = model.predict(dtest)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # R-squared (coefficient of determination)

        # Plot the true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('XGBoost - True vs Predicted')

        # Add the metrics to the plot as text
        metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        # Remove legend if not needed
        # plt.legend()  # Remove this line if no labels are explicitly added to the plot

        # Generate the directory path for saving the plot
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Create a filename with the current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"XGB_{timestamp}.png"
        file_path = os.path.join(results_dir, file_name)

        # Save the plot
        plt.savefig(file_path)
        logger.info(f"Plot saved to {file_path}")

        # Show the plot
        plt.show()

        # Print the evaluation metrics to the console
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
