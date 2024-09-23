import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from src.prediction.predictions_GBR import GradientBoostingPrediction


class GBRModelEvaluator:
    # Initialize a logger for this class
    logger = logging.getLogger(__name__)

    @staticmethod
    def evaluate_and_visualize(X_test, y_test):
        """
        Evaluate and visualize the performance of a Gradient Boosting Regressor model.

        Args:
            X_test (np.array): Features from the test dataset.
            y_test (np.array): True target values from the test dataset.

        This method:
        1. Initializes the `GradientBoostingPrediction` class to use a trained Gradient Boosting model.
        2. Makes predictions on the test dataset.
        3. Computes evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
        4. Visualizes the true vs predicted values with a scatter plot.
        """
        # Initialize the GradientBoostingPrediction class to load the trained Gradient Boosting model
        gb_predict = GradientBoostingPrediction()

        # Make predictions on the test data
        y_pred = gb_predict.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # R² score (coefficient of determination)

        # Log the evaluation metrics for tracking the model's performance
        GBRModelEvaluator.logger.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        ## Plot the true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('GBR - True vs Predicted')

        # Add the metrics to the plot as text
        metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        # Generate the directory path for saving the plot
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Create a filename with the current date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"GBR_{timestamp}.png"
        file_path = os.path.join(results_dir, file_name)

        # Save the plot
        plt.savefig(file_path)
        GBRModelEvaluator.logger.info(f"Plot saved to {file_path}")

        # Show the plot
        plt.show()

        # Print the evaluation metrics to the console
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")