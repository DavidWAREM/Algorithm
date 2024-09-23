import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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

        # Plot True vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)  # Scatter plot of true vs predicted values
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)  # Line of perfect prediction
        plt.xlabel('True Values')  # X-axis label
        plt.ylabel('Predicted Values')  # Y-axis label
        plt.title('Gradient Boosting - True vs Predicted')  # Plot title
        plt.show()  # Display the plot

        # Print the evaluation metrics to the console
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
