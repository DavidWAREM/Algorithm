import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.prediction.predictions_ANN import ANNPrediction

class ANNModelEvaluator:
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
        """
        # Initialize the ANNPrediction class to make predictions using the trained ANN model
        ann_predict = ANNPrediction()

        # Make predictions using the test data
        y_pred = ann_predict.predict(X_test).flatten()  # Flatten to match the shape of the target values

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # R² score (coefficient of determination)

        # Log the evaluation results
        logging.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot True vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)  # Scatter plot of true vs predicted values
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)  # Line for perfect prediction
        plt.xlabel('True Values')  # Label for X-axis
        plt.ylabel('Predicted Values')  # Label for Y-axis
        plt.title('ANN - True vs Predicted')  # Title of the plot
        plt.show()  # Display the plot

        # Print the evaluation metrics to the console
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
