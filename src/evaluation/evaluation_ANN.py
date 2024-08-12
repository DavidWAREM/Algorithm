import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.prediction.predictions_ANN import ANNPrediction

class ANNModelEvaluator:
    @staticmethod
    def evaluate_and_visualize(X_test, y_test):
        """
        Evaluate and visualize the ANN Model.

        Parameters:
        X_test: Testing features.
        y_test: Testing target.
        """
        # Initialize the ANNPrediction with the trained model
        ann_predict = ANNPrediction()

        # Make predictions
        y_pred = ann_predict.predict(X_test).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot True vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('ANN - True vs Predicted')
        plt.show()

        # Print evaluation results
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
