import logging
import os
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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

        # Log the evaluation metrics
        logger.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot the true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('XGBoost - True vs Predicted')
        plt.show()


