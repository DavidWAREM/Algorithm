import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.models.train_GBR import GradientBoostingModel


class GradientBoostingPrediction:
    def __init__(self):
        """
        Initialize the Gradient Boosting Prediction.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.model = self.load_model()

    def load_model(self):
        """
        Load a trained model from the 'results/models' directory using GradientBoostingModel class.

        Returns:
        model: The loaded model.
        """
        gb_model = GradientBoostingModel()
        model = gb_model.load_model()
        self.logger.info("Model loaded successfully.")
        return model

    def predict(self, X):
        """
        Make predictions using the loaded model.

        Parameters:
        X (array-like): Features for prediction.

        Returns:
        y_pred (array-like): Predicted values.
        """
        y_pred = self.model.predict(X)
        self.logger.debug("Prediction completed.")
        return y_pred


