import logging
import xgboost as xgb
import numpy as np


class XGBoostPrediction:
    """
    A class to load a trained XGBoost model and make predictions.
    """

    def __init__(self, model_file="xgboost_model.model"):
        """
        Initialize the XGBoostPrediction by loading the trained model.

        Parameters:
        model_file (str): Name of the file where the model is saved. Defaults to "xgboost_model.model".
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Load the saved XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_file)
        self.logger.info(f"Model loaded from {model_file}")

    def predict(self, X):
        """
        Make predictions using the loaded model.

        Parameters:
        X (array-like): Features for prediction.

        Returns:
        array-like: Predicted values.
        """
        # Create DMatrix for predictions
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        self.logger.debug("Prediction completed.")
        return predictions
