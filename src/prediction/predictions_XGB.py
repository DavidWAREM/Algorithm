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
        # Load the saved XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_file)

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
        return self.model.predict(dmatrix)


if __name__ == "__main__":
    # Test data for predictions
    X_test = np.random.rand(100, 20)

    # Make predictions with XGBoost
    predictor = XGBoostPrediction()
    predictions = predictor.predict(X_test)

    print("Predictions:", predictions)
