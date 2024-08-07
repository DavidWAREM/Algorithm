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
        self.model = self.load_model()

    def load_model(self):
        """
        Load a trained model from the 'results/models' directory using GradientBoostingModel class.

        Returns:
        model: The loaded model.
        """
        gb_model = GradientBoostingModel()
        model = gb_model.load_model()
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
        return y_pred


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Gradient Boosting Prediction with the trained model
    gb_predict = GradientBoostingPrediction()

    # Make predictions using the loaded model
    y_pred = gb_predict.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    # Print evaluation results
    print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    # Print predicted values
    print("Predicted values:")
    print(y_pred)
