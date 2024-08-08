import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.models.train_ANN import ANNModel
from tensorflow.keras.models import load_model

class ANNPrediction:
    def __init__(self):
        """
        Initialize the ANN Prediction.
        """
        self.model = self.load_model()

    def load_model(self):
        """
        Load a trained ANN model from a specified path.

        Parameters:
        model_path (str): Path to the saved ANN model.

        Returns:
        model: The loaded model.
        """
        ann_model = ANNModel()
        model = ann_model.load_model()
        logging.info(f"Model loaded from {model_path}")
        return model

    def predict(self, X):
        """
        Make predictions using the loaded model.

        Parameters:
        X (array-like): Features for prediction.

        Returns:
        y_pred (array-like): Predicted values.
        """
        y_pred = self.model.predict(X).flatten()
        return y_pred

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a sample regression dataset
    X, y = np.random.rand(1000, 10), np.random.rand(1000)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the ANN Prediction with the trained model
    ann_predict = ANNPrediction()

    # Make predictions using the loaded model
    y_pred = ann_predict.predict(X_test_scaled)

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
