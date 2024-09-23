import logging
import os
from tensorflow.keras.models import load_model


class ANNPrediction:
    def __init__(self, file_name="ann_model.h5"):
        """
        Initialize the ANN Prediction by loading the trained model.

        Parameters:
        file_name (str): Name of the file where the model is saved.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.model = self.load_model(file_name)

    def load_model(self, file_name):
        """
        Load a trained ANN model from a file in the 'results/models' directory.

        Parameters:
        file_name (str): Name of the file where the model is saved.

        Returns:
        model: The loaded Keras model.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        file_path = os.path.join(results_dir, file_name)
        model = load_model(file_path)
        self.logger.info(f"Model loaded from {file_path}")
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
