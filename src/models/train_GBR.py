import os
import logging
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


class GradientBoostingModel:
    def __init__(self, random_state=42):
        """
        Initialize the Gradient Boosting Model with predefined hyperparameters.

        Parameters:
        random_state (int): Random state for reproducibility.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.params = {
            'learning_rate': 0.2,
            'max_depth': 5,
            'min_samples_leaf': 4,
            'min_samples_split': 2,
            'n_estimators': 300,
            'subsample': 1.0
        }
        self.random_state = random_state
        self.model = GradientBoostingRegressor(**self.params, random_state=self.random_state)

    def train(self, X_train, y_train):
        """
        Train the Gradient Boosting Model.

        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        """
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed.")

    def save_model(self, file_name="gradient_boosting_model.joblib"):
        """
        Save the trained model to a file in the 'results/models' directory.

        Parameters:
        file_name (str): Name of the file where the model will be saved.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, file_name)
        joblib.dump(self.model, file_path)
        self.logger.info(f"Model saved to {file_path}")

    def load_model(self, file_name="gradient_boosting_model.joblib"):
        """
        Load a trained model from a file in the 'results/models' directory.

        Parameters:
        file_name (str): Name of the file where the model is saved.

        Returns:
        model: The loaded model.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        file_path = os.path.join(project_root, 'results', 'models', file_name)
        self.model = joblib.load(file_path)
        self.logger.info(f"Model loaded from {file_path}")
        return self.model
