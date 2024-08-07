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
        logging.info(f"Model saved to {file_path}")

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
        logging.info(f"Model loaded from {file_path}")
        return self.model


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Gradient Boosting model
    gb_model = GradientBoostingModel(random_state=42)
    gb_model.train(X_train, y_train)

    # Save the trained model
    gb_model.save_model()

    # Load the trained model
    gb_model.load_model()

    # Make predictions and evaluate the model
    y_pred = gb_model.model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    # Print evaluation results
    print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
