import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging


class XGBoostModel:
    """
    A class to train, tune, and evaluate an XGBoost regression model using GPU acceleration.
    """

    def __init__(self):
        logging.info("Initializing XGBoost model with predefined parameters.")
        # Parameters for the XGBoost model using GPU for training.
        self.params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Use hist for tree_method
            'device': 'cuda',  # Use GPU for training with CUDA
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 7,
            'n_estimators': 600,
            'subsample': 0.9
        }
        logging.debug(f"XGBoost parameters: {self.params}")
        self.model = xgb.XGBRegressor(**self.params)

    def hyperparameter_tuning(self, X_train, y_train):
        logging.info("Starting hyperparameter tuning.")

        param_grid = {
            'max_depth': [5, 6, 7],  # Finer gradation in tree depth
            'learning_rate': [0.05, 0.1, 0.15],  # Smaller learning rates for more stable updates
            'n_estimators': [400, 500, 600],  # More trees for better accuracy
            'subsample': [0.7, 0.8, 0.9],  # Slightly reduced subsampling to avoid overfitting
            'colsample_bytree': [0.7, 0.8, 0.9]  # More variance in selected features per tree
        }

        logging.debug(f"Parameter grid for hyperparameter tuning: {param_grid}")

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3,
                                   verbose=1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logging.info(f"Hyperparameter tuning completed. Best parameters: {grid_search.best_params_}")

    def train(self, X_train, y_train):
        logging.info("Starting training.")
        self.model.fit(X_train, y_train)
        logging.info("Training completed.")

    def save_model(self, file_name="xgboost_model.json"):
        logging.info(f"Saving model to file: {file_name}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, file_name)
        self.model.save_model(file_path)
        logging.info(f"Model saved to {file_path}")

    def load_model(self, file_name="xgboost_model.json"):
        logging.info(f"Loading model from file: {file_name}")
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            file_path = os.path.join(project_root, 'results', 'models', file_name)
            self.model = xgb.XGBRegressor()
            self.model.load_model(file_path)
            logging.info(f"Model loaded from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {file_path}", exc_info=True)

    def evaluate(self, X_test, y_test):
        logging.info("Starting evaluation.")

        y_pred = self.model.predict(X_test)
        logging.debug(f"Predictions: {y_pred}")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("model_training.log"), logging.StreamHandler()])

    model = XGBoostModel()
    # Ensure you have training data here before running these methods
    # model.hyperparameter_tuning(X_train, y_train)
    # model.train(X_train, y_train)
    model.save_model("xgboost_model.json")
    model.load_model("xgboost_model.json")
    # model.evaluate(X_test, y_test)
