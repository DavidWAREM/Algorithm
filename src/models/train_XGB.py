import logging
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


class XGBoostModel:
    """
    A class to train, tune, and evaluate an XGBoost regression model using GPU acceleration.
    """

    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)

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
        self.logger.debug(f"XGBoost parameters: {self.params}")
        self.model = xgb.XGBRegressor(**self.params)

    def hyperparameter_tuning(self, X_train, y_train):
        self.logger.info("Starting hyperparameter tuning.")

        param_grid = {
            'max_depth': [5, 6, 7],  # Finer gradation in tree depth
            'learning_rate': [0.05, 0.1, 0.15],  # Smaller learning rates for more stable updates
            'n_estimators': [400, 500, 600],  # More trees for better accuracy
            'subsample': [0.7, 0.8, 0.9],  # Slightly reduced subsampling to avoid overfitting
            'colsample_bytree': [0.7, 0.8, 0.9]  # More variance in selected features per tree
        }

        self.logger.debug(f"Parameter grid for hyperparameter tuning: {param_grid}")

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3,
                                   verbose=1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.logger.info(f"Hyperparameter tuning completed. Best parameters: {grid_search.best_params_}")

    def train(self, X_train, y_train):
        self.logger.info("Starting training.")
        try:
            self.logger.debug(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
            self.model.fit(X_train, y_train)
            self.logger.info("Training completed.")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}", exc_info=True)

    def save_model(self, file_name="xgboost_model.json"):
        self.logger.info(f"Saving model to file: {file_name}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir = os.path.join(project_root, 'results', 'models')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, file_name)
        self.model.save_model(file_path)
        self.logger.info(f"Model saved to {file_path}")

    def load_model(self, file_name="xgboost_model.json"):
        self.logger.info(f"Loading model from file: {file_name}")
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            file_path = os.path.join(project_root, 'results', 'models', file_name)
            self.model = xgb.XGBRegressor()
            self.model.load_model(file_path)
            self.logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {file_path}", exc_info=True)

    def evaluate(self, X_test, y_test):
        self.logger.info("Starting evaluation.")

        y_pred = self.model.predict(X_test)
        self.logger.debug(f"Predictions: {y_pred}")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        self.logger.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
