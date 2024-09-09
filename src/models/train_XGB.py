import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class XGBoostModel:
    """
    A class to train, tune, and evaluate an XGBoost regression model using GPU acceleration.
    """

    def __init__(self):
        """
        Initialize the XGBoost model with predefined parameters.
        """
        # Parameters for the XGBoost model using GPU for training.
        self.params = {
            'objective': 'reg:squarederror',
            'device': 'cuda',  # Use GPU for training
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 7,
            'n_estimators': 600,
            'subsample': 0.9
        }
        self.model = xgb.XGBRegressor(**self.params)

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Parameters:
        X_train (numpy.array or pandas.DataFrame): Training features.
        y_train (numpy.array or pandas.Series): Training target values.
        """
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'max_depth': [5, 6, 7],  # Finer gradation in tree depth
            'learning_rate': [0.05, 0.1, 0.15],  # Smaller learning rates for more stable updates
            'n_estimators': [400, 500, 600],  # More trees for better accuracy
            'subsample': [0.7, 0.8, 0.9],  # Slightly reduced subsampling to avoid overfitting
            'colsample_bytree': [0.7, 0.8, 0.9]  # More variance in selected features per tree
        }

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3,
                                   verbose=1)
        grid_search.fit(X_train, y_train)

        # Save the best model and hyperparameters
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

    def train(self, X_train, y_train):
        """
        Train the XGBoost model.

        Parameters:
        X_train (numpy.array or pandas.DataFrame): Training features.
        y_train (numpy.array or pandas.Series): Training target values.
        """
        self.model.fit(X_train, y_train)

    def save_model(self, file_name="xgboost_model.model"):
        """
        Save the trained XGBoost model to a file.

        Parameters:
        file_name (str): The filename to save the model. Defaults to "xgboost_model.model".
        """
        self.model.save_model(file_name)

    def load_model(self, file_name="xgboost_model.model"):
        """
        Load a trained XGBoost model from a file.

        Parameters:
        file_name (str): The filename from which to load the model. Defaults to "xgboost_model.model".
        """
        self.model = xgb.Booster()
        self.model.load_model(file_name)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the XGBoost model on test data.

        Parameters:
        X_test (numpy.array or pandas.DataFrame): Test features.
        y_test (numpy.array or pandas.Series): Test target values.
        """
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")
