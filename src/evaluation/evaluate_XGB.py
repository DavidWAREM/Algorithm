import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

class XGBoostModelEvaluator:
    @staticmethod
    def evaluate_and_visualize(X_test, y_test, model_file="xgboost_model.model"):
        """
        Loads a saved XGBoost model, makes predictions, and evaluates the model performance using test data.

        Args:
            X_test (np.array): Features of the test data.
            y_test (np.array): True values (target) for the test data.
            model_file (str): Path to the saved XGBoost model file (default: "xgboost_model.model").

        This method performs the following tasks:
        1. Loads the saved XGBoost model.
        2. Makes predictions on the test data.
        3. Evaluates the model performance using metrics such as MSE, RMSE, and R².
        4. Visualizes the true vs predicted values with a scatter plot.
        """
        # Load the saved XGBoost model from the file
        model = xgb.Booster()
        model.load_model(model_file)  # Load the model from the specified file

        # Create a DMatrix for the test data (required by XGBoost)
        dtest = xgb.DMatrix(X_test)

        # Make predictions on the test data
        y_pred = model.predict(dtest)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # R-squared (coefficient of determination)

        # Print the evaluation metrics
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot the true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)  # Scatter plot for true vs predicted values
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)  # Line for perfect prediction
        plt.xlabel('True Values')  # X-axis label
        plt.ylabel('Predicted Values')  # Y-axis label
        plt.title('XGBoost - True vs Predicted')  # Plot title
        plt.show()  # Display the plot

if __name__ == "__main__":
    """
    Example usage of the XGBoostModelEvaluator.

    This script generates dummy regression data, trains an XGBoost model (assumed to be pre-saved), 
    and evaluates the model using the test data.
    """
    # Generate dummy regression data using sklearn's make_regression
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X
