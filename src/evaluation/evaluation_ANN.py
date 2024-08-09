import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.models.train_ANN import ANNModel  # Importing ANNModel from train_ANN
from src.prediction.predictions_ANN import ANNPrediction

class ANNModelEvaluator:
    @staticmethod
    def evaluate_and_visualize(X_test, y_test):
        """
        Evaluate and visualize the ANN Model.

        Parameters:
        model: Trained ANN model.
        X_test: Testing features.
        y_test: Testing target.
        """
        # Initialize the ANN with the trained model
        ann_predict = ANNPrediction

        # Make predictions
        y_pred = ann_predict.predict(X_test).flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        # Plot True vs Predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('ANN - True vs Predicted')
        plt.show()

        # Print evaluation results
        print(f"Model Evaluation - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the ANN model (assuming train_ANN.ANNModel is correctly implemented)
    ann_model = ANNModel(input_shape=X_train_scaled.shape[1])
    ann_model.train(X_train_scaled, y_train)

    # Evaluate and visualize the model
    ANNModelEvaluator.evaluate_and_visualize(ann_model, X_test_scaled, y_test)
