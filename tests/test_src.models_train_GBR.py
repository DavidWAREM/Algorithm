import unittest
import os
import numpy as np
import logging
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.models.train_GBR import GradientBoostingModel

class TestGradientBoostingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create a sample regression dataset
        cls.X, cls.y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

        # Split the dataset into training and testing sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

        # Initialize the model
        cls.gb_model = GradientBoostingModel(random_state=42)

    def test_train_model(self):
        # Train the model
        self.gb_model.train(self.X_train, self.y_train)
        self.assertIsNotNone(self.gb_model.model, "Model training failed. Model is None.")

    def test_save_and_load_model(self):
        # Save the trained model
        model_file_path = "test_gradient_boosting_model.joblib"
        self.gb_model.save_model(model_file_path)

        # Ensure the model file is created
        self.assertTrue(os.path.exists(model_file_path), "Model file was not created.")

        # Load the model
        self.gb_model.load_model(model_file_path)
        self.assertIsNotNone(self.gb_model.model, "Model loading failed. Model is None.")

        # Cleanup
        if os.path.exists(model_file_path):
            os.remove(model_file_path)

    def test_model_prediction(self):
        # Train the model
        self.gb_model.train(self.X_train, self.y_train)

        # Make predictions
        y_pred = self.gb_model.model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test), "Prediction length does not match test set length.")

        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)

        self.assertGreaterEqual(r2, 0.8, f"Model R2 score is too low: {r2}")

if __name__ == '__main__':
    unittest.main()
