import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.prediction.predictions_GBR import GradientBoostingPrediction
from src.evaluation.evaluation_GBR import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):

    @patch.object(GradientBoostingPrediction, 'predict')
    @patch.object(GradientBoostingPrediction, '__init__', lambda x: None)
    @patch('matplotlib.pyplot.show')
    def test_evaluate_and_visualize(self, mock_show, mock_predict):
        # Mock data
        X_test = np.random.rand(100, 20)
        y_test = np.random.rand(100)

        # Mock predictions
        y_pred = np.random.rand(100)
        mock_predict.return_value = y_pred

        # Calculate expected metrics
        expected_mse = mean_squared_error(y_test, y_pred)
        expected_rmse = np.sqrt(expected_mse)
        expected_r2 = r2_score(y_test, y_pred)

        # Call the method
        ModelEvaluator.evaluate_and_visualize(X_test, y_test)

        # Assertions
        mock_predict.assert_called_once_with(X_test)
        self.assertEqual(mock_predict.call_count, 1)
        self.assertEqual(mock_show.call_count, 1)

        # Capture logging output
        with self.assertLogs(level='INFO') as log:
            ModelEvaluator.evaluate_and_visualize(X_test, y_test)
            self.assertIn(f"INFO:root:Model Evaluation - MSE: {expected_mse}, RMSE: {expected_rmse}, R2: {expected_r2}", log.output)

if __name__ == '__main__':
    unittest.main()
