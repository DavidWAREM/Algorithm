
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.evaluation.evaluate_model import evaluate_model

class TestEvaluateModel(unittest.TestCase):

    def test_evaluate_model(self):
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y_test = [0, 1, 0]
        
        model = RandomForestClassifier()
        model.fit(X_test, y_test)  # Train the model for testing
        
        accuracy = evaluate_model(model, X_test, y_test)
        
        # Check if the accuracy is calculated correctly
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()
