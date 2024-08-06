
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.prediction.make_prediction import make_prediction

class TestMakePrediction(unittest.TestCase):

    def test_make_prediction(self):
        X_new = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        model = RandomForestClassifier()
        model.fit(X_new, [0, 1, 0])  # Train the model for testing
        
        predictions = make_prediction(model, X_new)
        
        # Check if predictions are made correctly
        self.assertEqual(len(predictions), len(X_new))
        self.assertIn(predictions[0], [0, 1])

if __name__ == '__main__':
    unittest.main()
