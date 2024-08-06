
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.train_model import train_model, get_model

class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y_train = [0, 1, 0]
        
        model = get_model('random_forest')
        trained_model = train_model(model, X_train, y_train)
        
        # Check if the model is trained correctly
        self.assertIsNotNone(trained_model)
        self.assertEqual(len(trained_model.predict(X_train)), len(y_train))

if __name__ == '__main__':
    unittest.main()
