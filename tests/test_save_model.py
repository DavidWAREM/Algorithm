
import unittest
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.models.save_model import save_model, load_model

class TestSaveModel(unittest.TestCase):

    def test_save_and_load_model(self):
        model = RandomForestClassifier()
        filepath = 'results/models/test_model.pkl'
        
        # Save the model
        save_model(model, filepath)
        
        # Load the model
        loaded_model = load_model(filepath)
        
        # Check if the model is saved and loaded correctly
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, RandomForestClassifier)
        
        # Cleanup
        os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
