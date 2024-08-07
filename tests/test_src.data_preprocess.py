import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.data.preprocess import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        # Example data
        self.all_data = [
            ('file1.csv', pd.DataFrame({
                'RORL': [1, 2, 3], 'DM': [1, 2, 3], 'RAU': [1, 2, 3], 'FLUSS': [1, 2, 3], 'VM': [1, 2, 3],
                'DPREL': [1, 2, 3], 'RAISE': [1, 2, 3], 'DP': [1, 2, 3]
            })),
            ('file2.csv', pd.DataFrame({
                'RORL': [4, 5, 6], 'DM': [4, 5, 6], 'RAU': [4, 5, 6], 'FLUSS': [4, 5, 6], 'VM': [4, 5, 6],
                'DPREL': [4, 5, 6], 'RAISE': [4, 5, 6], 'DP': [4, 5, 6]
            }))
        ]

    def test_combine_data(self):
        engineer = FeatureEngineer(self.all_data)
        engineer.combine_data()
        combined_data = engineer.combined_data
        self.assertEqual(combined_data.shape, (6, 8))  # 6 rows and 8 columns
        self.assertTrue('RORL' in combined_data.columns)
        self.assertTrue('DM' in combined_data.columns)

    def test_add_polynomial_features(self):
        engineer = FeatureEngineer(self.all_data)
        engineer.combine_data()
        X_combined_poly, y_combined = engineer.add_polynomial_features()
        self.assertEqual(X_combined_poly.shape, (6, 35))  # 6 rows and polynomial features (degree 2) of 7 original features
        self.assertEqual(y_combined.shape, (6,))
        self.assertTrue(np.all(y_combined == [1, 2, 3, 4, 5, 6]))  # Checking target values

    def test_split_data(self):
        engineer = FeatureEngineer(self.all_data)
        engineer.combine_data()
        X_combined_poly, y_combined = engineer.add_polynomial_features()
        X_train, X_test, y_train, y_test = engineer.split_data(X_combined_poly, y_combined)
        self.assertEqual(X_train.shape[0], 4)  # 80% of 6 rows for training
        self.assertEqual(X_test.shape[0], 2)   # 20% of 6 rows for testing
        self.assertEqual(y_train.shape[0], 4)
        self.assertEqual(y_test.shape[0], 2)

    def test_scale_data(self):
        engineer = FeatureEngineer(self.all_data)
        engineer.combine_data()
        X_combined_poly, y_combined = engineer.add_polynomial_features()
        X_train, X_test, y_train, y_test = engineer.split_data(X_combined_poly, y_combined)
        engineer.scale_data(X_train, X_test)
        self.assertEqual(engineer.X_train_scaled.shape, X_train.shape)
        self.assertEqual(engineer.X_test_scaled.shape, X_test.shape)
        self.assertTrue(np.allclose(engineer.X_train_scaled.mean(axis=0), 0, atol=1e-7))  # Mean should be 0
        self.assertTrue(np.allclose(engineer.X_train_scaled.std(axis=0), 1, atol=1e-7))   # Std dev should be 1

    def test_process_features(self):
        engineer = FeatureEngineer(self.all_data)
        engineer.process_features()
        X_train, X_test, y_train, y_test = engineer.get_processed_data()
        self.assertEqual(X_train.shape[0], 4)  # 80% of 6 rows for training
        self.assertEqual(X_test.shape[0], 2)   # 20% of 6 rows for testing
        self.assertEqual(y_train.shape[0], 4)
        self.assertEqual(y_test.shape[0], 2)
        self.assertEqual(X_train.shape[1], 35)  # Polynomial features of 7 original features
        self.assertEqual(X_test.shape[1], 35)
        self.assertTrue(np.allclose(X_train.mean(axis=0), 0, atol=1e-7))  # Mean should be 0
        self.assertTrue(np.allclose(X_train.std(axis=0), 1, atol=1e-7))   # Std dev should be 1

if __name__ == '__main__':
    unittest.main()
