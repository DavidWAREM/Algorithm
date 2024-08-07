import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import logging


class FeatureEngineer:
    def __init__(self, all_data):
        self.all_data = all_data
        self.combined_data = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        logging.debug("FeatureEngineer initialized with provided data.")

    def combine_data(self):
        self.combined_data = pd.concat([data for _, data in self.all_data])
        logging.info(f"Combined data shape: {self.combined_data.shape}")
        logging.debug("Data combined successfully.")

    def add_polynomial_features(self):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_combined = self.combined_data[['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']]
        X_combined_poly = poly.fit_transform(X_combined)
        y_combined = self.combined_data['RAU']
        logging.info("Feature engineering complete with polynomial features of degree 2")
        logging.debug(
            f"Polynomial features added. X_combined shape: {X_combined.shape}, X_combined_poly shape: {X_combined_poly.shape}")
        return X_combined_poly, y_combined

    def split_data(self, X_combined_poly, y_combined):
        X_train, X_test, y_train, y_test = train_test_split(X_combined_poly, y_combined, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets with test size 20%")
        logging.debug(
            f"Data split completed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        logging.info("Data scaling complete using StandardScaler")
        logging.debug(
            f"Data scaled. X_train_scaled shape: {self.X_train_scaled.shape}, X_test_scaled shape: {self.X_test_scaled.shape}")

    def process_features(self):
        logging.info("Starting feature engineering process.")
        self.combine_data()
        X_combined_poly, y_combined = self.add_polynomial_features()
        X_train, X_test, y_train, y_test = self.split_data(X_combined_poly, y_combined)
        self.scale_data(X_train, X_test)
        self.y_train = y_train
        self.y_test = y_test
        logging.info("Feature engineering process completed successfully.")

    def get_processed_data(self):
        logging.debug("Processed data retrieved.")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info("Starting FeatureEngineer example usage.")
    # Simulated data loading process
    all_data = [
        ('file1.csv', pd.DataFrame(
            {'RORL': [1, 2, 3], 'DM': [1, 2, 3], 'RAU': [1, 2, 3], 'FLUSS': [1, 2, 3], 'VM': [1, 2, 3],
             'DPREL': [1, 2, 3], 'RAISE': [1, 2, 3], 'DP': [1, 2, 3]})),
        ('file2.csv', pd.DataFrame(
            {'RORL': [4, 5, 6], 'DM': [4, 5, 6], 'RAU': [4, 5, 6], 'FLUSS': [4, 5, 6], 'VM': [4, 5, 6],
             'DPREL': [4, 5, 6], 'RAISE': [4, 5, 6], 'DP': [4, 5, 6]}))
    ]

    engineer = FeatureEngineer(all_data)
    engineer.process_features()
    X_train, X_test, y_train, y_test = engineer.get_processed_data()
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")
    logging.info(f"Training target shape: {y_train.shape}")
    logging.info(f"Testing target shape: {y_test.shape}")
    logging.info("FeatureEngineer example usage completed.")
