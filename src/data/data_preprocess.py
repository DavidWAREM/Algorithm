import pandas as pd
import logging
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, all_data):
        self.all_data = all_data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.poly_save_path = os.path.join(project_root, 'results', 'models', 'poly_transformer.joblib')
        self.scaler_save_path = os.path.join(project_root, 'results', 'models', 'scaler.joblib')
        self.combined_data = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.poly = None
        self.scaler = None
        logging.debug("FeatureEngineer initialized with provided data.")

    def combine_data(self):
        self.combined_data = pd.concat([data for _, data in self.all_data])
        logging.info(f"Combined data shape: {self.combined_data.shape}")
        logging.debug("Data combined successfully.")

    def add_polynomial_features(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X_combined = self.combined_data[['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']]
        X_combined_poly = self.poly.fit_transform(X_combined)
        y_combined = self.combined_data['RAU']
        logging.info("Feature engineering complete with polynomial features of degree 2")
        logging.debug(f"Polynomial features added. X_combined shape: {X_combined.shape}, X_combined_poly shape: {X_combined_poly.shape}")

        # Save the polynomial features transformer
        joblib.dump(self.poly, self.poly_save_path)
        logging.info(f"Polynomial features transformer saved to {self.poly_save_path}")

        return X_combined_poly, y_combined

    def split_data(self, X_combined_poly, y_combined):
        X_train, X_test, y_train, y_test = train_test_split(X_combined_poly, y_combined, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets with test size 20%")
        logging.debug(f"Data split completed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):z7
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        logging.info("Data scaling complete using StandardScaler")
        logging.debug(f"Data scaled. X_train_scaled shape: {self.X_train_scaled.shape}, X_test_scaled shape: {self.X_test_scaled.shape}")

        # Save the scaler
        joblib.dump(self.scaler, self.scaler_save_path)
        logging.info(f"Scaler saved to {self.scaler_save_path}")

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

class PredictionPreprocessor:
    def __init__(self, new_data):
        self.new_data = new_data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.poly_path = os.path.join(project_root, 'results', 'models', 'poly_transformer.joblib')
        self.scaler_path = os.path.join(project_root, 'results', 'models', 'scaler.joblib')
        self.poly = None
        self.scaler = None
        self.processed_data = None
        logging.debug("PredictionPreprocessor initialized with provided data.")

    def load_transformers(self):
        self.poly = joblib.load(self.poly_path)
        self.scaler = joblib.load(self.scaler_path)
        logging.info("Loaded polynomial transformer and scaler from disk.")

    def preprocess(self):
        logging.info("Starting preprocessing for prediction data.")
        X_new = self.new_data[['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']]
        X_new_poly = self.poly.transform(X_new)
        X_new_scaled = self.scaler.transform(X_new_poly)
        self.processed_data = pd.DataFrame(X_new_scaled)
        logging.info("Preprocessing for prediction data completed successfully.")
        return self.processed_data

# Example usage for training data
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

    # Example usage for prediction data
    new_data = pd.DataFrame(
        {'RORL': [7, 8, 9], 'DM': [7, 8, 9], 'FLUSS': [7, 8, 9], 'VM': [7, 8, 9], 'DPREL': [7, 8, 9], 'RAISE': [7, 8, 9], 'DP': [7, 8, 9]}
    )
    prediction_preprocessor = PredictionPreprocessor(new_data)
    prediction_preprocessor.load_transformers()
    processed_new_data = prediction_preprocessor.preprocess()
    logging.info(f"Processed new data shape: {processed_new_data.shape}")
    logging.info("PredictionPreprocessor example usage completed.")
