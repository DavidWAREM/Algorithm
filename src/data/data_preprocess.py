import pandas as pd
import logging
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Setup logging for the script with INFO level and a specific format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FeatureEngineer:
    def __init__(self, all_data):
        """
        Initializes the FeatureEngineer class.

        Args:
            all_data (list): A list of tuples containing file names and their corresponding DataFrames.

        This class is responsible for handling feature engineering tasks, including combining data,
        generating polynomial features, scaling, and splitting the data into training and testing sets.
        """
        self.all_data = all_data  # Input data consisting of multiple DataFrames
        # Define paths for saving the polynomial transformer and scaler
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.poly_save_path = os.path.join(project_root, 'results', 'models', 'poly_transformer.joblib')
        self.scaler_save_path = os.path.join(project_root, 'results', 'models', 'scaler.joblib')

        # Initialize attributes for storing processed data
        self.combined_data = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.poly = None
        self.scaler = None
        logging.debug("FeatureEngineer initialized with provided data.")

    def combine_data(self):
        """
        Combines all data into a single DataFrame.

        This method concatenates multiple DataFrames into one and stores the result in `self.combined_data`.
        """
        self.combined_data = pd.concat([data for _, data in self.all_data])  # Combine all DataFrames
        logging.info(f"Combined data shape: {self.combined_data.shape}")  # Log the shape of the combined data
        logging.debug("Data combined successfully.")

    def add_polynomial_features(self):
        """
        Adds polynomial features to the data using a degree of 2.

        This method generates polynomial features from the existing input data,
        fits a PolynomialFeatures transformer, and saves it for future use.

        Returns:
            np.array: Transformed input data with polynomial features.
            pd.Series: Target data (y) which is the 'RAU' column.
        """
        self.poly = PolynomialFeatures(degree=2, include_bias=False)  # Initialize PolynomialFeatures
        # Select the relevant columns for feature engineering
        X_combined = self.combined_data[['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']]
        X_combined_poly = self.poly.fit_transform(X_combined)  # Generate polynomial features
        y_combined = self.combined_data['RAU']  # Target variable
        logging.info("Feature engineering complete with polynomial features of degree 2")
        logging.debug(
            f"Polynomial features added. X_combined shape: {X_combined.shape}, X_combined_poly shape: {X_combined_poly.shape}")

        # Save the polynomial transformer to disk
        joblib.dump(self.poly, self.poly_save_path)
        logging.info(f"Polynomial features transformer saved to {self.poly_save_path}")

        return X_combined_poly, y_combined

    def split_data(self, X_combined_poly, y_combined):
        """
        Splits the data into training and testing sets.

        Args:
            X_combined_poly (np.array): Input data with polynomial features.
            y_combined (pd.Series): Target data (RAU column).

        Returns:
            np.array: X_train (training features).
            np.array: X_test (testing features).
            pd.Series: y_train (training target).
            pd.Series: y_test (testing target).

        This method uses an 80/20 train-test split.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_combined_poly, y_combined, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets with test size 20%")
        logging.debug(
            f"Data split completed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """
        Scales the training and testing data using StandardScaler.

        Args:
            X_train (np.array): Training features.
            X_test (np.array): Testing features.

        This method fits a StandardScaler to the training data and transforms both the training and testing data.
        It also saves the scaler to disk.
        """
        self.scaler = StandardScaler()  # Initialize StandardScaler
        self.X_train_scaled = self.scaler.fit_transform(X_train)  # Scale the training data
        self.X_test_scaled = self.scaler.transform(X_test)  # Scale the testing data
        logging.info("Data scaling complete using StandardScaler")
        logging.debug(
            f"Data scaled. X_train_scaled shape: {self.X_train_scaled.shape}, X_test_scaled shape: {self.X_test_scaled.shape}")

        # Save the scaler to disk
        joblib.dump(self.scaler, self.scaler_save_path)
        logging.info(f"Scaler saved to {self.scaler_save_path}")

    def process_features(self):
        """
        Handles the complete feature engineering process.

        This method combines the data, adds polynomial features, splits the data, and scales the features.
        """
        logging.info("Starting feature engineering process.")
        self.combine_data()  # Combine all input data
        X_combined_poly, y_combined = self.add_polynomial_features()  # Add polynomial features
        X_train, X_test, y_train, y_test = self.split_data(X_combined_poly, y_combined)  # Split the data
        self.scale_data(X_train, X_test)  # Scale the data
        self.y_train = y_train  # Store the training target
        self.y_test = y_test  # Store the testing target
        logging.info("Feature engineering process completed successfully.")

    def get_processed_data(self):
        """
        Returns the processed data after feature engineering.

        Returns:
            np.array: X_train_scaled.
            np.array: X_test_scaled.
            pd.Series: y_train.
            pd.Series: y_test.

        This method returns the scaled training and testing features along with the corresponding target values.
        """
        logging.debug("Processed data retrieved.")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test


class PredictionPreprocessor:
    def __init__(self, new_data):
        """
        Initializes the PredictionPreprocessor class with new data for prediction.

        Args:
            new_data (pd.DataFrame): The new data for which predictions need to be made.

        This class handles preprocessing for new data by loading previously saved transformers (polynomial and scaler)
        and applying them to the new data.
        """
        self.new_data = new_data
        # Define paths to load the saved polynomial transformer and scaler
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.poly_path = os.path.join(project_root, 'results', 'models', 'poly_transformer.joblib')
        self.scaler_path = os.path.join(project_root, 'results', 'models', 'scaler.joblib')

        # Initialize variables for the transformers
        self.poly = None
        self.scaler = None
        self.processed_data = None
        logging.debug("PredictionPreprocessor initialized with provided data.")

    def load_transformers(self):
        """
        Loads the saved polynomial transformer and scaler from disk.

        This method loads the transformers previously saved during the training phase.
        """
        self.poly = joblib.load(self.poly_path)  # Load the polynomial transformer
        self.scaler = joblib.load(self.scaler_path)  # Load the scaler
        logging.info("Loaded polynomial transformer and scaler from disk.")

    def preprocess(self):
        """
        Preprocesses the new data by applying polynomial transformation and scaling.

        Returns:
            pd.DataFrame: Preprocessed and scaled data ready for prediction.

        This method applies the saved polynomial transformer and scaler to the new data and returns the result.
        """
        logging.info("Starting preprocessing for prediction data.")
        # Select the relevant columns from the new data for preprocessing
        X_new = self.new_data[['RORL', 'DM', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']]
        X_new_poly = self.poly.transform(X_new)  # Apply the polynomial transformation
        X_new_scaled = self.scaler.transform(X_new_poly)  # Scale the transformed data
        self.processed_data = pd.DataFrame(X_new_scaled)  # Store the processed data as a DataFrame
        logging.info("Preprocessing for prediction data completed successfully.")
        return self.processed_data  # Return the preprocessed data


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

    # Initialize and run the FeatureEngineer for the simulated data
    engineer = FeatureEngineer(all_data)
    engineer.process_features()  # Run feature engineering
    X_train, X_test, y_train, y_test = engineer.get_processed_data()  # Retrieve processed data
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")
    logging.info(f"Training target shape: {y_train.shape}")
    logging.info(f"Testing target shape: {y_test.shape}")
    logging.info("FeatureEngineer example usage completed.")

    # Example usage for prediction data
    new_data = pd.DataFrame(
        {'RORL': [7, 8, 9], 'DM': [7, 8, 9], 'FLUSS': [7, 8, 9], 'VM': [7, 8, 9], 'DPREL': [7, 8, 9],
         'RAISE': [7, 8, 9], 'DP': [7, 8, 9]}
    )

    # Initialize and run the PredictionPreprocessor for new data
    prediction_preprocessor = PredictionPreprocessor(new_data)
    prediction_preprocessor.load_transformers()  # Load transformers
    processed_new_data = prediction_preprocessor.preprocess()  # Preprocess the new data
    logging.info(f"Processed new data shape: {processed_new_data.shape}")
    logging.info("PredictionPreprocessor example usage completed.")
