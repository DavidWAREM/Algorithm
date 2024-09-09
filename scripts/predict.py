import logging
import argparse
import yaml
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logging_config import setup_logging  # Custom logging configuration
from src.data.data_load import CSVDataLoader  # Custom CSV data loading utility
from src.data.data_preprocess import PredictionPreprocessor  # Preprocessing class for predictions
from src.prediction.predictions_GBR import GradientBoostingPrediction  # Custom prediction class

# Setup logging using a custom configuration
setup_logging()


def load_config(config_file='config/config.yaml'):
    """
    Load the configuration file for the script from the specified path.

    Args:
        config_file (str): Path to the configuration file, default is 'config/config.yaml'.

    Returns:
        dict: The loaded configuration settings.

    This function constructs the absolute path to the config file and loads its contents using yaml.
    It logs the success of loading the configuration.
    """
    # Get the current script directory and project root for constructing the config file path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    config_path = os.path.join(project_root, config_file)

    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded from {config_path}")
    return config  # Return the loaded configuration as a dictionary


def load_data(folder_path_data_predicted):
    """
    Load all CSV files from the specified folder, concatenate them into a single DataFrame.

    Args:
        folder_path_data_predicted (str): Path to the folder containing CSV files for prediction.

    Returns:
        tuple: A combined pandas DataFrame and a list of tuples with filenames and DataFrames.

    This function finds all CSV files in the folder, loads each into a DataFrame,
    and concatenates them into one large DataFrame. It logs how many files were loaded and the shape of the combined data.
    """
    logging.info(f"Loading data from folder: {folder_path_data_predicted}")

    # List all CSV files in the given folder
    all_files = [f for f in os.listdir(folder_path_data_predicted) if f.endswith('.csv')]
    logging.info(f"Found {len(all_files)} CSV files to load")

    # Load each file into a DataFrame and store them in a list
    dataframes = [(file, pd.read_csv(os.path.join(folder_path_data_predicted, file), sep=';')) for file in all_files]

    # Concatenate all DataFrames into one combined DataFrame
    combined_data = pd.concat([df for _, df in dataframes], ignore_index=True)
    logging.info(f"Combined data shape: {combined_data.shape}")  # Log the shape of the combined DataFrame

    return combined_data, dataframes  # Return the combined DataFrame and list of individual DataFrames


def preprocess_data(new_data):
    """
    Preprocess the loaded data using the PredictionPreprocessor.

    Args:
        new_data (pd.DataFrame): The raw DataFrame to preprocess.

    Returns:
        np.array: Scaled and preprocessed data ready for prediction.

    This function creates an instance of `PredictionPreprocessor`, loads any necessary transformations,
    and applies the preprocessing steps. The preprocessed data is returned.
    """
    logging.info("Starting data preprocessing")

    # Create an instance of PredictionPreprocessor with the new data
    processor = PredictionPreprocessor(new_data)

    # Load the transformers (e.g., scalers, encoders) for preprocessing
    processor.load_transformers()

    # Apply preprocessing (e.g., scaling)
    X_new_scaled = processor.preprocess()

    logging.info("Data preprocessing completed")
    return X_new_scaled  # Return the preprocessed and scaled data ready for prediction


def predict(X):
    """
    Perform predictions using a trained Gradient Boosting model.

    Args:
        X (np.array): Preprocessed input data for prediction.

    Returns:
        np.array: The predicted values.

    This function initializes the `GradientBoostingPrediction` class and calls its `predict()` method
    to generate predictions based on the provided preprocessed data.
    """
    logging.info("Starting prediction")

    # Initialize the prediction model (Gradient Boosting Regressor)
    prediction_model = GradientBoostingPrediction()

    # Predict the target variable (e.g., 'RAU') based on the preprocessed input
    y_pred = prediction_model.predict(X)

    logging.info("Prediction completed")
    return y_pred  # Return the predicted values


def save_predictions(y_pred, dataframes, folder_path_data_results):
    """
    Save the predictions back into the original DataFrames and write them as new CSV files.

    Args:
        y_pred (np.array): Predicted values to be saved.
        dataframes (list): List of tuples (filename, DataFrame) for each original file.
        folder_path_data_results (str): Path to the folder where the new CSV files should be saved.

    This function takes the predictions, inserts them into the respective DataFrames,
    and saves the updated DataFrames as new CSV files. The predictions are added to a column named 'RAU'.
    """
    logging.info("Saving predictions to CSV")

    # Start index for slicing the predictions to fit individual DataFrames
    start_idx = 0
    for file_name, df in dataframes:
        # Determine the range of predictions for the current DataFrame
        end_idx = start_idx + len(df)
        df['RAU'] = y_pred[start_idx:end_idx]  # Add the predicted values to a new column 'RAU'

        # Create the output file name by appending '_predicted' to the original file name
        output_file_name = f"{os.path.splitext(file_name)[0]}_predicted.csv"
        output_path = os.path.join(folder_path_data_results, output_file_name)

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_path, index=False, sep=';')
        logging.info(f"Predictions saved to {output_path}")

        # Update the start index for the next DataFrame
        start_idx = end_idx


if __name__ == "__main__":
    # Load configuration settings from the YAML config file
    config = load_config()

    # Set up argument parsing for command-line options
    parser = argparse.ArgumentParser(description='Predict with a trained model')
    parser.add_argument('--algorithm', type=str, required=True,
                        help='Algorithm to use for prediction (e.g., gradient_boosting)')
    args = parser.parse_args()

    # Log the algorithm chosen for prediction
    logger = logging.getLogger(__name__)
    logger.info(f"Predicting with algorithm: {args.algorithm}")

    # Load new data for prediction from the folder specified in the config
    folder_path_data_predicted = config['paths']['folder_path_data_predicted']
    combined_data, dataframes = load_data(folder_path_data_predicted)

    # Preprocess the loaded data
    X_new_scaled = preprocess_data(combined_data)

    # Make predictions using the preprocessed data
    y_pred = predict(X_new_scaled)
    logger.info("Prediction completed successfully")

    # Save the predictions to CSV files in the results folder specified in the config
    folder_path_data_results = config['paths']['folder_path_data_results']
    save_predictions(y_pred, dataframes, folder_path_data_results)
