import logging
import argparse
import yaml
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logging_config import setup_logging
from src.data.data_load import CSVDataLoader
from src.data.data_preprocess import PredictionPreprocessor
from src.prediction.predictions_GBR import GradientBoostingPrediction

# Setup logging
setup_logging()

def load_config(config_file='config/config.yaml'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    config_path = os.path.join(project_root, config_file)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded from {config_path}")
    return config

def load_data(folder_path_data_predicted):
    logging.info(f"Loading data from folder: {folder_path_data_predicted}")
    all_files = [f for f in os.listdir(folder_path_data_predicted) if f.endswith('.csv')]
    logging.info(f"Found {len(all_files)} CSV files to load")
    dataframes = [(file, pd.read_csv(os.path.join(folder_path_data_predicted, file), sep=';')) for file in all_files]
    combined_data = pd.concat([df for _, df in dataframes], ignore_index=True)
    logging.info(f"Combined data shape: {combined_data.shape}")
    return combined_data, dataframes

def preprocess_data(new_data):
    logging.info("Starting data preprocessing")
    processor = PredictionPreprocessor(new_data)
    processor.load_transformers()
    X_new_scaled = processor.preprocess()
    logging.info("Data preprocessing completed")
    return X_new_scaled

def predict(X):
    logging.info("Starting prediction")
    prediction_model = GradientBoostingPrediction()
    y_pred = prediction_model.predict(X)
    logging.info("Prediction completed")
    return y_pred

def save_predictions(y_pred, dataframes, folder_path_data_results):
    """
    Fügt die Daten y_pred in den jeweiligen DataFrame ein und speichert die aktualisierten Daten als neue CSV-Dateien.
    Die Daten y_pred werden in der Spalte mit dem Header RAU eingefügt.
    """
    logging.info("Saving predictions to CSV")
    start_idx = 0
    for file_name, df in dataframes:
        end_idx = start_idx + len(df)
        df['RAU'] = y_pred[start_idx:end_idx]
        output_file_name = f"{os.path.splitext(file_name)[0]}_predicted.csv"
        output_path = os.path.join(folder_path_data_results, output_file_name)
        df.to_csv(output_path, index=False, sep=';')
        logging.info(f"Predictions saved to {output_path}")
        start_idx = end_idx

if __name__ == "__main__":
    # Load configuration file
    config = load_config()

    parser = argparse.ArgumentParser(description='Predict with a trained model')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to use for prediction (e.g., gradient_boosting)')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(f"Predicting with algorithm: {args.algorithm}")

    # Load new data
    folder_path_data_predicted = config['paths']['folder_path_data_predicted']
    combined_data, dataframes = load_data(folder_path_data_predicted)

    # Preprocess new data
    X_new_scaled = preprocess_data(combined_data)

    # Make predictions
    y_pred = predict(X_new_scaled)
    logger.info("Prediction completed successfully")

    # Save predictions to individual CSV files
    folder_path_data_results = config['paths']['folder_path_data_results']
    save_predictions(y_pred, dataframes, folder_path_data_results)
