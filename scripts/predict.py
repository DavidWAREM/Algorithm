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
    return config

def load_data(config_path):
    data_loader = CSVDataLoader(config_file=config_path)
    all_data = data_loader.get_data()
    combined_data = pd.concat([df for _, df in all_data])
    return combined_data

def preprocess_data(new_data):
    processor = PredictionPreprocessor(new_data)
    processor.load_transformers()
    X_new_scaled = processor.preprocess()
    return X_new_scaled

def predict(X):
    prediction_model = GradientBoostingPrediction()
    y_pred = prediction_model.predict(X)
    return y_pred

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
    new_data = load_data(folder_path_data_predicted)

    # Preprocess new data
    X_new_scaled = preprocess_data(new_data)

    # Make predictions
    y_pred = predict(X_new_scaled)

    # Print the predicted values to console
    print("Predicted values:")
    print(y_pred)
    logger.info("Prediction completed successfully")
