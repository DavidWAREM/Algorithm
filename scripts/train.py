import os
import logging
import yaml
from data.rawdata_loader import DataLoader
from data.rawdata_preprocess import DataProcessor


def load_config(config_file='config/config.yaml'):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    # Get the directory of the script being executed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Move up one level to the project root
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    # Construct the full path to the config file
    config_path = os.path.join(project_root, config_file)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_file(file_path, file_name):
    logging.info(f"Starting data processing for file: {file_name}")
    try:
        # Create an instance of the DataLoader and load the data
        data_loader = DataLoader(file_path)
        data = data_loader.custom_read_csv()
        logging.info("Data loaded successfully.")

        # Create an instance of the DataProcessor and process the data
        data_processor = DataProcessor(data, file_path)
        kno_df, lei_df = data_processor.split_data()

        logging.info("Data split successfully.")
        data_processor.save_dataframes()

        # Combine connected pipes
        data_processor.combine_connected_pipes(kno_df, lei_df)

        logging.info(f"Data processing complete for file: {file_name}")

    except Exception as e:
        logging.error(f"Error processing data for file {file_name}: {e}")

def main():
    # Load configuration
    config = load_config()

    # Directory containing the CSV files
    data_dir = config['paths']['folder_path_rawdata']

    # Process each file in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            process_file(file_path, file_name)

if __name__ == "__main__":
    main()
