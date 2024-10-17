import os
import logging  # Korrigierter Import
import yaml
from data.rawdata_load import DataLoader  # Custom class to load raw data
from data.rawdata_preprocess import DataProcessor, DataCombiner  # Custom class to process raw data
from src.logging_config import setup_logging  # Custom function to set up logging


def load_config(config_file='config/config.yaml'):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file (default is 'config/config.yaml').

    Returns:
        dict: Configuration dictionary containing paths and other settings.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root directory (one level up from the script)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Construct the full path to the YAML configuration file
    config_path = os.path.join(project_root, config_file)

    # Open and load the YAML file as a Python dictionary
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config  # Return the loaded configuration


def process_file(file_path, file_name, logger):
    """
    Process a single CSV file: load, preprocess, split, and save the data.

    Args:
        file_path (str): Full path to the CSV file.
        file_name (str): Name of the file being processed.
        logger (logging.Logger): Logger instance for logging.

    This function handles the entire data processing pipeline for a single file.
    It loads the file, preprocesses the data, splits the data into two DataFrames, and then processes them further (e.g., combining pipes).
    """
    logger.debug(f"Starting data processing for file: {file_name}")  # Log the start of the file processing

    try:
        # Create an instance of DataLoader to read the CSV data
        data_loader = DataLoader(file_path)
        data = data_loader.custom_read_csv()  # Load the CSV data using a custom method
        logger.debug("Data loaded successfully.")  # Log successful data loading

        # Create an instance of DataProcessor to handle the loaded data
        data_processor = DataProcessor(data, file_path)

        # Split the data into two DataFrames: 'KNO' and 'LEI'
        kno_df, lei_df = data_processor.split_data()
        logger.debug("Data split successfully.")  # Log successful data split

        # Save the split DataFrames to separate files
        data_processor.save_dataframes()

        logger.debug(f"Data processing complete for file: {file_name}")  # Log completion of processing

    except Exception as e:
        # Log any errors that occur during processing
        logger.error(f"Error processing data for file {file_name}: {e}")


def main():
    """
    Main function that sets up logging, loads configuration, and processes files.

    This function sets up logging, loads the YAML configuration, and processes each CSV file in the specified directory.
    """
    setup_logging()  # Set up the logging configuration
    logger = logging.getLogger(__name__)  # Create a logger for this script
    logger.info('Logger gestartet')

    # Load configuration settings from the YAML file
    config = load_config()

    # Directory containing the raw CSV files
    data_dir = config['paths']['folder_path_rawdata']
    logger.info(f"Processing files from directory: {data_dir}")

    # Loop through all files in the directory and process each CSV file
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):  # Only process files with a '.csv' extension
            file_path = os.path.join(data_dir, file_name)  # Get the full path to the CSV file
            process_file(file_path, file_name, logger)  # Process the file

    # Initialize DataCombiner and combine the processed files
    data_combiner = DataCombiner(data_dir)
    data_combiner.combine_with_without_load('Pipes')
    data_combiner.combine_with_without_load('Node')


if __name__ == "__main__":
    main()  # Entry point: run the main function if this script is executed directly
