import os
import logging
import pandas as pd
from data.data_loader import DataLoader
from data.preprocess import DataProcessor

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
    # Get the directory of the script being executed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'data_processing.log')

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the root logger level to DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # Ensure UTF-8 encoding for the file handler
            logging.StreamHandler()  # Console handler
        ]
    )

    # Set specific levels for handlers
    logging.getLogger().handlers[0].setLevel(logging.DEBUG)  # FileHandler
    logging.getLogger().handlers[1].setLevel(logging.INFO)   # StreamHandler

    # Directory containing the CSV files
    data_dir = r'C:\\Users\\d.muehlfeld\\Berechnungsdaten\\'

    # Process each file in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            process_file(file_path, file_name)

if __name__ == "__main__":
    main()
