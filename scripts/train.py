import os
import logging
import pandas as pd
from data.data_loader import DataLoader
from data.preprocess import DataProcessor


def process_file(file_path):
    logging.info(f"Starting data processing for file: {file_path}")
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

        logging.info(f"Data processing complete for file: {file_path}")

    except Exception as e:
        logging.error(f"Error processing data for file {file_path}: {e}")


def main():
    # Setup logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'data_processing.log')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Add console handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Directory containing the CSV files
    data_dir = r'C:\\Users\\d.muehlfeld\\Berechnungsdaten\\'

    # Process each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.CSV'):
            file_path = os.path.join(data_dir, filename)
            process_file(file_path)


if __name__ == "__main__":
    main()
