import pandas as pd
import logging
import os
from data.data_loader import DataLoader
from data.preprocess import DataProcessor


def configure_logging(log_file):
    """
    Configure logging to write logs to a specified file.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO to capture all messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """
    Main function to load, process, and save data.
    """
    # Path to the CSV file
    file_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\25_Schopfloch.CSV'

    # Determine the directory of the current script and set the log file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'data_processing.log')

    # Configure logging
    configure_logging(log_file)

    logging.info(f'Starting data processing for file: {file_path}')

    # Create an instance of the DataLoader and load the data
    try:
        data_loader = DataLoader(file_path)
        data = data_loader.custom_read_csv()
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    # Create an instance of the DataProcessor and process the data
    try:
        data_processor = DataProcessor(data, file_path)
        kno_df, lei_df = data_processor.split_data()
        logging.info('Data split successfully.')

        # Save the resulting DataFrames to CSV files
        data_processor.save_dataframes()

        # Log 'KNAM' values for rows where 'ABGAENGE' is '2'
        data_processor.log_knam_for_abgaenge_2(kno_df)

    except Exception as e:
        logging.error(f'Error processing data: {e}')
        return

    logging.info("Data processing complete. DataFrames saved.")


if __name__ == "__main__":
    main()
