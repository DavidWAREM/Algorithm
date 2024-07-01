import os
import logging
import pandas as pd
from data.data_loader import DataLoader
from data.preprocess import DataProcessor


def main():
    # Setup logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'data_processing.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Add console handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Starting data processing for file: C:\\Users\\d.muehlfeld\\Berechnungsdaten\\11_Spechbach_RNAB.CSV")

    try:
        # Path to the CSV file
        file_path = r'C:\\Users\\d.muehlfeld\\Berechnungsdaten\\11_Spechbach_RNAB.CSV'

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

        logging.info("Data processing complete.")

    except Exception as e:
        logging.error(f"Error processing data: {e}")


if __name__ == "__main__":
    main()
