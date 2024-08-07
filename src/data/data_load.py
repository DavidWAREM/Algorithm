import os
import pandas as pd
import yaml
import logging

class CSVDataLoader:
    def __init__(self, config_file=None):
        # Verwenden Sie den absoluten Pfad zur Konfigurationsdatei
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
        config_file = os.path.join(project_root, 'config', 'config.yaml')

        self.config = self.load_main_config(config_file)
        self.folder_path = self.config['paths']['folder_path_data']
        self.required_columns = ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP']
        self.all_data = []
        self.load_all_data()

    def load_main_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_data_from_csv(self, file_path):
        data = pd.read_csv(file_path, sep=';')
        logging.debug(f"Loaded data from {file_path} with shape {data.shape}")
        return data

    def load_all_data(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('_Pipes.csv'):  # Only process files ending with '_Pipes.csv'
                file_path = os.path.join(self.folder_path, file_name)
                data = self.load_data_from_csv(file_path)

                # Log the columns of the file for debugging purposes
                logging.debug(f"Columns in {file_name}: {data.columns.tolist()}")

                # Check if required columns are present
                if not all(column in data.columns for column in self.required_columns):
                    logging.warning(f"Required columns not found in file: {file_name}")
                    continue

                self.all_data.append((file_name, data))

        # Raise an error if no valid CSV files are found
        if not self.all_data:
            logging.error("No valid CSV files found with the required columns.")
            raise ValueError("No valid CSV files found with the required columns.")

    def get_data(self):
        return self.all_data


# Example usage
if __name__ == "__main__":
    loader = CSVDataLoader()
    data = loader.get_data()
    for file_name, df in data:
        print(f"File: {file_name}, Shape: {df.shape}")
