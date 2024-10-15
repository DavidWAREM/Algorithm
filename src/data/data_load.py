import os
import pandas as pd
import yaml
import logging


class CSVDataLoader:
    def __init__(self, config_file=None):
        """
        Initializes the CSVDataLoader class by loading the configuration file and setting up
        the folder path and required columns. It then loads all valid CSV files from the folder.

        Args:
            config_file (str): Path to the configuration file (optional, defaults to 'config.yaml').

        This class is responsible for loading CSV files from a specified folder, ensuring that they contain the required columns,
        and storing them for further processing.
        """

        # Initialize get logger for data_load
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Logger set for data_load")

        # Get the absolute path to the config file relative to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # Two levels up
        config_file = os.path.join(project_root, 'config', 'config.yaml')  # Default path to config.yaml

        # Load the configuration file
        self.config = self.load_main_config(config_file)

        # Set the folder path from the config file
        self.folder_path = self.config['paths']['folder_path_data']

        # Define the required columns that each CSV file must contain
        self.required_columns = ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE']

        # Initialize a list to store loaded data
        self.all_data = []

        # Load all valid CSV files from the folder
        self.load_all_data()


    def load_main_config(self, config_file):
        """
        Loads the main configuration from the specified YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.

        Returns:
            dict: The loaded configuration dictionary.

        This function reads and parses the YAML configuration file, returning its contents as a dictionary.
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_data_from_csv(self, file_path):
        """
        Loads a single CSV file from the specified file path.

        Args:
            file_path (str): Full path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        This function reads the CSV file and logs its shape for debugging purposes.
        """
        data = pd.read_csv(file_path, sep=';')  # Load CSV with ';' as the separator
        self.logger.debug(f"Loaded data from {file_path} with shape {data.shape}")  # Log the shape of the loaded data
        return data

    def load_all_data(self):
        """
        Loads all valid CSV files from the folder specified in the config.

        This function iterates over all files in the folder, filters those ending with '_Pipes.csv',
        and checks if they contain the required columns. If valid, the data is stored in a list.

        Cave: Beachte, das ich hier aktuell nur die Rohre, nicht die Knoten lade!!
        """
        # Iterate through all files in the folder
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('_Pipes.csv'):  # Only process files that end with '_Pipes.csv'
                file_path = os.path.join(self.folder_path, file_name)
                data = self.load_data_from_csv(file_path)  # Load the data from the CSV file

                # Log the columns present in the current CSV file for debugging
                self.logger.debug(f"Columns in {file_name}: {data.columns.tolist()}")

                # Check if all required columns are present in the DataFrame
                if not all(column in data.columns for column in self.required_columns):
                    self.logger.warning(
                        f"Required columns not found in file: {file_name}")  # Log a warning if columns are missing
                    continue  # Skip this file if required columns are missing

                # Append the valid file name and data to the all_data list
                self.all_data.append((file_name, data))

        # If no valid files are found, raise an error
        if not self.all_data:
            self.logger.error(
                "No valid CSV files found with the required columns.")  # Log an error if no valid files are found
            raise ValueError("No valid CSV files found with the required columns.")  # Raise an exception

        self.logger.info("All data loaded.")

    def get_data(self):
        """
        Returns the loaded data.

        Returns:
            list: A list of tuples containing file names and their corresponding DataFrames.

        This function returns the loaded CSV data stored in the all_data list.
        """
        return self.all_data


# Example usage
if __name__ == "__main__":
    # Instantiate the CSVDataLoader and get the loaded data
    loader = CSVDataLoader()
    data = loader.get_data()

    # Print the file names and shapes of the loaded DataFrames
    for file_name, df in data:
        print(f"File: {file_name}, Shape: {df.shape}")
