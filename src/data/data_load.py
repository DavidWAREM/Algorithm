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

        # Initialize logger for data_load
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set logger level to INFO
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.debug("Logger set for data_load")

        # Get the absolute path to the config file relative to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # Two levels up
        config_file = os.path.join(project_root, 'config', 'config.yaml')  # Default path to config.yaml

        # Load the configuration file
        self.config = self.load_main_config(config_file)

        # Set the folder path from the config file
        self.folder_path = self.config['paths']['folder_path_data']

        # Define the required columns that each CSV file must contain, inklusive 'Type'
        self.required_columns = [
            'Type', 'ANFNAM', 'ENDNAM', 'RORL', 'DM', 'RAU', 'DPREL',
            'ROHRTYP', 'RAISE', 'VM_WL', 'FLUSS_WL',
            'VM_WOL', 'FLUSS_WOL', 'RE_WL', 'RE_WOL',
            'delta_PRECH_WOL', 'delta_PRECH_WL'
        ]

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
        self.logger.debug(f"Configuration loaded from {config_file}")
        return config

    def load_data_from_csv(self, file_path):
        """
        Loads a single CSV file from the specified file path and filters rows where the 'Type' column is 'LEI'.

        Args:
            file_path (str): Full path to the CSV file.

        Returns:
            pd.DataFrame: The filtered data as a pandas DataFrame without the 'Type' column.

        This function reads the CSV file, logs its shape for debugging purposes, and filters rows based on the 'Type' column value.
        """
        try:
            data = pd.read_csv(
                file_path,
                sep=';',
                usecols=self.required_columns
            )
            self.logger.debug(f"Loaded data from {file_path} with shape {data.shape}")  # Log the shape of the loaded data
        except ValueError as e:
            self.logger.warning(f"Skipping file {file_path} due to missing columns: {e}")
            return pd.DataFrame()  # Return empty DataFrame if required columns are missing

        if data.empty:
            self.logger.warning(f"The file {file_path} is empty after loading.")
            return data

        # Sicherstellen, dass die 'Type'-Spalte vorhanden ist
        if 'Type' not in data.columns:
            self.logger.warning(f"The 'Type' column is missing in file: {file_path}")
            return pd.DataFrame()

        # Temporär: Inhalt der Daten anzeigen (optional)
        self.logger.debug(f"Data preview from {file_path}:\n{data.head()}")

        # Sicherstellen, dass die 'Type'-Spalte als String behandelt wird und mögliche NaNs handhaben
        data['Type'] = data['Type'].astype(str)

        # Konvertiere die notwendigen Spalten in die gewünschten Datentypen
        try:
            data['RORL'] = pd.to_numeric(data['RORL'], errors='coerce')
            data['DM'] = pd.to_numeric(data['DM'], errors='coerce').astype('Int64')  # Verwende Int64, um NaNs zu erlauben
            data['RAU'] = pd.to_numeric(data['RAU'], errors='coerce')
            data['DPREL'] = pd.to_numeric(data['DPREL'], errors='coerce')
            data['RAISE'] = pd.to_numeric(data['RAISE'], errors='coerce')
            data['VM_WL'] = pd.to_numeric(data['VM_WL'], errors='coerce')
            data['FLUSS_WL'] = pd.to_numeric(data['FLUSS_WL'], errors='coerce')
            data['VM_WOL'] = pd.to_numeric(data['VM_WOL'], errors='coerce')
            data['FLUSS_WOL'] = pd.to_numeric(data['FLUSS_WOL'], errors='coerce')
            data['RE_WL'] = pd.to_numeric(data['RE_WL'], errors='coerce').astype('Int64')
            data['RE_WOL'] = pd.to_numeric(data['RE_WOL'], errors='coerce').astype('Int64')
            data['delta_PRECH_WOL'] = pd.to_numeric(data['delta_PRECH_WOL'], errors='coerce')
            data['delta_PRECH_WL'] = pd.to_numeric(data['delta_PRECH_WL'], errors='coerce')
        except Exception as e:
            self.logger.warning(f"Error converting data types in file {file_path}: {e}")
            return pd.DataFrame()

        # Filter rows where the 'Type' column has the value 'LEI', case-insensitive und entfernt Whitespaces
        filtered_data = data[data['Type'].str.strip().str.upper() == 'LEI']
        self.logger.debug(f"Filtered data from {file_path} to shape {filtered_data.shape}")

        # Entferne die 'Type'-Spalte aus dem DataFrame
        if 'Type' in filtered_data.columns:
            filtered_data = filtered_data.drop(columns=['Type'])
            self.logger.debug(f"'Type' column removed. New shape: {filtered_data.shape}")

        return filtered_data

    def load_all_data(self):
        """
        Loads all valid CSV files from the folder specified in the config.

        This function iterates over all files in the folder, filters those ending with '_combined.csv',
        and checks if they contain the required columns. If valid, it filters the data for rows where
        the 'Type' column is 'LEI' and stores them in a list.

        Cave: Beachte, dass ich hier aktuell nur die Rohre, nicht die Knoten lade!!
        """
        # Iterate through all files in the folder
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('_combined.csv'):  # Now processing files that end with '_combined.csv'
                file_path = os.path.join(self.folder_path, file_name)
                self.logger.debug(f"Processing file: {file_name}")

                data = self.load_data_from_csv(file_path)  # Load and filter the data from the CSV file

                if data.empty:
                    self.logger.warning(f"No rows with 'LEI' found in file: {file_name}")
                    continue  # Skip this file if no relevant rows are found

                # Log the columns present in the current CSV file for debugging
                self.logger.debug(f"Columns in {file_name}: {data.columns.tolist()}")

                # Check if all required columns (excluding 'Type') are present in the DataFrame
                required_columns_excl_type = [col for col in self.required_columns if col != 'Type']
                if not all(column in data.columns for column in required_columns_excl_type):
                    self.logger.warning(
                        f"Required columns not found in file: {file_name}")  # Log a warning if columns are missing
                    continue  # Skip this file if required columns are missing

                # Append the valid file name and data to the all_data list
                self.all_data.append((file_name, data))
                self.logger.debug(f"Loaded and filtered data from {file_name}")

        # If no valid files are found, raise an error
        if not self.all_data:
            self.logger.error(
                "No valid CSV files found with the required columns and 'LEI' rows.")  # Log an error if no valid files are found
            raise ValueError("No valid CSV files found with the required columns and 'LEI' rows.")  # Raise an exception

        self.logger.info("All data loaded and filtered.")

    def get_data(self):
        """
        Returns the loaded data.

        Returns:
            list: A list of tuples containing file names and their corresponding DataFrames.

        This function returns the loaded CSV data stored in the all_data list.
        """
        return self.all_data
