import os
import re
import pandas as pd
import logging


class DataCombiner:
    def __init__(self, directory):
        """
        Initializes the DataCombiner with the directory where the 'Zwischenspeicher' folder is located.

        Args:
            directory (str): The base directory path containing the 'Zwischenspeicher' folder.
        """
        self.logger = logging.getLogger(__name__)
        self.directory = directory
        self.logger.info(f"DataCombiner initialized with directory: {directory}")

    def combine_with_without_load(self, file_type):
        """
        Combines data from 'with_load' and 'without_load' CSV files based on matching numbers in filenames.
        This function processes either 'Pipes' or 'Node' type files, merges the relevant columns, and saves
        the result into a new CSV file, excluding the '_with' part from the file name.

        Args:
            file_type (str): The type of file to combine, either 'Pipes' or 'Node'.
        """
        try:
            # Construct the path to the 'Zwischenspeicher' directory
            zwischenspeicher_dir = os.path.join(self.directory, 'Zwischenspeicher')
            self.logger.info(f"Looking for files in directory: {zwischenspeicher_dir}")

            if not os.path.exists(zwischenspeicher_dir):
                self.logger.error(f"'Zwischenspeicher' directory not found at: {zwischenspeicher_dir}")
                return

            # Initialize dictionaries to store with_load and without_load files
            with_load_files = {}
            without_load_files = {}

            # Loop through files in the 'Zwischenspeicher' directory and match based on file_type
            for file in os.listdir(zwischenspeicher_dir):
                with_load_match = re.match(rf"(.+)_with_load_(\d+)_({file_type})\.csv", file)
                without_load_match = re.match(rf"(.+)_without_load_(\d+)_({file_type})\.csv", file)

                if with_load_match:
                    number = with_load_match.group(2)
                    with_load_files[number] = os.path.join(zwischenspeicher_dir, file)
                    self.logger.debug(f"Found with_load file: {file} with number: {number}")
                elif without_load_match:
                    number = without_load_match.group(2)
                    without_load_files[number] = os.path.join(zwischenspeicher_dir, file)
                    self.logger.debug(f"Found without_load file: {file} with number: {number}")

            # Process matching pairs of with_load and without_load files
            for number in with_load_files.keys():
                if number in without_load_files:
                    wl_file = with_load_files[number]
                    wol_file = without_load_files[number]
                    self.logger.info(f"Combining with_load file: {wl_file} and without_load file: {wol_file}")

                    # Read both CSV files into dataframes
                    df_wl = pd.read_csv(wl_file, sep=';')
                    df_wol = pd.read_csv(wol_file, sep=';')

                    if file_type == 'Pipes':
                        # Columns for 'Pipes' files
                        key_columns = ['ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'ROHRTYP', 'RAISE']
                        self.logger.debug(f"Key columns for 'Pipes': {key_columns}")

                        # Create a new dataframe with the key columns from 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Add VM and FLUSS columns from both with_load and without_load
                        combined_df['VM_WL'] = df_wl['VM']
                        combined_df['FLUSS_WL'] = df_wl['FLUSS']
                        combined_df['VM_WOL'] = df_wol['VM']
                        combined_df['FLUSS_WOL'] = df_wol['FLUSS']

                    elif file_type == 'Node':
                        # Columns for 'Node' files
                        key_columns = ['KNAM', 'GEOH', 'XRECHTS', 'YHOCH']
                        self.logger.debug(f"Key columns for 'Node': {key_columns}")

                        # Create a new dataframe with the key columns from 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Add PRECH, HP, and ZUFLUSS columns from both with_load and without_load
                        combined_df['PRECH_WL'] = df_wl['PRECH']
                        combined_df['HP_WL'] = df_wl['HP']
                        combined_df['ZUFLUSS_WL'] = df_wl['ZUFLUSS']
                        combined_df['PRECH_WOL'] = df_wol['PRECH']
                        combined_df['HP_WOL'] = df_wol['HP']
                        combined_df['ZUFLUSS_WOL'] = df_wol['ZUFLUSS']

                        # Add dp column for Node files as PRECH_WOL - PRECH_WL
                        combined_df['dp'] = combined_df['PRECH_WOL'] - combined_df['PRECH_WL']
                        self.logger.debug(f"Added dp column as PRECH_WOL - PRECH_WL for Node")

                    # Extract the base file name without "_with" and define output file name
                    base_name = re.sub(rf'_with_load_\d+_({file_type})\.csv', '', os.path.basename(wl_file))
                    output_file = os.path.join(zwischenspeicher_dir, f"{base_name}_{number}_combined_{file_type}.csv")

                    # Save the combined dataframe to a new CSV file
                    combined_df.to_csv(output_file, index=False, sep=';')
                    self.logger.info(f"Combined CSV file saved successfully to: {output_file}")

        except Exception as e:
            self.logger.error(f"Error combining files: {e}", exc_info=True)


# Example usage of the class
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    test = DataCombiner('C:\\Users\\D.Muehlfeld\\Documents\\Berechnungsdaten')
    # test.combine_with_without_load('Pipes')  # For Pipes files
    test.combine_with_without_load('Node')  # For Node files
