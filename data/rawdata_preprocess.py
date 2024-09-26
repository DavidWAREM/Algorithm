import pandas as pd
import os
import logging
import re


class DataProcessor:
    def __init__(self, dataframe, original_file_path):
        """
        Initialize the DataProcessor with a pandas DataFrame and the path to the original file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to process.
            original_file_path (str): The path to the original CSV file.

        The constructor sets up the necessary attributes such as the DataFrame to process and the file path details.
        It also extracts the base filename and directory to use for saving output files.
        """
        self.logger = logging.getLogger(__name__)  # Initialize logger
        self.dataframe = dataframe  # The DataFrame provided for processing.
        self.original_file_path = original_file_path  # The original file's path, used for saving files.
        self.directory = os.path.dirname(original_file_path)  # The directory where the file is located.
        self.base_filename = os.path.splitext(os.path.basename(original_file_path))[0]  # Extract the base filename without extension.
        self.logger.info(f"DataProcessor initialized with file path: {original_file_path}")

    def split_data(self):
        """
        Split the DataFrame into two DataFrames based on the value in the first column ('KNO' and 'LEI').
        The row immediately above the first occurrence of 'KNO' and 'LEI' is used as headers.

        Returns:
            tuple: Two DataFrames, one for rows with 'KNO' and one for rows with 'LEI'.

        This method identifies the rows corresponding to 'KNO' and 'LEI', splits the DataFrame accordingly,
        and assigns the header row before splitting. It also filters relevant columns for both 'KNO' and 'LEI' DataFrames.
        """
        try:
            # Find the first occurrence of 'KNO' and 'LEI' in the DataFrame.
            kno_index = self.dataframe.index[self.dataframe.iloc[:, 0] == 'KNO'][0]
            lei_index = self.dataframe.index[self.dataframe.iloc[:, 0] == 'LEI'][0]

            # Define header rows (one row above 'KNO' and 'LEI' for column names).
            kno_header_index = kno_index - 1 if kno_index > 0 else kno_index
            kno_df = self.dataframe.iloc[kno_index:].copy()  # Copy the data starting from 'KNO'.
            kno_df.columns = self.dataframe.iloc[kno_header_index]  # Set header row as column names.

            lei_header_index = lei_index - 1 if lei_index > 0 else lei_index
            lei_df = self.dataframe.iloc[lei_index:].copy()  # Copy the data starting from 'LEI'.
            lei_df.columns = self.dataframe.iloc[lei_header_index]  # Set header row as column names.

            # Filter rows with 'KNO' and 'LEI' in the first column.
            kno_df = kno_df[kno_df.iloc[:, 0] == 'KNO']
            lei_df = lei_df[lei_df.iloc[:, 0] == 'LEI']

            # Define relevant columns for each DataFrame.
            kno_columns = ['REM', 'FLDNAM', 'KNO', 'KNAM', 'ZUFLUSS', 'GEOH', 'PRECH',
                             'XRECHTS', 'YHOCH', 'HP']
            lei_columns = ['REM', 'FLDNAM', 'LEI', 'ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'DM', 'RAU', 'FLUSS',
                           'VM', 'DPREL', 'ROHRTYP', 'RAISE']

            # Ensure that only relevant columns in kno_columns are extracted from kno_df.
            kno_columns_present = [col for col in kno_columns if col in kno_df.columns]
            kno_df = kno_df[kno_columns_present]

            # Ensure that only relevant columns in lei_columns are extracted from lei_df.
            lei_columns_present = [col for col in lei_columns if col in lei_df.columns]
            lei_df = lei_df[lei_columns_present]

            self.logger.info("Data split successfully into 'KNO' and 'LEI'")
            return kno_df, lei_df  # Return the split DataFrames.
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")  # Log the error if splitting fails.
            return None, None  # Return None for both DataFrames in case of error.



    def save_dataframes(self):
        """
        Save the two DataFrames (split by 'KNO' and 'LEI') into separate CSV files in the same directory as the original file.
        """
        try:
            # Split the data into 'KNO' and 'LEI' DataFrames.
            kno_df, lei_df = self.split_data()
            if kno_df is not None and lei_df is not None:
                # Define file paths for saving the DataFrames.
                kno_path = os.path.join(self.directory + '\\Zwischenspeicher', f"{self.base_filename}_Node.csv")
                lei_path = os.path.join(self.directory + '\\Zwischenspeicher', f"{self.base_filename}_Pipes.csv")

                # Save 'KNO' DataFrame as CSV.
                kno_df.to_csv(kno_path, index=False, sep=';')
                # Save 'LEI' DataFrame as CSV.
                lei_df.to_csv(lei_path, index=False, sep=';')

                self.logger.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
            else:
                self.logger.error("DataFrames could not be saved due to an earlier error.")
        except Exception as e:
            self.logger.error(f"Error saving DataFrames: {e}")  # Log any errors encountered during saving.


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





