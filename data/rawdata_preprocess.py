import pandas as pd
import os
import logging
import re
import numpy as np  # Für numerische Berechnungen

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
            kno_columns = ['KNAM', 'ZUFLUSS', 'GEOH', 'PRECH',
                           'XRECHTS', 'YHOCH', 'HP']
            lei_columns = ['ANFNAM', 'ENDNAM', 'RORL', 'DM', 'RAU', 'FLUSS',
                           'VM', 'DPREL', 'ROHRTYP', 'RAISE', 'RE']

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
                # Define file paths für das Speichern der DataFrames.
                zwischenspeicher_dir = os.path.join(self.directory, 'Zwischenspeicher')
                if not os.path.exists(zwischenspeicher_dir):
                    os.makedirs(zwischenspeicher_dir)
                    self.logger.debug(f"Created 'Zwischenspeicher' directory: {zwischenspeicher_dir}")

                kno_path = os.path.join(zwischenspeicher_dir, f"{self.base_filename}_Node.csv")
                lei_path = os.path.join(zwischenspeicher_dir, f"{self.base_filename}_Pipes.csv")

                # Speichern des 'KNO' DataFrames als CSV.
                kno_df.to_csv(kno_path, index=False, sep=';')
                # Speichern des 'LEI' DataFrames als CSV.
                lei_df.to_csv(lei_path, index=False, sep=';')

                self.logger.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
            else:
                self.logger.error("DataFrames could not be saved due to an earlier error.")
        except Exception as e:
            self.logger.error(f"Error saving DataFrames: {e}")  # Log any errors encountered during saving.



class DataCombiner:
    def __init__(self, directory):
        """
        Initializes the DataCombiner with the directory containing the 'Zwischenspeicher' folder.

        Args:
            directory (str): The base directory path containing the 'Zwischenspeicher' folder.
        """
        self.logger = logging.getLogger(__name__)
        self.directory = directory
        self.logger.info(f"DataCombiner initialized with directory: {directory}")

    def combine_with_without_load(self, file_type):
        """
        Combines data from 'with_load' and 'without_load' CSV files based on matching numbers in filenames.
        This function processes either 'Pipes' or 'Node' type files, combines relevant columns, and saves
        the result in a new CSV file with the '_with' part removed from the filename.

        Args:
            file_type (str): The type of files to combine, either 'Pipes' or 'Node'.
        """
        try:

            # Path to the 'Zwischenspeicher' directory
            zwischenspeicher_dir = os.path.join(self.directory, 'Zwischenspeicher')
            self.logger.info(f"Searching for files in directory: {zwischenspeicher_dir}")

            if not os.path.exists(zwischenspeicher_dir):
                self.logger.error(f"'Zwischenspeicher' directory not found at: {zwischenspeicher_dir}")
                return

            # Initialize dictionaries to store with_load and without_load files
            with_load_files = {}
            without_load_files = {}

            # Iterate through files in the 'Zwischenspeicher' directory and categorize them
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

                    # Read both CSV files into DataFrames
                    df_wl = pd.read_csv(wl_file, sep=';', decimal='.', encoding='utf-8')
                    df_wol = pd.read_csv(wol_file, sep=';', decimal='.', encoding='utf-8')

                    if file_type == 'Pipes':
                        # Columns for 'Pipes' files (Excluding 'RE', 'FLUSS', 'VM')
                        key_columns = ['ANFNAM', 'ENDNAM', 'RORL', 'DM', 'RAU', 'DPREL', 'ROHRTYP', 'RAISE']
                        self.logger.debug(f"Key columns for 'Pipes' (excluding 'RE', 'FLUSS', 'VM'): {key_columns}")

                        # Create a new DataFrame with key columns from 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Add new columns from both with_load and without_load
                        combined_df['VM_WL'] = pd.to_numeric(df_wl['VM'], errors='coerce')
                        combined_df['FLUSS_WL'] = pd.to_numeric(df_wl['FLUSS'], errors='coerce')
                        combined_df['VM_WOL'] = pd.to_numeric(df_wol['VM'], errors='coerce')
                        combined_df['FLUSS_WOL'] = pd.to_numeric(df_wol['FLUSS'], errors='coerce')
                        combined_df['RE_WL'] = pd.to_numeric(df_wl['RE'], errors='coerce')
                        combined_df['RE_WOL'] = pd.to_numeric(df_wol['RE'], errors='coerce')

                        # Add RAU column if present
                        if 'RAU' in df_wl.columns:
                            combined_df['RAU'] = pd.to_numeric(df_wl['RAU'], errors='coerce')
                            self.logger.info(f"Added RAU column for file: {wl_file}")
                        else:
                            combined_df['RAU'] = pd.NA
                            self.logger.warning(f"RAU column missing in file: {wl_file}")

                        # Convert specified columns to numeric types
                        numeric_columns = ['DM', 'VM_WL', 'VM_WOL', 'FLUSS_WL', 'FLUSS_WOL', 'RAU', 'RORL', 'RE_WL', 'RE_WOL']
                        for col in numeric_columns:
                            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

                        # **Exclude Original Columns ('RE', 'FLUSS', 'VM')**
                        # Since 'RE', 'FLUSS', 'VM' were excluded from key_columns, they are not present in combined_df
                        # Only the new '_WL' and '_WOL' columns are included along with key columns

                        # Optionally, drop any remaining unwanted columns if necessary
                        # Not needed here as 'RE', 'FLUSS', 'VM' are already excluded

                    elif file_type == 'Node':
                        # Columns for 'Node' files
                        key_columns = ['KNAM', 'GEOH', 'XRECHTS', 'YHOCH']
                        self.logger.debug(f"Key columns for 'Node': {key_columns}")

                        # Create a new DataFrame with key columns from 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Add PRECH, HP, and ZUFLUSS columns from both with_load and without_load
                        combined_df['PRECH_WL'] = pd.to_numeric(df_wl['PRECH'], errors='coerce')
                        combined_df['HP_WL'] = pd.to_numeric(df_wl['HP'], errors='coerce')
                        combined_df['ZUFLUSS_WL'] = pd.to_numeric(df_wl['ZUFLUSS'], errors='coerce')
                        combined_df['PRECH_WOL'] = pd.to_numeric(df_wol['PRECH'], errors='coerce')
                        combined_df['HP_WOL'] = pd.to_numeric(df_wol['HP'], errors='coerce')
                        combined_df['ZUFLUSS_WOL'] = pd.to_numeric(df_wol['ZUFLUSS'], errors='coerce')

                        # Add dp column as PRECH_WOL - PRECH_WL
                        combined_df['dp'] = combined_df['PRECH_WOL'] - combined_df['PRECH_WL']
                        self.logger.debug("Calculated 'dp' as PRECH_WOL - PRECH_WL for Node")

                        # Add delta_H column as HP_WL - HP_WOL
                        combined_df['delta_H'] = combined_df['HP_WL'] - combined_df['HP_WOL']

                        # Ensure no infinite or NaN values in the calculations
                        node_calc_columns = ['PRECH_WL', 'HP_WL', 'ZUFLUSS_WL', 'PRECH_WOL',
                                             'HP_WOL', 'ZUFLUSS_WOL', 'dp', 'delta_H']
                        for col in node_calc_columns:
                            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], pd.NA).fillna(0)

                    # Extract the base filename without "_with" and define the output filename
                    base_name = re.sub(rf'_with_load_\d+_({file_type})\.csv', '', os.path.basename(wl_file))
                    output_file = os.path.join(zwischenspeicher_dir, f"{base_name}_{number}_combined_{file_type}.csv")

                    # Save the combined DataFrame to a new CSV file
                    combined_df.to_csv(output_file, index=False, sep=';', decimal='.')
                    self.logger.info(f"Combined CSV file saved successfully at: {output_file}")

        except Exception as e:
            self.logger.error(f"Error combining files: {e}", exc_info=True)




