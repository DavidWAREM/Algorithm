import pandas as pd
import os
import logging


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
        self.dataframe = dataframe  # The DataFrame provided for processing.
        self.original_file_path = original_file_path  # The original file's path, used for saving files.
        self.directory = os.path.dirname(original_file_path)  # The directory where the file is located.
        self.base_filename = os.path.splitext(os.path.basename(original_file_path))[
            0]  # Extract the base filename without extension.
        logging.info(f"DataProcessor initialized with file path: {original_file_path}")

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
            kno_columns = ['REM', 'FLDNAM', 'KNO', 'KNAM', 'ZUFLUSS', 'FSTATUS', 'PMESS', 'DSTATUS', 'GEOH', 'PRECH',
                           'DP', 'XRECHTS', 'YHOCH', 'HP', 'SYMBOL', 'ABGAENGE', 'NETZNR']
            lei_columns = ['REM', 'FLDNAM', 'LEI', 'ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'DM', 'RAU', 'FLUSS',
                           'VM', 'DP', 'DPREL', 'ROHRTYP', 'RAISE']

            # Ensure that only relevant columns in kno_columns are extracted from kno_df.
            kno_columns_present = [col for col in kno_columns if col in kno_df.columns]
            kno_df = kno_df[kno_columns_present]

            # Ensure that only relevant columns in lei_columns are extracted from lei_df.
            lei_columns_present = [col for col in lei_columns if col in lei_df.columns]
            lei_df = lei_df[lei_columns_present]

            logging.info("Data split successfully into 'KNO' and 'LEI'")
            return kno_df, lei_df  # Return the split DataFrames.
        except Exception as e:
            logging.error(f"Error splitting data: {e}")  # Log the error if splitting fails.
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

                logging.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
            else:
                logging.error("DataFrames could not be saved due to an earlier error.")
        except Exception as e:
            logging.error(f"Error saving DataFrames: {e}")  # Log any errors encountered during saving.

    def combine_connected_pipes(self, kno_df, lei_df):
        """
        Log 'KNAM' values for rows where 'ABGAENGE' equals 2 in the 'kno_df' DataFrame,
        and log the indices of matching 'KNAM' values in the 'lei_df' DataFrame if 'ANFNAM' and 'ENDNAM' match
        and 'ROHRTYP' matches as well. Save matched pairs in a new DataFrame.

        Args:
            kno_df (pd.DataFrame): The DataFrame containing node data ('KNO').
            lei_df (pd.DataFrame): The DataFrame containing pipe data ('LEI').

        This method identifies matching pipes in the 'LEI' DataFrame based on node connections in the 'KNO' DataFrame.
        It groups pipes that share the same 'ANFNAM' or 'ENDNAM' and have the same 'ROHRTYP', assigning a 'GroupID' to them.
        """
        if 'ABGAENGE' in kno_df.columns and 'KNAM' in kno_df.columns:
            logging.debug("ABGAENGE and KNAM columns found in kno_df.")
            kno_df['ABGAENGE'] = pd.to_numeric(kno_df['ABGAENGE'], errors='coerce')  # Ensure 'ABGAENGE' is numeric.
            found = False  # Flag to check if any matches are found.

            # Add a new 'GroupID' column to 'lei_df' to store group information for connected pipes.
            lei_df['GroupID'] = ''

            # Initialize an empty DataFrame to store matched pairs of connected pipes.
            matched_pairs_df = pd.DataFrame(columns=['anf_idx', 'end_idx', 'ANFNAM', 'ENDNAM', 'ROHRTYP'])

            # Initialize a counter for assigning unique GroupIDs.
            n = 1

            # Iterate over each row in 'kno_df' where 'ABGAENGE' equals 2 (indicating a connection point).
            for idx, row in kno_df.iterrows():
                logging.debug(f"Checking row with ABGAENGE={row['ABGAENGE']} and KNAM={row['KNAM']} at index {idx}")
                if row['ABGAENGE'] == 2:
                    found = True  # Set the flag to true when a match is found.

                    # Find matching pipes in 'lei_df' with the same 'ANFNAM' or 'ENDNAM'.
                    matching_anf_df = lei_df[lei_df['ANFNAM'] == row['KNAM']]
                    matching_end_df = lei_df[lei_df['ENDNAM'] == row['KNAM']]

                    # Log and process matches from 'ANFNAM' and 'ENDNAM' columns.
                    for lei_idx in matching_anf_df.index:
                        rohrtyp = matching_anf_df.at[lei_idx, 'ROHRTYP']
                        logging.debug(
                            f"Matching KNAM value found in lei_df (ANFNAM) at index {lei_idx} with ROHRTYP={rohrtyp}")

                    for lei_idx in matching_end_df.index:
                        rohrtyp = matching_end_df.at[lei_idx, 'ROHRTYP']
                        logging.debug(
                            f"Matching KNAM value found in lei_df (ENDNAM) at index {lei_idx} with ROHRTYP={rohrtyp}")

                    # Further match pipes by comparing 'ANFNAM', 'ENDNAM', and 'ROHRTYP'.
                    for anf_idx in matching_anf_df.index:
                        for end_idx in matching_end_df.index:
                            if (matching_anf_df.at[anf_idx, 'ANFNAM'] == matching_end_df.at[end_idx, 'ENDNAM'] and
                                    matching_anf_df.at[anf_idx, 'ROHRTYP'] == matching_end_df.at[end_idx, 'ROHRTYP']):
                                logging.debug(f"Matching row found: ANFNAM={matching_anf_df.at[anf_idx, 'ANFNAM']}, "
                                              f"ENDNAM={matching_end_df.at[end_idx, 'ENDNAM']}, "
                                              f"ROHRTYP={matching_anf_df.at[anf_idx, 'ROHRTYP']}, "
                                              f"indices {anf_idx} and {end_idx}")

                                # Store the matched rows in a new row of the DataFrame.
                                new_row = {
                                    'anf_idx': anf_idx,
                                    'end_idx': end_idx,
                                    'ANFNAM_anf': matching_anf_df.at[anf_idx, 'ANFNAM'],
                                    'ENDNAM_anf': matching_anf_df.at[anf_idx, 'ENDNAM'],
                                    'ANFNAM_end': matching_end_df.at[end_idx, 'ANFNAM'],
                                    'ENDNAM_end': matching_end_df.at[end_idx, 'ENDNAM'],
                                    'ROHRTYP_anf': matching_anf_df.at[anf_idx, 'ROHRTYP'],
                                    'ROHRTYP_end': matching_end_df.at[end_idx, 'ROHRTYP']
                                }
                                matched_pairs_df = pd.concat([matched_pairs_df, pd.DataFrame([new_row])],
                                                             ignore_index=True)

                                # Assign or copy 'GroupID' between connected pipes.
                                if lei_df.at[anf_idx, 'GroupID'] == '' and lei_df.at[end_idx, 'GroupID'] == '':
                                    lei_df.at[anf_idx, 'GroupID'] = n
                                    lei_df.at[end_idx, 'GroupID'] = n
                                    n += 1  # Increment the group counter for the next group.
                                    logging.debug(f"Assigned GroupID {n - 1} to both indices {anf_idx} and {end_idx}")
                                elif lei_df.at[anf_idx, 'GroupID'] != '' and lei_df.at[end_idx, 'GroupID'] == '':
                                    lei_df.at[end_idx, 'GroupID'] = lei_df.at[anf_idx, 'GroupID']
                                    logging.debug(f"Copied GroupID from {anf_idx} to {end_idx}")
                                elif lei_df.at[anf_idx, 'GroupID'] == '' and lei_df.at[end_idx, 'GroupID'] != '':
                                    lei_df.at[anf_idx, 'GroupID'] = lei_df.at[end_idx, 'GroupID']
                                    logging.debug(f"Copied GroupID from {end_idx} to {anf_idx}")

            if not found:
                logging.info("No rows with ABGAENGE == 2 found.")  # Log if no matching rows were found.

            # Save the DataFrame with the matched pairs.
            matched_pairs_df_path = os.path.join(self.directory + '\\Zwischenspeicher',
                                                 f"{self.base_filename}_matched_pairs.csv")
            matched_pairs_df.to_csv(matched_pairs_df_path, index=False, sep=';')
            logging.info(f"Matched pairs DataFrame saved to {matched_pairs_df_path}")

            # Save the updated 'lei_df' with 'GroupID' column back to a CSV file.
            lei_path = os.path.join(self.directory + '\\Zwischenspeicher', f"{self.base_filename}_Pipes.csv")
            lei_df.to_csv(lei_path, index=False, sep=';')

            return lei_df  # Return the updated 'lei_df' with 'GroupID'.

        else:
            logging.warning(
                "'ABGAENGE' or 'KNAM' columns not found in the DataFrame.")  # Log warning if columns are missing.
