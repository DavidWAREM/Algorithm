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
        """
        self.dataframe = dataframe
        self.original_file_path = original_file_path
        self.directory = os.path.dirname(original_file_path)
        self.base_filename = os.path.splitext(os.path.basename(original_file_path))[0]
        logging.info(f"DataProcessor initialized with file path: {original_file_path}")

    def split_data(self):
        """
        Split the DataFrame into two DataFrames based on the value in the first column.
        Use the row immediately above the first occurrence of 'KNO' and 'LEI' as headers.

        Returns:
            tuple: Two DataFrames, one for rows with 'KNO' and one for rows with 'LEI'.
        """
        try:
            # Find the indices where 'KNO' and 'LEI' first occur
            kno_index = self.dataframe.index[self.dataframe.iloc[:, 0] == 'KNO'][0]
            lei_index = self.dataframe.index[self.dataframe.iloc[:, 0] == 'LEI'][0]

            # Use the rows immediately before these indices as headers and create the DataFrames
            kno_header_index = kno_index - 1 if kno_index > 0 else kno_index
            kno_df = self.dataframe.iloc[kno_index + 1:].copy()
            kno_df.columns = self.dataframe.iloc[kno_header_index]

            lei_header_index = lei_index - 1 if lei_index > 0 else lei_index
            lei_df = self.dataframe.iloc[lei_index + 1:].copy()
            lei_df.columns = self.dataframe.iloc[lei_header_index]

            # Filter rows where the first column is 'KNO' or 'LEI'
            kno_df = kno_df[kno_df.iloc[:, 0] == 'KNO']
            lei_df = lei_df[lei_df.iloc[:, 0] == 'LEI']

            #Extraction of only relevant columns.
            kno_columns = ['REM', 'FLDNAM', 'KNO', 'KNAM', 'ZUFLUSS', 'FSTATUS', 'PMESS', 'DSTATUS', 'GEOH', 'PRECH',
                           'DP', 'XRECHTS', 'YHOCH', 'HP', 'SYMBOL', 'ABGAENGE', 'NETZNR']
            lei_columns = ['REM', 'FLDNAM', 'LEI', 'ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'DM', 'RAU', 'FLUSS',
                           'VM', 'DP', 'NETZNR', 'ZETA', 'STRASSE', 'PARALLEL', 'MATERIAL', 'STRANUM', 'BAUJAHR',
                           'DPREL', 'ROHRTYP']

            # Ensure that the columns in kno_columns are present in kno_df
            kno_columns_present = [col for col in kno_columns if col in kno_df.columns]

            # Extract the new DataFrame for kno_df
            kno_df = kno_df[kno_columns_present]

            # Ensure that the columns in lei_columns are present in lei_df
            lei_columns_present = [col for col in lei_columns if col in lei_df.columns]

            # Extract the new DataFrame for lei_df
            lei_df = lei_df[lei_columns_present]

            logging.info("Data split successfully into 'KNO' and 'LEI'")
            return kno_df, lei_df
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            return None, None

    def save_dataframes(self):
        """
        Save the two DataFrames to the same directory as the original file with modified names.
        """
        try:
            kno_df, lei_df = self.split_data()
            if kno_df is not None and lei_df is not None:
                kno_path = os.path.join(self.directory + '\\Zwischenspeicher', f"{self.base_filename}_Node.csv")
                lei_path = os.path.join(self.directory + '\\Zwischenspeicher', f"{self.base_filename}_Pipes.csv")
                kno_df.to_csv(kno_path, index=True, sep=';')
                lei_df.to_csv(lei_path, index=True, sep=';')
                logging.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
            else:
                logging.error("DataFrames could not be saved due to an earlier error.")
        except Exception as e:
            logging.error(f"Error saving DataFrames: {e}")

    def combine_connected_pipes(self, kno_df, lei_df):
        """
        Log the 'KNAM' values for rows where 'ABGAENGE' is '2' in the kno_df DataFrame,
        and log the indices of the same 'KNAM' values in the lei_df DataFrame,
        only if 'ANFNAM' and 'ENDNAM' match and 'ROHRTYP' also matches.
        Also, save the matched pairs in a new DataFrame.

        Args:
            kno_df (pd.DataFrame): The DataFrame to check.
            lei_df (pd.DataFrame): The DataFrame to search for matching 'KNAM' values.
        """
        if 'ABGAENGE' in kno_df.columns and 'KNAM' in kno_df.columns:
            logging.debug("ABGAENGE and KNAM columns found in kno_df.")
            kno_df['ABGAENGE'] = pd.to_numeric(kno_df['ABGAENGE'], errors='coerce')  # Convert to numeric
            found = False

            # Add additional column to the df kno_df for the GroupID
            lei_df['GroupID'] = ''

            # Initialize an empty DataFrame to store the matching pairs
            matched_pairs_df = pd.DataFrame(columns=['anf_idx', 'end_idx', 'ANFNAM', 'ENDNAM', 'ROHRTYP'])

            # Initialize the start value for the counter n
            n = 1
            for idx, row in kno_df.iterrows():
                logging.debug(f"Checking row with ABGAENGE={row['ABGAENGE']} and KNAM={row['KNAM']} at index {idx}")
                if row['ABGAENGE'] == 2:
                    found = True
                    # Check for the same 'KNAM' in ANFNAM and ENDNAM columns of lei_df
                    matching_anf_df = lei_df[lei_df['ANFNAM'] == row['KNAM']]
                    matching_end_df = lei_df[lei_df['ENDNAM'] == row['KNAM']]
                    for lei_idx in matching_anf_df.index:
                        rohrtyp = matching_anf_df.at[lei_idx, 'ROHRTYP']
                        logging.debug(
                            f"Matching KNAM value found in lei_df (ANFNAM) at index {lei_idx} with ROHRTYP={rohrtyp}")
                    for lei_idx in matching_end_df.index:
                        rohrtyp = matching_end_df.at[lei_idx, 'ROHRTYP']
                        logging.debug(
                            f"Matching KNAM value found in lei_df (ENDNAM) at index {lei_idx} with ROHRTYP={rohrtyp}")
                    # Additional loop to check for matching rows in both matching_anf_df and matching_end_df
                    for end_idx in matching_anf_df.index:
                        for check_idx in matching_end_df.index:
                            if (matching_anf_df.at[end_idx, 'ANFNAM'] == matching_end_df.at[check_idx, 'ENDNAM'] and
                                    matching_anf_df.at[end_idx, 'ROHRTYP'] == matching_end_df.at[check_idx, 'ROHRTYP']):
                                logging.debug(f"Matching row found: ANFNAM={matching_anf_df.at[end_idx, 'ANFNAM']}, "
                                              f"ENDNAM={matching_end_df.at[check_idx, 'ENDNAM']}, "
                                              f"ROHRTYP={matching_anf_df.at[end_idx, 'ROHRTYP']}, "
                                              f"indices {end_idx} and {check_idx}")
                                # Adding the matched pairs to the DataFrame
                                new_row = {
                                    'anf_idx': end_idx,
                                    'end_idx': check_idx,
                                    'ANFNAM_anf': matching_anf_df.at[end_idx, 'ANFNAM'],
                                    'ENDNAM_anf': matching_anf_df.at[end_idx, 'ENDNAM'],
                                    'ANFNAM_end': matching_end_df.at[check_idx, 'ANFNAM'],
                                    'ENDNAM_end': matching_end_df.at[check_idx, 'ENDNAM'],
                                    'ROHRTYP_anf': matching_anf_df.at[end_idx, 'ROHRTYP'],
                                    'ROHRTYP_end': matching_end_df.at[check_idx, 'ROHRTYP']
                                }
                                matched_pairs_df = pd.concat([matched_pairs_df, pd.DataFrame([new_row])],
                                                             ignore_index=True)

                                if lei_df.at[end_idx, 'GroupID'] == '' and lei_df.at[check_idx, 'GroupID'] == '':
                                    # If both are empty, assign the counter value n
                                    lei_df.at[end_idx, 'GroupID'] = n
                                    lei_df.at[check_idx, 'GroupID'] = n
                                    n += 1  # Increment the counter for the next group
                                    logging.debug(f"Assigned GroupID {n - 1} to both indices {end_idx} and {check_idx}")
                                elif lei_df.at[end_idx, 'GroupID'] != '' and lei_df.at[check_idx, 'GroupID'] == '':
                                    # If only end_idx has a value, copy it to check_idx
                                    lei_df.at[check_idx, 'GroupID'] = lei_df.at[end_idx, 'GroupID']
                                    logging.debug(f"Copied GroupID from {end_idx} to {check_idx}")
                                elif lei_df.at[end_idx, 'GroupID'] == '' and lei_df.at[check_idx, 'GroupID'] != '':
                                    # If only check_idx has a value, copy it to end_idx
                                    lei_df.at[end_idx, 'GroupID'] = lei_df.at[check_idx, 'GroupID']
                                    logging.debug(f"Copied GroupID from {check_idx} to {end_idx}")
                                # If both have values, do nothing

                    # Additional loop to check for matching pipes with same ANFNAM and ROHRTYP but different indices
                    for end_idx in matching_anf_df.index:
                        for check_idx in matching_anf_df.index:
                            if (end_idx != check_idx and
                                    matching_anf_df.at[end_idx, 'ANFNAM'] == matching_anf_df.at[check_idx, 'ANFNAM'] and
                                    matching_anf_df.at[end_idx, 'ROHRTYP'] == matching_anf_df.at[check_idx, 'ROHRTYP']):
                                logging.debug(f"Matching row found with same ANFNAM and ROHRTYP but different indices: "
                                              f"ANFNAM={matching_anf_df.at[end_idx, 'ANFNAM']}, "
                                              f"ENDNAM={matching_anf_df.at[check_idx, 'ENDNAM']}, "
                                              f"ROHRTYP={matching_anf_df.at[end_idx, 'ROHRTYP']}, "
                                              f"indices {end_idx} and {check_idx}")
                                # Adding the matched pairs to the DataFrame
                                new_row = {
                                    'anf_idx': end_idx,
                                    'end_idx': check_idx,
                                    'ANFNAM_anf': matching_anf_df.at[end_idx, 'ANFNAM'],
                                    'ENDNAM_anf': matching_anf_df.at[end_idx, 'ENDNAM'],
                                    'ANFNAM_end': matching_anf_df.at[check_idx, 'ANFNAM'],
                                    'ENDNAM_end': matching_anf_df.at[check_idx, 'ENDNAM'],
                                    'ROHRTYP_anf': matching_anf_df.at[end_idx, 'ROHRTYP'],
                                    'ROHRTYP_end': matching_anf_df.at[check_idx, 'ROHRTYP']
                                }
                                matched_pairs_df = pd.concat([matched_pairs_df, pd.DataFrame([new_row])],
                                                             ignore_index=True)

                                if lei_df.at[end_idx, 'GroupID'] == '' and lei_df.at[check_idx, 'GroupID'] == '':
                                    # If both are empty, assign the counter value n
                                    lei_df.at[end_idx, 'GroupID'] = n
                                    lei_df.at[check_idx, 'GroupID'] = n
                                    n += 1  # Increment the counter for the next group
                                    logging.debug(f"Assigned GroupID {n - 1} to both indices {end_idx} and {check_idx}")
                                elif lei_df.at[end_idx, 'GroupID'] != '' and lei_df.at[check_idx, 'GroupID'] == '':
                                    # If only end_idx has a value, copy it to check_idx
                                    lei_df.at[check_idx, 'GroupID'] = lei_df.at[end_idx, 'GroupID']
                                    logging.debug(f"Copied GroupID from {end_idx} to {check_idx}")
                                elif lei_df.at[end_idx, 'GroupID'] == '' and lei_df.at[check_idx, 'GroupID'] != '':
                                    # If only check_idx has a value, copy it to end_idx
                                    lei_df.at[end_idx, 'GroupID'] = lei_df.at[check_idx, 'GroupID']
                                    logging.debug(f"Copied GroupID from {check_idx} to {end_idx}")
                                # If both have values, do nothing

            if not found:
                logging.info("No rows with ABGAENGE == 2 found.")

            # Save the DataFrame with the matched pairs
            matched_pairs_df_path = os.path.join(self.directory + '\\Zwischenspeicher',
                                                 f"{self.base_filename}_matched_pairs.csv")
            matched_pairs_df.to_csv(matched_pairs_df_path, index=False, sep=';')
            logging.info(f"Matched pairs DataFrame saved to {matched_pairs_df_path}")

            # "Re-exporting kno_df as Excel to include the new GroupID column."
            lei_path = os.path.join(self.directory + '\\Zwischenspeicher', f"{self.base_filename}_Pipes.csv")
            lei_df.to_csv(lei_path, index=True, sep=';')

            return lei_df


        else:
            logging.warning("'ABGAENGE' or 'KNAM' columns not found in the DataFrame.")


