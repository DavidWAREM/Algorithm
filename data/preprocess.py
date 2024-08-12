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
        kno_df = self.dataframe.iloc[kno_index:].copy()
        kno_df.columns = self.dataframe.iloc[kno_header_index]

        lei_header_index = lei_index - 1 if lei_index > 0 else lei_index
        lei_df = self.dataframe.iloc[lei_index:].copy()
        lei_df.columns = self.dataframe.iloc[lei_header_index]

        # Filter rows where the first column is 'KNO' or 'LEI'
        kno_df = kno_df[kno_df.iloc[:, 0] == 'KNO']
        lei_df = lei_df[lei_df.iloc[:, 0] == 'LEI']

        # Extraction of only relevant columns.
        kno_columns = ['REM', 'FLDNAM', 'KNO', 'KNAM', 'ZUFLUSS', 'FSTATUS', 'PMESS', 'DSTATUS', 'GEOH', 'PRECH',
                       'DP', 'XRECHTS', 'YHOCH', 'HP', 'SYMBOL', 'ABGAENGE', 'NETZNR']
        lei_columns = ['REM', 'FLDNAM', 'LEI', 'ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'DM', 'RAU', 'FLUSS',
                       'VM', 'DP', 'DPREL', 'ROHRTYP', 'RAISE']

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
            kno_df.to_csv(kno_path, index=False, sep=';')
            lei_df.to_csv(lei_path, index=False, sep=';')
            logging.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
        else:
            logging.error("DataFrames could not be saved due to an earlier error.")
    except Exception as e:
        logging.error(f"Error saving DataFrames: {e}")
