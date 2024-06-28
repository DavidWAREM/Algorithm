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

        Returns:
            tuple: Two DataFrames, one for rows with 'KNO' and one for rows with 'LEI'.
        """
        try:
            # Filter rows where the first column is 'KNO' or 'LEI'
            kno_df = self.dataframe[self.dataframe.iloc[:, 0] == 'KNO']
            lei_df = self.dataframe[self.dataframe.iloc[:, 0] == 'LEI']

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
                kno_path = os.path.join(self.directory, f"{self.base_filename}_Node.csv")
                lei_path = os.path.join(self.directory, f"{self.base_filename}_Pipes.csv")
                kno_df.to_csv(kno_path, index=False, sep=';')
                lei_df.to_csv(lei_path, index=False, sep=';')
                logging.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
            else:
                logging.error("DataFrames could not be saved due to an earlier error.")
        except Exception as e:
            logging.error(f"Error saving DataFrames: {e}")

    def log_knam_for_abgaenge_2(self, kno_df):
        """
        Log the 'KNAM' values for rows where 'ABGAENGE' is '2' in the kno_df DataFrame.

        Args:
            kno_df (pd.DataFrame): The DataFrame to check.
        """
        if 'ABGAENGE' in kno_df.columns and 'KNAM' in kno_df.columns:
            logging.debug("ABGAENGE and KNAM columns found.")
            found = False
            for _, row in kno_df.iterrows():
                logging.debug(f"Checking row with ABGAENGE={row['ABGAENGE']} and KNAM={row['KNAM']}")
                if row['ABGAENGE'] == 2:  # Ensure this is not a string comparison
                    logging.info(f"KNAM value with ABGAENGE == 2: {row['KNAM']}")
                    found = True
            if not found:
                logging.info("No rows with ABGAENGE == 2 found.")
        else:
            logging.warning("'ABGAENGE' or 'KNAM' columns not found in the DataFrame.")
