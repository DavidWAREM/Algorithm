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
        Also includes the row immediately above the first occurrence of 'KNO' in the 'KNO' DataFrame.

        Returns:
            tuple: Two DataFrames, one for rows with 'KNO' and one for rows with 'LEI'.
        """
        try:
            kno_indices = self.dataframe.index[self.dataframe.iloc[:, 0] == 'KNO'].tolist()
            if kno_indices:
                first_kno_index = kno_indices[0]
                header_index = first_kno_index - 1 if first_kno_index > 0 else first_kno_index
                kno_df = self.dataframe.loc[header_index:].copy()
                kno_df = kno_df[kno_df.iloc[:, 0].str.contains('KNO', na=False) | (kno_df.index == header_index)]
            else:
                kno_df = pd.DataFrame()

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
