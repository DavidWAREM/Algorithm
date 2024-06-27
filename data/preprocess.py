import pandas as pd
import os

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

    def split_data(self):
        """
        Split the DataFrame into two DataFrames based on the value in the first column.

        Returns:
            tuple: Two DataFrames, one for rows with 'KNO' and one for rows with 'LEI'.
        """
        kno_df = self.dataframe[self.dataframe.iloc[:, 0] == 'KNO']
        lei_df = self.dataframe[self.dataframe.iloc[:, 0] == 'LEI']
        return kno_df, lei_df

    def save_dataframes(self):
        """
        Save the two DataFrames to the same directory as the original file with modified names.
        """
        kno_df, lei_df = self.split_data()
        kno_path = os.path.join(self.directory, f"{self.base_filename}_Node.csv")
        lei_path = os.path.join(self.directory, f"{self.base_filename}_Pipes.csv")
        kno_df.to_csv(kno_path, index=False, sep=';')
        lei_df.to_csv(lei_path, index=False, sep=';')
