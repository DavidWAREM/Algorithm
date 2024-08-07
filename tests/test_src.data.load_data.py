import os
import io
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import yaml
import logging
from src.data.data_load import CSVDataLoader


class TestCSVDataLoader(unittest.TestCase):

    def setUp(self):
        # Example configuration data
        self.config_data = {
            'paths': {
                'folder_path': '/path/to/folder'
            }
        }

        # Mock the configuration file content
        self.mock_config = mock_open(read_data=yaml.dump(self.config_data))

        # Example CSV data with all required columns
        self.csv_data = """
RORL;DM;RAU;FLUSS;VM;DPREL;RAISE;DP;ExtraColumn
1;2;3;4;5;6;7;8;9
10;11;12;13;14;15;16;17;18
"""
        # Create a DataFrame from the CSV data
        self.df = pd.read_csv(io.StringIO(self.csv_data), sep=';')

    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data='')
    @patch('pandas.read_csv')
    def test_load_all_data(self, mock_read_csv, mock_open, mock_listdir):
        # Mock os.listdir to return a list of files
        mock_listdir.return_value = ['file1_Pipes.csv', 'file2_Pipes.csv']

        # Mock open to read the configuration
        mock_open.side_effect = [self.mock_config.return_value]

        # Mock pandas.read_csv to return the DataFrame
        mock_read_csv.return_value = self.df

        # Initialize the CSVDataLoader
        loader = CSVDataLoader()

        # Check if the data is loaded correctly
        data = loader.get_data()
        self.assertEqual(len(data), 2)

        # Collect filenames from the loaded data for verification
        filenames = [file_name for file_name, _ in data]
        self.assertIn('file1_Pipes.csv', filenames)
        self.assertIn('file2_Pipes.csv', filenames)

        for file_name, df in data:
            self.assertEqual(df.shape, (2, 9))  # 2 rows and 9 columns in the mock CSV data
            # Verify the content of the DataFrame
            self.assertTrue(
                (df.columns == ['RORL', 'DM', 'RAU', 'FLUSS', 'VM', 'DPREL', 'RAISE', 'DP', 'ExtraColumn']).all())
            self.assertTrue((df.iloc[0] == [1, 2, 3, 4, 5, 6, 7, 8, 9]).all())
            self.assertTrue((df.iloc[1] == [10, 11, 12, 13, 14, 15, 16, 17, 18]).all())

    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data='')
    @patch('pandas.read_csv')
    def test_missing_required_columns(self, mock_read_csv, mock_open, mock_listdir):
        # Mock os.listdir to return a list of files
        mock_listdir.return_value = ['file1_Pipes.csv']

        # Mock open to read the configuration
        mock_open.side_effect = [self.mock_config.return_value]

        # Mock pandas.read_csv to return a DataFrame with missing required columns
        incomplete_csv_data = """
DM;RAU;FLUSS;VM;DPREL;RAISE;DP;ExtraColumn
1;2;3;4;5;6;7;8
"""
        # Create a DataFrame from the incomplete CSV data
        incomplete_df = pd.read_csv(io.StringIO(incomplete_csv_data), sep=';')
        mock_read_csv.return_value = incomplete_df

        # Initialize the CSVDataLoader and expect a ValueError
        with self.assertRaises(ValueError):
            CSVDataLoader()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Ensure logging is configured
    print("PYTHONPATH:", os.getenv('PYTHONPATH'))  # Ensure os is imported
    unittest.main()
