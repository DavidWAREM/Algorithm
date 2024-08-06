import unittest
from unittest.mock import mock_open, patch
import logging
import pandas as pd
import io
from data.rawdata_loader import DataLoader  # Adjust the import according to your file structure

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case environment.
        """
        self.file_path = 'test.csv'
        self.data_loader = DataLoader(self.file_path)
        logging.disable(logging.CRITICAL)  # Disable logging for testing

    def tearDown(self):
        """
        Clean up after each test case.
        """
        logging.disable(logging.NOTSET)  # Enable logging after testing

    def test_find_max_columns(self):
        """
        Test finding the maximum number of columns in a CSV file.
        """
        mock_data = 'col1;col2;col3\nval1;val2;val3\nval1;val2\n'
        with patch('builtins.open', mock_open(read_data=mock_data)):
            max_columns = self.data_loader.find_max_columns()
            self.assertEqual(max_columns, 3)

    def test_clean_line(self):
        """
        Test cleaning a line to ensure it has the correct number of columns.
        """
        line = 'val1;val2'
        expected_columns = 3
        cleaned_line = self.data_loader.clean_line(line, expected_columns)
        self.assertEqual(cleaned_line, 'val1;val2;')

    def test_custom_read_csv(self):
        """
        Test reading and cleaning a CSV file.
        """
        mock_data = 'col1;col2;col3\nval1;val2;val3\nval1;val2\n'
        expected_cleaned_data = 'col1;col2;col3\nval1;val2;val3\nval1;val2;\n'
        with patch('builtins.open', mock_open(read_data=mock_data)):
            with patch('os.makedirs'):
                df = self.data_loader.custom_read_csv()
                self.assertEqual(df.shape, (2, 3))
                pd.testing.assert_frame_equal(
                    df,
                    pd.read_csv(io.StringIO(expected_cleaned_data), sep=';', dtype=str)
                )

if __name__ == '__main__':
    unittest.main()
