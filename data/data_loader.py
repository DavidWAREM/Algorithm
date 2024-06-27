import pandas as pd
import io
import logging

class DataLoader:
    def __init__(self, file_path):
        """
        Initialize the DataLoader with the path to the CSV file.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path
        logging.info(f"DataLoader initialized with file path: {file_path}")

    def find_max_columns(self):
        """
        Find the maximum number of columns in the CSV file.

        Returns:
            int: Maximum number of columns.
        """
        max_columns = 0
        try:
            with open(self.file_path, 'r', encoding='ISO-8859-1') as file:
                for line in file:
                    num_columns = len(line.split(';'))
                    if num_columns > max_columns:
                        max_columns = num_columns
            logging.info(f"Maximum number of columns found: {max_columns}")
        except Exception as e:
            logging.error(f"Error finding max columns: {e}")
        return max_columns

    def clean_line(self, line, expected_columns):
        """
        Ensure a line has the expected number of columns by adding empty fields if necessary.

        Args:
            line (str): The line to clean.
            expected_columns (int): The expected number of columns.

        Returns:
            str: The cleaned line with the correct number of columns.
        """
        fields = line.split(';')
        if len(fields) < expected_columns:
            fields += [''] * (expected_columns - len(fields))
        cleaned_line = ';'.join(fields[:expected_columns])
        logging.debug(f"Cleaned line: {cleaned_line}")
        return cleaned_line

    def custom_read_csv(self):
        """
        Read the CSV file, clean it to ensure all rows have the same number of columns,
        and return it as a pandas DataFrame.

        Returns:
            DataFrame: The CSV data as a pandas DataFrame.
        """
        try:
            expected_columns = self.find_max_columns()
            cleaned_lines = []
            with open(self.file_path, 'r', encoding='ISO-8859-1') as file:
                for line in file:
                    cleaned_lines.append(self.clean_line(line.strip(), expected_columns))
            cleaned_content = "\n".join(cleaned_lines)
            dtype_dict = {i: 'str' for i in range(expected_columns)}
            logging.info("CSV file read and cleaned successfully.")
            return pd.read_csv(io.StringIO(cleaned_content), sep=';', encoding='ISO-8859-1', dtype=dtype_dict)
        except Exception as e:
            logging.error(f"Error reading and cleaning CSV file: {e}")
            return pd.DataFrame()


