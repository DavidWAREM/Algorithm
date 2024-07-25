import pandas as pd
import io
import logging
import os

class DataLoader:
    def __init__(self, file_path):
        """
        Initialize the DataLoader with the path to the CSV file.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path
        logging.info(f"DataLoader initialized with file path: {file_path}")

    """The functions `def find_max_columns`, `def clean_lines`, and `def custom_read_csv` were introduced because the
        raw data from STANET in CSV format caused errors when being read into Python. Issues included varying numbers
        of columns per row and ";" at the end of the lines. Therefore, these issues are cleaned before the data
        is further processed."""

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
            # Determine the expected number of columns
            expected_columns = self.find_max_columns()
            cleaned_lines = []

            # Read and clean the lines
            with open(self.file_path, 'r', encoding='ISO-8859-1') as file:
                for line in file:
                    cleaned_lines.append(self.clean_line(line.strip(), expected_columns))

            # Join the cleaned lines into a single string
            cleaned_content = "\n".join(cleaned_lines)

            # Define data types for each column
            dtype_dict = {i: 'str' for i in range(expected_columns)}

            # Log success message
            logging.info("CSV file read and cleaned successfully.")

            # Create DataFrame from cleaned content
            df = pd.read_csv(io.StringIO(cleaned_content), sep=';', encoding='ISO-8859-1', dtype=dtype_dict)

            # Determine the directory of the original file and the new directory for cleaned files
            original_dir = os.path.dirname(self.file_path)
            cleaned_dir = os.path.join(original_dir, 'Zwischenspeicher')

            # Create the new directory if it doesn't exist
            os.makedirs(cleaned_dir, exist_ok=True)

            # Generate the path for the cleaned file
            cleaned_file_path = os.path.join(cleaned_dir, os.path.basename(self.file_path).replace('.csv', '_cleaned.csv'))

            # Export the cleaned lines to the new CSV file
            with open(cleaned_file_path, 'w', encoding='ISO-8859-1') as cleaned_file:
                cleaned_file.write(cleaned_content)

            logging.info(f"Cleaned CSV file saved to {cleaned_file_path}.")

            return df
        except Exception as e:
            # Log error message
            logging.error(f"Error reading and cleaning CSV file: {e}")

            # Return an empty DataFrame in case of error
            return pd.DataFrame()
