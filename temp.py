import pandas as pd
import io


def find_max_columns(file_path):
    """
    Find the maximum number of columns in a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        int: Maximum number of columns.
    """
    max_columns = 0
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            num_columns = len(line.split(';'))
            if num_columns > max_columns:
                max_columns = num_columns
    return max_columns


def clean_line(line, expected_columns):
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
    return ';'.join(fields[:expected_columns])


def custom_read_csv(file_path):
    """
    Read a CSV file, clean it to ensure all rows have the same number of columns,
    and return it as a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: The CSV data as a pandas DataFrame.
    """
    expected_columns = find_max_columns(file_path)

    cleaned_lines = []
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            cleaned_lines.append(clean_line(line.strip(), expected_columns)) #Uses the clean_line function to clean each
                        # line, ensuring it has the expected number of columns, and adds it to the cleaned_lines list.

    cleaned_content = "\n".join(cleaned_lines)

    # Assume all columns are strings to avoid dtype warnings
    dtype_dict = {i: 'str' for i in range(expected_columns)}

    return pd.read_csv(io.StringIO(cleaned_content), sep=';', encoding='ISO-8859-1', dtype=dtype_dict)


# Path to the CSV file
file_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\25_Schopfloch_NODE.CSV'

# Read the CSV file and convert it to a DataFrame
data = custom_read_csv(file_path)
print(data.head())
