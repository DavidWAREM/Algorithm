import pandas as pd
import io


def find_max_columns(file_path):
    max_columns = 0
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            num_columns = len(line.split(';'))
            if num_columns > max_columns:
                max_columns = num_columns
    return max_columns


def clean_line(line, expected_columns):
    fields = line.split(';')
    if len(fields) < expected_columns:
        fields += [''] * (expected_columns - len(fields))
    return ';'.join(fields[:expected_columns])


def custom_read_csv(file_path):
    expected_columns = find_max_columns(file_path)

    cleaned_lines = []
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            cleaned_lines.append(clean_line(line.strip(), expected_columns))

    cleaned_content = "\n".join(cleaned_lines)

    # Angenommene Datentypen für die Spalten
    dtype_dict = {i: 'str' for i in range(expected_columns)}  # Beispielsweise alle Spalten als String

    return pd.read_csv(io.StringIO(cleaned_content), sep=';', encoding='ISO-8859-1', dtype=dtype_dict)


# Pfad zur CSV-Datei
file_path = r'C:\Users\d.muehlfeld\Berechnungsdaten\25_Schopfloch_NODE.CSV'

# Einlesen der CSV-Datei und Konvertierung in ein DataFrame
data = custom_read_csv(file_path)
print(data.head())
