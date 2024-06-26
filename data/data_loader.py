import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='ISO-8859-1') as file:
                lines = file.readlines()

            cleaned_lines = []
            for line in lines:
                if line.strip().endswith(';'):
                    cleaned_lines.append(line.strip()[:-1])  # Entfernt das letzte Semikolon
                else:
                    cleaned_lines.append(line.strip())

            cleaned_content = "\n".join(cleaned_lines)

            # Speichern der bereinigten Daten in einer temporären Datei
            temp_file_path = 'cleaned_file.csv'
            with open(temp_file_path, 'w', encoding='ISO-8859-1') as cleaned_file:
                cleaned_file.write(cleaned_content)

"""
            # Lesen der bereinigten Datei
            data = pd.read_csv(temp_file_path, sep=';', encoding='ISO-8859-1', engine='python')
            return data
"""
        except Exception as e:
            print(f"Failed to load data from {self.file_path}. Error: {e}")
            raise e

        except Exception as e:
            print(f"Failed to load data from {self.file_path}. Error: {e}")
            raise e


daten_knoten = DataLoader(file_path='C:\\Users\\d.muehlfeld\Berechnungsdaten\\25_Schopfloch_NODE.CSV')
data = daten_knoten.load()
print(data)