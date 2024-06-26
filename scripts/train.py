# Import global packages
import sys
import os

# Fügen Sie das Projektverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local settings
from data import DataLoader, preprocess_data


def main():
    # Daten Knoten laden
    daten_knoten = DataLoader(file_path='C:\\Users\\d.muehlfeld\Berechnungsdaten\\25_Schopfloch_NODE.CSV')
    data = daten_knoten.load()
    print(data)

    # daten verarbeiten
    processed_data = preprocess_data(data)
    print(processed_data)


if __name__ == "__main__":
    main()


