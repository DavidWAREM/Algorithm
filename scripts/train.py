# Import global packages
import sys
import os

# Fügen Sie das Projektverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local settings
from data import DataLoader, preprocess_data


def main():
    # Daten laden
    beispieldaten = DataLoader(file_path='temp.xlsx')
    data = beispieldaten.load()
    print(data)

    # daten verarbeiten
    processed_data = preprocess_data(data)
    print(processed_data)


if __name__ == "__main__":
    main()
