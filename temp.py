import pandas as pd
import numpy as np
import os


# Funktion zum Erstellen eines Excel-Dateis
def create_excel_file(file_name, num_rows=100):
    # Erstellen zufälliger Daten für Spalte A
    data = np.random.rand(num_rows)

    # Berechnen der Werte für Spalte B
    results = data * 0.5

    # Erstellen eines DataFrame
    df = pd.DataFrame({'A': data, 'B': results})

    # Speichern des DataFrame in einer Excel-Datei
    df.to_excel(file_name, index=False)


# Ordner erstellen, in dem die Dateien gespeichert werden
output_folder = "generated_excel_files"
os.makedirs(output_folder, exist_ok=True)

# Erstellen und Speichern der 10 Excel-Dateien
for i in range(1, 101):
    file_name = os.path.join(output_folder, f"file_{i}.xlsx")
    create_excel_file(file_name)

print(f"10 Excel files have been created in the folder '{output_folder}'")
