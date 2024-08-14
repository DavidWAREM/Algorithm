import pandas as pd
import numpy as np
import os

# Beispiel-Ordner für die Speicherung der CSV-Dateien
output_folder = "C:/Users/d.muehlfeld/Berechnungsdaten/Zwischenspeicher/Beispieldaten"
os.makedirs(output_folder, exist_ok=True)

# Anzahl der Knoten und Leitungen
num_nodes = 50
num_edges = 60

# Generiere Beispiel-Knoten-Daten
knoten_data = pd.DataFrame({
    'KNAM': [f'K{str(i).zfill(4)}' for i in range(1, num_nodes + 1)],
    'ZUFLUSS': np.random.uniform(-0.05, 0.05, size=num_nodes),
    'FSTATUS': np.random.choice([np.nan, 'Active', 'Inactive'], size=num_nodes),
    'PMESS': np.random.uniform(0, 10, size=num_nodes),
    'PRECH': np.random.uniform(0, 10, size=num_nodes),
    'DP': np.random.uniform(-0.1, 0.1, size=num_nodes),
    'HP': np.random.uniform(230, 240, size=num_nodes),
    'XRECHTS': np.random.uniform(500000, 600000, size=num_nodes),
    'YHOCH': np.random.uniform(5000000, 5100000, size=num_nodes),
    'GEOH': np.random.uniform(50, 200, size=num_nodes),
    'SYMBOL': np.random.randint(0, 3, size=num_nodes),
    'ABGAENGE': np.random.randint(1, 5, size=num_nodes),
    'NETZNR': np.random.randint(1, 3, size=num_nodes)
})

# Generiere Beispiel-Leitungen-Daten
leitungen_data = pd.DataFrame({
    'ANFNAM': np.random.choice(knoten_data['KNAM'], size=num_edges),
    'ENDNAM': np.random.choice(knoten_data['KNAM'], size=num_edges),
    'ANFNR': np.random.randint(1, num_nodes + 1, size=num_edges),
    'ENDNR': np.random.randint(1, num_nodes + 1, size=num_edges),
    'RORL': np.random.uniform(50, 150, size=num_edges),
    'DM': np.random.uniform(100, 300, size=num_edges),
    'RAU': np.random.uniform(0.001, 0.01, size=num_edges),
    'FLUSS': np.random.uniform(-100, 100, size=num_edges),
    'VM': np.random.uniform(0, 5, size=num_edges),
    'DP': np.random.uniform(-0.5, 0.5, size=num_edges),
    'DPREL': np.random.uniform(-1, 1, size=num_edges),
    'ROHRTYP': np.random.choice(['100 GG', '150 GG', '200 GG'], size=num_edges),
    'RAISE': np.random.uniform(-50, 50, size=num_edges),
    'GroupID': np.random.choice([np.nan, 'Group1', 'Group2'], size=num_edges)
})

# Sicherstellen, dass keine selbstverbindenden Kanten existieren
leitungen_data = leitungen_data[leitungen_data['ANFNAM'] != leitungen_data['ENDNAM']]

# Beispiel CSV-Dateien speichern
knoten_file_path = os.path.join(output_folder, 'XXX_Node.csv')
leitungen_file_path = os.path.join(output_folder, 'XXX_Pipes.csv')

knoten_data.to_csv(knoten_file_path, sep=';', index=False, encoding='latin1')
leitungen_data.to_csv(leitungen_file_path, sep=';', index=False, encoding='latin1')

print(f"Beispiel-Knoten-Daten gespeichert unter: {knoten_file_path}")
print(f"Beispiel-Leitungen-Daten gespeichert unter: {leitungen_file_path}")
