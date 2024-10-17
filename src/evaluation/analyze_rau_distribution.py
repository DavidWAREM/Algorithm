import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Initialisiere das Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Funktion zum Laden und Überprüfen der Verteilung der RAU-Werte
def load_and_analyze_rau_values(directory, num_valves=100):
    all_rau_values = []

    # Iteriere durch alle Dateien und lade die RAU-Werte
    for i in range(1, num_valves + 1):
        edge_file = os.path.join(directory, f'SyntheticData-Spechbach_Simplification^2_Roughness_{i}_combined_Pipes.csv')

        try:
            edges_df = pd.read_csv(edge_file, delimiter=';', decimal='.')
            logger.info(f"Datei {edge_file} geladen.")

            # Stelle sicher, dass die Spalte 'RAU' vorhanden ist
            if 'RAU' not in edges_df.columns:
                logger.error(f"Die Spalte 'RAU' fehlt in {edge_file}.")
                continue

            # Füge die RAU-Werte zur Liste hinzu
            all_rau_values.extend(edges_df['RAU'].values)

        except Exception as e:
            logger.error(f"Fehler beim Laden der Datei {edge_file}: {e}")

    # Konvertiere die RAU-Werte in einen Pandas DataFrame für die Analyse
    all_rau_values_df = pd.DataFrame(all_rau_values, columns=['RAU'])

    # Zeige die statistische Verteilung der RAU-Werte
    logger.info("Statistische Beschreibung der RAU-Werte:")
    logger.info(all_rau_values_df.describe())

    # Visualisiere die Verteilung der RAU-Werte
    plt.figure(figsize=(10, 6))
    sns.histplot(all_rau_values_df['RAU'], bins=50, kde=True)
    plt.title('Verteilung der RAU-Werte')
    plt.xlabel('RAU-Wert')
    plt.ylabel('Häufigkeit')
    plt.grid(True)
    plt.show()


# Hauptfunktion zum Ausführen der Analyse
def main():
    # Setze das Verzeichnis für die Daten (wie in deinem ursprünglichen Code)
    directory = 'C:\\Users\\D.Muehlfeld\\Documents\\Berechnungsdaten_Roughness\\Zwischenspeicher'  # Pfad zu deinen Daten anpassen

    # Setze die Anzahl der Dateien, die du analysieren möchtest (basiert auf deinem ursprünglichen Code)
    num_valves = 990  # Anpassen, falls du mehr Dateien hast

    # Lade und analysiere die RAU-Werte
    load_and_analyze_rau_values(directory, num_valves=num_valves)


if __name__ == "__main__":
    main()
