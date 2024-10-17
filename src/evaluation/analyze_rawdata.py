import os
import pandas as pd
import numpy as np
import logging
from scipy.stats import skew, kurtosis
import glob

# Funktion zur Einrichtung des Loggings
def setup_logging(directory):
    log_file = os.path.join(directory, 'analysis.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Entferne alle vorherigen Handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # Erstelle einen FileHandler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Definiere das Format
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    # Füge den Handler dem Logger hinzu
    logger.addHandler(fh)

    return logger

# Funktion zum Laden und Überprüfen der Daten
def load_and_analyze_data(directory, logger):
    master_data = []

    # Definiere die relevanten Features für Pipes und Nodes, einschließlich der neuen physikalischen Variablen
    pipe_features = [
        'DM', 'RAU', 'RAISE', 'VM_WL', 'VM_WOL', 'FLUSS_WL', 'FLUSS_WOL', 'RORL',
        'Re_WL', 'Re_WOL', 'f_WL', 'f_WOL', 'tau_w_WL', 'tau_w_WOL',
        'u_star_WL', 'u_star_WOL', 'delta_WL', 'delta_WOL',
        'h_f_WL', 'h_f_WOL', 'S_WL', 'S_WOL',
        'Reibungsverlust_mbar_km_WL', 'Reibungsverlust_mbar_km_WOL',
        'flow_regime_WL', 'flow_regime_WOL'
    ]

    node_features = [
        'PRECH_WL', 'PRECH_WOL', 'HP_WL', 'HP_WOL', 'ZUFLUSS_WL', 'ZUFLUSS_WOL',
        'dp', 'delta_H'
    ]

    # Verwende glob, um alle relevanten Dateien zu finden
    node_pattern = os.path.join(directory, '*_combined_Node.csv')
    pipe_pattern = os.path.join(directory, '*_combined_Pipes.csv')

    node_files = glob.glob(node_pattern)
    pipe_files = glob.glob(pipe_pattern)

    logger.info(f"Gefundene Node-Dateien: {len(node_files)}")
    logger.info(f"Gefundene Pipe-Dateien: {len(pipe_files)}")

    # Extrahiere die Basisnamen (ohne Suffix), um Node- und Pipe-Dateien zu paaren
    node_bases = set([os.path.basename(f).replace('_combined_Node.csv', '') for f in node_files])
    pipe_bases = set([os.path.basename(f).replace('_combined_Pipes.csv', '') for f in pipe_files])

    # Finde die gemeinsamen Basen
    common_bases = node_bases.intersection(pipe_bases)
    logger.info(f"Gefundene Paare: {len(common_bases)}")

    # Iteriere durch die gemeinsamen Basen und lade die entsprechenden Dateien
    for base in common_bases:
        node_file = os.path.join(directory, f"{base}_combined_Node.csv")
        pipe_file = os.path.join(directory, f"{base}_combined_Pipes.csv")

        try:
            # Lade Pipes-Daten
            pipes_df = pd.read_csv(pipe_file, delimiter=';', decimal='.')
            logger.info(f"Datei {pipe_file} geladen.")

            # Überprüfe, ob alle erforderlichen Pipe-Features vorhanden sind
            missing_pipe_cols = [col for col in pipe_features if col not in pipes_df.columns]
            if missing_pipe_cols:
                logger.error(f"Fehlende Pipe-Spalten in {pipe_file}: {missing_pipe_cols}.")
                continue

            # Extrahiere relevante Pipe-Features
            pipes_data = pipes_df[pipe_features].copy()

            # Lade Nodes-Daten
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            logger.info(f"Datei {node_file} geladen.")

            # Überprüfe, ob alle erforderlichen Node-Features vorhanden sind
            missing_node_cols = [col for col in node_features if col not in nodes_df.columns]
            if missing_node_cols:
                logger.error(f"Fehlende Node-Spalten in {node_file}: {missing_node_cols}.")
                continue

            # Extrahiere relevante Node-Features
            nodes_data = nodes_df[node_features].copy()

            # Füge die Daten zusammen
            combined_data = pd.concat([pipes_data, nodes_data], axis=1)

            # Füge die kombinierten Daten zur Liste hinzu
            master_data.append(combined_data)

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten von {base}: {e}")
            continue

    # Kombiniere alle Daten in einen DataFrame
    if not master_data:
        logger.error("Keine Daten zum Analysieren gefunden.")
        return None

    master_df = pd.concat(master_data, ignore_index=True)
    logger.info(f"Gesamte Daten geladen: {master_df.shape[0]} Zeilen.")

    return master_df

# Funktion zur Erstellung des Analyseberichts
def save_analysis_report(df, directory, logger):
    report_path = os.path.join(directory, 'RAU_Analysis_Report.txt')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        # Statistische Beschreibung der RAU-Werte
        report_file.write("### Statistische Beschreibung der RAU-Werte:\n")
        rau_desc = df['RAU'].describe()
        report_file.write(str(rau_desc) + "\n\n")

        # Schiefe und Kurtosis
        rau_skewness = skew(df['RAU'].dropna())
        rau_kurtosis = kurtosis(df['RAU'].dropna())
        report_file.write(f"Schiefe der RAU-Verteilung: {rau_skewness:.2f}\n")
        report_file.write(f"Kurtosis der RAU-Verteilung: {rau_kurtosis:.2f}\n\n")

        # Korrelationsmatrix (mit Fokus auf RAU)
        report_file.write("### Korrelationsmatrix (mit Fokus auf RAU):\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        rau_corr = corr_matrix[['RAU']]
        report_file.write(str(rau_corr) + "\n\n")

        # Untersuchung nicht-linearer Beziehungen
        report_file.write("### Untersuchung nicht-linearer Beziehungen:\n")
        transformations = ['square', 'sqrt', 'log']
        transformed_corrs = pd.DataFrame()

        for col in numeric_cols:
            if col == 'RAU':
                continue  # Skip RAU itself

            for trans in transformations:
                transformed_col_name = f"{col}_{trans}"
                try:
                    if trans == 'square':
                        df[transformed_col_name] = df[col] ** 2
                    elif trans == 'sqrt':
                        df[transformed_col_name] = np.sqrt(df[col].clip(lower=0))
                    elif trans == 'log':
                        # Add a small constant to avoid log(0)
                        df[transformed_col_name] = np.log(df[col].clip(lower=1e-8))
                    # Calculate correlation with RAU
                    corr = df[['RAU', transformed_col_name]].corr().iloc[0, 1]
                    transformed_corrs.loc[transformed_col_name, 'Correlation'] = corr
                except Exception as e:
                    logger.warning(f"Transformation {trans} konnte für Spalte {col} nicht angewendet werden: {e}")

        # Sortiere die Korrelationen nach absolutem Wert absteigend
        transformed_corrs = transformed_corrs.sort_values(by='Correlation', key=lambda x: x.abs(), ascending=False)

        # Formatierung der transformierten Korrelationen für die Ausgabe
        report_file.write("Korrelationskoeffizienten der transformierten Variablen mit RAU:\n")
        if not transformed_corrs.empty:
            # Überschrift
            report_file.write(f"{'Variable':<30} {'Correlation':>12}\n")
            report_file.write(f"{'-'*30} {'-'*12}\n")
            # Jede Zeile der transformierten Korrelationen
            for index, row in transformed_corrs.iterrows():
                report_file.write(f"{index:<30} {row['Correlation']:>12.4f}\n")
        else:
            report_file.write("Keine transformierten Variablen gefunden.\n")
        report_file.write("\n")

        # Empfehlungen
        report_file.write("### Empfehlungen:\n")
        recommendations = provide_recommendations(df, transformed_corrs)
        for rec in recommendations:
            report_file.write(f"- {rec}\n")

    logger.info(f"Analysebericht gespeichert unter: {report_path}")


# Funktion zur Generierung von Empfehlungen
def provide_recommendations(df, transformed_corrs):
    recommendations = []

    # Schiefe
    skew_val = skew(df['RAU'].dropna())
    if abs(skew_val) > 1:
        recommendations.append("Die RAU-Verteilung ist stark schief. Erwäge eine Transformation.")
    elif abs(skew_val) > 0.5:
        recommendations.append("Die RAU-Verteilung ist mäßig schief. Eine Transformation könnte hilfreich sein.")
    else:
        recommendations.append("Die RAU-Verteilung ist ungefähr symmetrisch.")

    # Korrelationen prüfen
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    rau_corr = corr_matrix['RAU'].drop('RAU')

    high_corr_features = rau_corr[abs(rau_corr) >= 0.3].index.tolist()
    low_corr_features = rau_corr[abs(rau_corr) < 0.3].index.tolist()

    if high_corr_features:
        recommendations.append(f"Die folgenden Features haben eine hohe Korrelation mit RAU: {', '.join(high_corr_features)}")
    else:
        recommendations.append("Keine Features mit hoher Korrelation zu RAU in den ursprünglichen Variablen gefunden.")

    # Überprüfung der transformierten Variablen
    high_corr_transformed = transformed_corrs[abs(transformed_corrs['Correlation']) >= 0.3].index.tolist()
    if high_corr_transformed:
        recommendations.append(f"Nach Anwendung nicht-linearer Transformationen haben die folgenden Variablen eine hohe Korrelation mit RAU: {', '.join(high_corr_transformed)}")
    else:
        recommendations.append("Auch nach nicht-linearer Transformation wurden keine starken Korrelationen gefunden.")

    if low_corr_features and not high_corr_transformed:
        recommendations.append("Erwäge, zusätzliche Features hinzuzufügen oder Feature Engineering durchzuführen.")

    return recommendations

# Hauptfunktion
def main():
    # Pfad anpassen
    directory = 'C:\\Users\\D.Muehlfeld\\Documents\\Berechnungsdaten_Roughness\\Zwischenspeicher'  # Passe diesen Pfad an

    # Logging einrichten
    logger = setup_logging(directory)

    # Daten laden und analysieren
    df = load_and_analyze_data(directory, logger)
    if df is None:
        logger.error("Analyse abgebrochen.")
        return

    # Analysebericht speichern
    save_analysis_report(df, directory, logger)

    logger.info("Analyse abgeschlossen.")

if __name__ == "__main__":
    main()
