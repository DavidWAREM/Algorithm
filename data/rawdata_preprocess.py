import pandas as pd
import os
import logging
import re
import numpy as np  # Für numerische Berechnungen

class DataProcessor:
    def __init__(self, dataframe, original_file_path):
        """
        Initialize the DataProcessor with a pandas DataFrame and the path to the original file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to process.
            original_file_path (str): The path to the original CSV file.

        The constructor sets up the necessary attributes such as the DataFrame to process and the file path details.
        It also extracts the base filename and directory to use for saving output files.
        """
        self.logger = logging.getLogger(__name__)  # Initialize logger
        self.dataframe = dataframe  # The DataFrame provided for processing.
        self.original_file_path = original_file_path  # The original file's path, used for saving files.
        self.directory = os.path.dirname(original_file_path)  # The directory where the file is located.
        self.base_filename = os.path.splitext(os.path.basename(original_file_path))[0]  # Extract the base filename without extension.
        self.logger.info(f"DataProcessor initialized with file path: {original_file_path}")

    def split_data(self):
        """
        Split the DataFrame into two DataFrames based on the value in the first column ('KNO' and 'LEI').
        The row immediately above the first occurrence of 'KNO' and 'LEI' is used as headers.

        Returns:
            tuple: Two DataFrames, one for rows with 'KNO' and one for rows with 'LEI'.

        This method identifies the rows corresponding to 'KNO' and 'LEI', splits the DataFrame accordingly,
        and assigns the header row before splitting. It also filters relevant columns for both 'KNO' and 'LEI' DataFrames.
        """
        try:
            # Find the first occurrence of 'KNO' and 'LEI' in the DataFrame.
            kno_index = self.dataframe.index[self.dataframe.iloc[:, 0] == 'KNO'][0]
            lei_index = self.dataframe.index[self.dataframe.iloc[:, 0] == 'LEI'][0]

            # Define header rows (one row above 'KNO' and 'LEI' for column names).
            kno_header_index = kno_index - 1 if kno_index > 0 else kno_index
            kno_df = self.dataframe.iloc[kno_index:].copy()  # Copy the data starting from 'KNO'.
            kno_df.columns = self.dataframe.iloc[kno_header_index]  # Set header row as column names.

            lei_header_index = lei_index - 1 if lei_index > 0 else lei_index
            lei_df = self.dataframe.iloc[lei_index:].copy()  # Copy the data starting from 'LEI'.
            lei_df.columns = self.dataframe.iloc[lei_header_index]  # Set header row as column names.

            # Filter rows with 'KNO' and 'LEI' in the first column.
            kno_df = kno_df[kno_df.iloc[:, 0] == 'KNO']
            lei_df = lei_df[lei_df.iloc[:, 0] == 'LEI']

            # Define relevant columns for each DataFrame.
            kno_columns = ['KNAM', 'ZUFLUSS', 'GEOH', 'PRECH',
                           'XRECHTS', 'YHOCH', 'HP']
            lei_columns = ['ANFNAM', 'ENDNAM', 'RORL', 'DM', 'RAU', 'FLUSS',
                           'VM', 'DPREL', 'ROHRTYP', 'RAISE', 'RE']

            # Ensure that only relevant columns in kno_columns are extracted from kno_df.
            kno_columns_present = [col for col in kno_columns if col in kno_df.columns]
            kno_df = kno_df[kno_columns_present]

            # Ensure that only relevant columns in lei_columns are extracted from lei_df.
            lei_columns_present = [col for col in lei_columns if col in lei_df.columns]
            lei_df = lei_df[lei_columns_present]

            self.logger.info("Data split successfully into 'KNO' and 'LEI'")
            return kno_df, lei_df  # Return the split DataFrames.
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")  # Log the error if splitting fails.
            return None, None  # Return None for both DataFrames in case of error.

    def save_dataframes(self):
        """
        Save the two DataFrames (split by 'KNO' and 'LEI') into separate CSV files in the same directory as the original file.
        """
        try:
            # Split the data into 'KNO' and 'LEI' DataFrames.
            kno_df, lei_df = self.split_data()
            if kno_df is not None and lei_df is not None:
                # Define file paths für das Speichern der DataFrames.
                zwischenspeicher_dir = os.path.join(self.directory, 'Zwischenspeicher')
                if not os.path.exists(zwischenspeicher_dir):
                    os.makedirs(zwischenspeicher_dir)
                    self.logger.debug(f"Created 'Zwischenspeicher' directory: {zwischenspeicher_dir}")

                kno_path = os.path.join(zwischenspeicher_dir, f"{self.base_filename}_Node.csv")
                lei_path = os.path.join(zwischenspeicher_dir, f"{self.base_filename}_Pipes.csv")

                # Speichern des 'KNO' DataFrames als CSV.
                kno_df.to_csv(kno_path, index=False, sep=';')
                # Speichern des 'LEI' DataFrames als CSV.
                lei_df.to_csv(lei_path, index=False, sep=';')

                self.logger.info(f"DataFrames saved successfully: {kno_path} and {lei_path}")
            else:
                self.logger.error("DataFrames could not be saved due to an earlier error.")
        except Exception as e:
            self.logger.error(f"Error saving DataFrames: {e}")  # Log any errors encountered during saving.


class DataCombiner:
    def __init__(self, directory):
        """
        Initialisiert den DataCombiner mit dem Verzeichnis, in dem sich der 'Zwischenspeicher' Ordner befindet.

        Args:
            directory (str): Der Basisverzeichnis-Pfad, der den 'Zwischenspeicher' Ordner enthält.
        """
        self.logger = logging.getLogger(__name__)
        self.directory = directory
        self.logger.info(f"DataCombiner initialisiert mit Verzeichnis: {directory}")

    def combine_with_without_load(self, file_type):
        """
        Kombiniert Daten aus 'with_load' und 'without_load' CSV-Dateien basierend auf übereinstimmenden Zahlen in den Dateinamen.
        Diese Funktion verarbeitet entweder 'Pipes' oder 'Node' Typ Dateien, kombiniert die relevanten Spalten und speichert
        das Ergebnis in einer neuen CSV-Datei, wobei der '_with' Teil aus dem Dateinamen entfernt wird.

        Zusätzlich berechnet sie physikalische Größen für 'Pipes' Dateien und integriert diese in das kombinierte DataFrame,
        einschließlich der Berechnungen für die vier neuen nicht-linearen Features mit der höchsten Korrelation zu RAU.

        Args:
            file_type (str): Der Typ der zu kombinierenden Datei, entweder 'Pipes' oder 'Node'.
        """
        try:
            # Konstanten für physikalische Berechnungen
            rho = 1000  # Dichte von Wasser in kg/m^3
            mu = 0.001  # Dynamische Viskosität von Wasser in Pa·s
            g = 9.81    # Erdbeschleunigung in m/s^2
            nu = mu / rho  # Kinematische Viskosität in m^2/s

            # Pfad zum 'Zwischenspeicher' Verzeichnis
            zwischenspeicher_dir = os.path.join(self.directory, 'Zwischenspeicher')
            self.logger.info(f"Suche nach Dateien im Verzeichnis: {zwischenspeicher_dir}")

            if not os.path.exists(zwischenspeicher_dir):
                self.logger.error(f"'Zwischenspeicher' Verzeichnis nicht gefunden unter: {zwischenspeicher_dir}")
                return

            # Initialisiere Dictionaries zum Speichern von with_load und without_load Dateien
            with_load_files = {}
            without_load_files = {}

            # Durchlaufe Dateien im 'Zwischenspeicher' Verzeichnis und gleiche basierend auf file_type ab
            for file in os.listdir(zwischenspeicher_dir):
                with_load_match = re.match(rf"(.+)_with_load_(\d+)_({file_type})\.csv", file)
                without_load_match = re.match(rf"(.+)_without_load_(\d+)_({file_type})\.csv", file)

                if with_load_match:
                    number = with_load_match.group(2)
                    with_load_files[number] = os.path.join(zwischenspeicher_dir, file)
                    self.logger.debug(f"Gefundene with_load Datei: {file} mit Nummer: {number}")
                elif without_load_match:
                    number = without_load_match.group(2)
                    without_load_files[number] = os.path.join(zwischenspeicher_dir, file)
                    self.logger.debug(f"Gefundene without_load Datei: {file} mit Nummer: {number}")

            # Verarbeite übereinstimmende Paare von with_load und without_load Dateien
            for number in with_load_files.keys():
                if number in without_load_files:
                    wl_file = with_load_files[number]
                    wol_file = without_load_files[number]
                    self.logger.info(f"Kombiniere with_load Datei: {wl_file} und without_load Datei: {wol_file}")

                    # Lese beide CSV Dateien in DataFrames ein
                    df_wl = pd.read_csv(wl_file, sep=';', decimal='.', encoding='utf-8')
                    df_wol = pd.read_csv(wol_file, sep=';', decimal='.', encoding='utf-8')

                    if file_type == 'Pipes':
                        # Spalten für 'Pipes' Dateien ohne DPREL
                        key_columns = ['ANFNAM', 'ENDNAM', 'RORL', 'DM', 'ROHRTYP', 'RAISE']

                        self.logger.debug(f"Key Spalten für 'Pipes' (ohne DPREL): {key_columns}")

                        # Erstelle ein neues DataFrame mit den Key Spalten aus 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Füge VM und FLUSS Spalten sowohl aus with_load als auch ohne_load hinzu
                        combined_df['VM_WL'] = pd.to_numeric(df_wl['VM'], errors='coerce')
                        combined_df['FLUSS_WL'] = pd.to_numeric(df_wl['FLUSS'], errors='coerce')
                        combined_df['VM_WOL'] = pd.to_numeric(df_wol['VM'], errors='coerce')
                        combined_df['FLUSS_WOL'] = pd.to_numeric(df_wol['FLUSS'], errors='coerce')
                        combined_df['RE_WL'] = pd.to_numeric(df_wl['RE'], errors='coerce')
                        combined_df['RE_WOL'] = pd.to_numeric(df_wol['RE'], errors='coerce')

                        # Füge RAU Spalte hinzu
                        if 'RAU' in df_wl.columns:
                            combined_df['RAU'] = pd.to_numeric(df_wl['RAU'], errors='coerce')
                            self.logger.info(f"RAU Spalte hinzugefügt für Datei: {wl_file}")
                        else:
                            combined_df['RAU'] = np.nan
                            self.logger.warning(f"RAU Spalte fehlt in Datei: {wl_file}")

                        # Konvertiere Spalten in numerische Typen
                        numeric_columns = ['DM', 'VM_WL', 'VM_WOL', 'FLUSS_WL', 'FLUSS_WOL', 'RAU', 'RORL', 'RE_WL', 'RE_WOL']
                        for col in numeric_columns:
                            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

                        # **Konvertiere DM und RORL von mm zu Metern**
                        combined_df['DM'] = combined_df['DM'] / 1000  # mm zu m

                        # Berechne Reynolds-Zahlen
                        combined_df['Re_WL'] = (combined_df['VM_WL'] * combined_df['DM']) / nu
                        combined_df['Re_WOL'] = (combined_df['VM_WOL'] * combined_df['DM']) / nu

                        # **Berechne Reibungsfaktoren ohne RAU**
                        def calculate_friction_factor(Re, D, assumed_roughness=0.0001):
                            """
                            Berechnet den Reibungsfaktor basierend auf der Reynolds-Zahl und dem Rohrdurchmesser.
                            Verwendet die Blasius-Gleichung für turbulente Strömung in glatten Rohren und Standardformel für laminare Strömung.

                            Parameters:
                            - Re: Reynolds-Zahl (arrayähnlich)
                            - D: Rohrdurchmesser (arrayähnlich)
                            - assumed_roughness: Angenommene Rauhigkeit des Rohrs (Standardwert 0.0001 Meter)

                            Returns:
                            - f: Reibungsfaktor (arrayähnlich)
                            """
                            Re = np.maximum(Re, 1e-6)  # Vermeide Division durch Null
                            D = np.maximum(D, 1e-6)    # Vermeide Division durch Null
                            # Relative Rauhigkeit wird nicht verwendet, da glatte Rohre angenommen werden
                            # Berechne Reibungsfaktor für laminare Strömung
                            f_laminar = 64 / Re
                            # Berechne Reibungsfaktor für turbulente Strömung mit Blasius-Gleichung
                            f_turbulent = 0.3164 * Re**-0.25
                            # Definiere Masken für verschiedene Strömungsregime
                            laminar_mask = Re < 2000
                            turbulent_mask = Re > 4000
                            transition_mask = (~laminar_mask) & (~turbulent_mask)
                            # Initialisiere Reibungsfaktor Array
                            f = np.zeros_like(Re)
                            # Weisen Sie Reibungsfaktoren basierend auf den Strömungsregimen zu
                            f[laminar_mask] = f_laminar[laminar_mask]
                            f[turbulent_mask] = f_turbulent[turbulent_mask]
                            # Lineare Interpolation für Übergangsregime
                            f[transition_mask] = f_laminar[transition_mask] + (
                                (Re[transition_mask] - 2000) * (f_turbulent[transition_mask] - f_laminar[transition_mask]) / 2000
                            )
                            return f

                        # Berechne Reibungsfaktoren ohne RAU
                        combined_df['f_WL'] = calculate_friction_factor(combined_df['Re_WL'], combined_df['DM'])
                        combined_df['f_WOL'] = calculate_friction_factor(combined_df['Re_WOL'], combined_df['DM'])

                        # Berechne Wandreibungsspannung
                        combined_df['tau_w_WL'] = (combined_df['f_WL'] / 8) * rho * (combined_df['VM_WL'] ** 2)
                        combined_df['tau_w_WOL'] = (combined_df['f_WOL'] / 8) * rho * (combined_df['VM_WOL'] ** 2)

                        # Berechne Reibungsgeschwindigkeit
                        combined_df['u_star_WL'] = np.sqrt(combined_df['tau_w_WL'] / rho)
                        combined_df['u_star_WOL'] = np.sqrt(combined_df['tau_w_WOL'] / rho)

                        # Berechne Dicke der laminaren Unterschicht
                        combined_df['delta_WL'] = (5 * nu) / combined_df['u_star_WL']
                        combined_df['delta_WOL'] = (5 * nu) / combined_df['u_star_WOL']

                        # Berechne Reibungsverlust h_f mittels Darcy-Weisbach-Gleichung
                        combined_df['h_f_WL'] = combined_df['f_WL'] * (combined_df['RORL'] / combined_df['DM']) * \
                                                (combined_df['VM_WL'] ** 2) / (2 * g)
                        combined_df['h_f_WOL'] = combined_df['f_WOL'] * (combined_df['RORL'] / combined_df['DM']) * \
                                                 (combined_df['VM_WOL'] ** 2) / (2 * g)

                        # Berechne Energiegradient S
                        combined_df['S_WL'] = combined_df['h_f_WL'] / combined_df['RORL']
                        combined_df['S_WOL'] = combined_df['h_f_WOL'] / combined_df['RORL']

                        # Berechne Strömungsregime-Indikator
                        combined_df['flow_regime_WL'] = np.where(
                            combined_df['Re_WL'] < 2000, 0,
                            np.where(combined_df['Re_WL'] <= 4000, 1, 2)
                        )
                        combined_df['flow_regime_WOL'] = np.where(
                            combined_df['Re_WOL'] < 2000, 0,
                            np.where(combined_df['Re_WOL'] <= 4000, 1, 2)
                        )

                        # Berechne Reibungsverlust pro Kilometer (mbar/km)
                        # Formel: ΔP(mbar/km) = (f * rho * v^2 * 5) / D
                        combined_df['Reibungsverlust_mbar_km_WL'] = (combined_df['f_WL'] * rho * (combined_df['VM_WL'] ** 2) * 5) / combined_df['DM']
                        combined_df['Reibungsverlust_mbar_km_WOL'] = (combined_df['f_WOL'] * rho * (combined_df['VM_WOL'] ** 2) * 5) / combined_df['DM']

                        # Sicherstellen, dass keine unendlichen oder NaN Werte in den Berechnungen vorhanden sind
                        calc_columns = ['Re_WL', 'Re_WOL', 'f_WL', 'f_WOL', 'tau_w_WL', 'tau_w_WOL',
                                        'u_star_WL', 'u_star_WOL', 'delta_WL', 'delta_WOL',
                                        'h_f_WL', 'h_f_WOL', 'S_WL', 'S_WOL',
                                        'Reibungsverlust_mbar_km_WL', 'Reibungsverlust_mbar_km_WOL']
                        for col in calc_columns:
                            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

                        # Konvertiere Strömungsregime-Indikatoren in Ganzzahltypen
                        combined_df['flow_regime_WL'] = combined_df['flow_regime_WL'].astype(int)
                        combined_df['flow_regime_WOL'] = combined_df['flow_regime_WOL'].astype(int)

                        # **Berechne die vier neuen nicht-linearen Features mit höchster Korrelation zu RAU**

                        # 1. RAISE_log und RAISE_sqrt
                        combined_df['RAISE_log'] = combined_df['RAISE'].apply(lambda x: np.log(x) if x > 0 else np.nan)
                        combined_df['RAISE_sqrt'] = combined_df['RAISE'].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)

                        # 2. h_f_WL_sqrt und h_f_WOL_sqrt
                        combined_df['h_f_WL_sqrt'] = combined_df['h_f_WL'].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
                        combined_df['h_f_WOL_sqrt'] = combined_df['h_f_WOL'].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)

                        # Überprüfen, ob die transformierten Features berechnet wurden
                        required_transformed_features = ['RAISE_log', 'RAISE_sqrt', 'h_f_WL_sqrt', 'h_f_WOL_sqrt']
                        for feature in required_transformed_features:
                            if feature not in combined_df.columns or combined_df[feature].isnull().all():
                                combined_df[feature] = np.nan
                                self.logger.warning(f"{feature} wurde nicht berechnet und auf NaN gesetzt.")

                        # Füllen von NaN-Werten mit 0 (oder einer anderen geeigneten Methode)
                        for feature in required_transformed_features:
                            if feature in combined_df.columns:
                                combined_df[feature] = combined_df[feature].fillna(0)
                                self.logger.debug(f"{feature} fehlende Werte mit 0 gefüllt.")

                    elif file_type == 'Node':
                        # Spalten für 'Node' Dateien
                        key_columns = ['KNAM', 'GEOH', 'XRECHTS', 'YHOCH']
                        self.logger.debug(f"Key Spalten für 'Node': {key_columns}")

                        # Erstelle ein neues DataFrame mit den Key Spalten aus 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Füge PRECH, HP und ZUFLUSS Spalten sowohl aus with_load als auch ohne_load hinzu
                        combined_df['PRECH_WL'] = pd.to_numeric(df_wl['PRECH'], errors='coerce')
                        combined_df['HP_WL'] = pd.to_numeric(df_wl['HP'], errors='coerce')
                        combined_df['ZUFLUSS_WL'] = pd.to_numeric(df_wl['ZUFLUSS'], errors='coerce')
                        combined_df['PRECH_WOL'] = pd.to_numeric(df_wol['PRECH'], errors='coerce')
                        combined_df['HP_WOL'] = pd.to_numeric(df_wol['HP'], errors='coerce')
                        combined_df['ZUFLUSS_WOL'] = pd.to_numeric(df_wol['ZUFLUSS'], errors='coerce')

                        # Füge dp Spalte für Node Dateien als PRECH_WOL - PRECH_WL hinzu
                        combined_df['dp'] = combined_df['PRECH_WOL'] - combined_df['PRECH_WL']
                        self.logger.debug(f"Spalte dp als PRECH_WOL - PRECH_WL für Node berechnet")

                        # Berechne Änderung in der hydraulischen Höhe (delta_H)
                        combined_df['delta_H'] = combined_df['HP_WL'] - combined_df['HP_WOL']

                        # Sicherstellen, dass keine unendlichen oder NaN Werte in den Berechnungen vorhanden sind
                        node_calc_columns = ['PRECH_WL', 'HP_WL', 'ZUFLUSS_WL', 'PRECH_WOL',
                                             'HP_WOL', 'ZUFLUSS_WOL', 'dp', 'delta_H']
                        for col in node_calc_columns:
                            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

                    # Extrahiere den Basisdateinamen ohne "_with" und definiere den Ausgabedateinamen
                    base_name = re.sub(rf'_with_load_\d+_({file_type})\.csv', '', os.path.basename(wl_file))
                    output_file = os.path.join(zwischenspeicher_dir, f"{base_name}_{number}_combined_{file_type}.csv")

                    # Speichere das kombinierte DataFrame in eine neue CSV Datei
                    combined_df.to_csv(output_file, index=False, sep=';', decimal='.')
                    self.logger.info(f"Kombinierte CSV Datei erfolgreich gespeichert unter: {output_file}")

        except Exception as e:
            self.logger.error(f"Fehler beim Kombinieren der Dateien: {e}", exc_info=True)


