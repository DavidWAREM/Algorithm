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
            kno_columns = ['REM', 'FLDNAM', 'KNO', 'KNAM', 'ZUFLUSS', 'GEOH', 'PRECH',
                           'XRECHTS', 'YHOCH', 'HP']
            lei_columns = ['REM', 'FLDNAM', 'LEI', 'ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'DM', 'RAU', 'FLUSS',
                           'VM', 'DPREL', 'ROHRTYP', 'RAISE', 'DPREL', 'DPREL']

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
        Initializes the DataCombiner with the directory where the 'Zwischenspeicher' folder is located.

        Args:
            directory (str): The base directory path containing the 'Zwischenspeicher' folder.
        """
        self.logger = logging.getLogger(__name__)
        self.directory = directory
        self.logger.info(f"DataCombiner initialized with directory: {directory}")

    def combine_with_without_load(self, file_type):
        """
        Combines data from 'with_load' and 'without_load' CSV files based on matching numbers in filenames.
        This function processes either 'Pipes' or 'Node' type files, merges the relevant columns, and saves
        the result into a new CSV file, excluding the '_with' part from the file name.

        Additionally, it calculates physical quantities for 'Pipes' files and integrates them into the combined DataFrame,
        including the calculations for the top variables with the highest correlation to RAU.

        Args:
            file_type (str): The type of file to combine, either 'Pipes' or 'Node'.
        """
        try:
            # Constants for physical calculations
            rho = 1000  # Density of water in kg/m^3
            mu = 0.001  # Dynamic viscosity of water in Pa·s
            g = 9.81    # Acceleration due to gravity in m/s^2
            nu = mu / rho  # Kinematic viscosity in m^2/s

            # Construct the path to the 'Zwischenspeicher' directory
            zwischenspeicher_dir = os.path.join(self.directory, 'Zwischenspeicher')
            self.logger.info(f"Looking for files in directory: {zwischenspeicher_dir}")

            if not os.path.exists(zwischenspeicher_dir):
                self.logger.error(f"'Zwischenspeicher' directory not found at: {zwischenspeicher_dir}")
                return

            # Initialize dictionaries to store with_load and without_load files
            with_load_files = {}
            without_load_files = {}

            # Loop through files in the 'Zwischenspeicher' directory and match based on file_type
            for file in os.listdir(zwischenspeicher_dir):
                with_load_match = re.match(rf"(.+)_with_load_(\d+)_({file_type})\.csv", file)
                without_load_match = re.match(rf"(.+)_without_load_(\d+)_({file_type})\.csv", file)

                if with_load_match:
                    number = with_load_match.group(2)
                    with_load_files[number] = os.path.join(zwischenspeicher_dir, file)
                    self.logger.debug(f"Found with_load file: {file} with number: {number}")
                elif without_load_match:
                    number = without_load_match.group(2)
                    without_load_files[number] = os.path.join(zwischenspeicher_dir, file)
                    self.logger.debug(f"Found without_load file: {file} with number: {number}")

            # Process matching pairs of with_load and without_load files
            for number in with_load_files.keys():
                if number in without_load_files:
                    wl_file = with_load_files[number]
                    wol_file = without_load_files[number]
                    self.logger.info(f"Combining with_load file: {wl_file} and without_load file: {wol_file}")

                    # Read both CSV files into dataframes
                    df_wl = pd.read_csv(wl_file, sep=';', decimal='.', encoding='utf-8')
                    df_wol = pd.read_csv(wol_file, sep=';', decimal='.', encoding='utf-8')

                    if file_type == 'Pipes':
                        # Columns for 'Pipes' files without DPREL
                        key_columns = ['ANFNAM', 'ENDNAM', 'ANFNR', 'ENDNR', 'RORL', 'DM', 'ROHRTYP', 'RAISE']

                        self.logger.debug(f"Key columns for 'Pipes' (without DPREL): {key_columns}")

                        # Create a new dataframe with the key columns from 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Add VM and FLUSS columns from both with_load and without_load
                        combined_df['VM_WL'] = pd.to_numeric(df_wl['VM'], errors='coerce')
                        combined_df['FLUSS_WL'] = pd.to_numeric(df_wl['FLUSS'], errors='coerce')
                        combined_df['VM_WOL'] = pd.to_numeric(df_wol['VM'], errors='coerce')
                        combined_df['FLUSS_WOL'] = pd.to_numeric(df_wol['FLUSS'], errors='coerce')

                        # Add RAU column
                        if 'RAU' in df_wl.columns:
                            combined_df['RAU'] = pd.to_numeric(df_wl['RAU'], errors='coerce')
                            self.logger.info(f"RAU column added for file: {wl_file}")
                        else:
                            combined_df['RAU'] = np.nan
                            self.logger.warning(f"RAU column missing in file: {wl_file}")

                        # Convert columns to numeric types
                        numeric_columns = ['DM', 'VM_WL', 'VM_WOL', 'FLUSS_WL', 'FLUSS_WOL', 'RAU', 'RORL']
                        for col in numeric_columns:
                            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

                        # **Convert DM, RAU, and RORL from mm to meters**
                        combined_df['DM'] = combined_df['DM'] / 1000  # mm to m
                        combined_df['RAU'] = combined_df['RAU'] / 1000  # mm to m
                        combined_df['RORL'] = combined_df['RORL'].apply(lambda x: float(str(x).replace(',', '.')))  # Ensure proper decimal format
                        combined_df['RORL'] = combined_df['RORL'] / 1000  # mm to m (falls RORL in mm ist, sonst diesen Schritt entfernen)

                        # Compute Reynolds numbers
                        combined_df['Re_WL'] = (rho * combined_df['VM_WL'] * combined_df['DM']) / mu
                        combined_df['Re_WOL'] = (rho * combined_df['VM_WOL'] * combined_df['DM']) / mu

                        # Define function to calculate friction factor including roughness
                        def calculate_friction_factor(Re, epsilon, D):
                            rel_roughness = epsilon / D
                            # Handle cases where Re is zero or NaN to avoid log10 issues
                            rel_roughness = rel_roughness.replace([np.inf, -np.inf], np.nan)
                            rel_roughness = rel_roughness.fillna(0)
                            # Avoid log10 of non-positive numbers
                            term = (rel_roughness / 3.7) + (5.74 / Re**0.9)
                            # Replace non-positive terms with a small positive number to avoid log10 errors
                            term = np.where(term <= 0, 1e-10, term)
                            f_turbulent = 0.25 / (np.log10(term))**2
                            f = np.where(Re < 2000, 64 / Re, f_turbulent)
                            return f

                        # Compute friction factors
                        combined_df['f_WL'] = calculate_friction_factor(combined_df['Re_WL'], combined_df['RAU'], combined_df['DM'])
                        combined_df['f_WOL'] = calculate_friction_factor(combined_df['Re_WOL'], combined_df['RAU'], combined_df['DM'])

                        # Compute wall shear stress (Wandreibungsspannung)
                        combined_df['tau_w_WL'] = (combined_df['f_WL'] / 8) * rho * (combined_df['VM_WL'] ** 2)
                        combined_df['tau_w_WOL'] = (combined_df['f_WOL'] / 8) * rho * (combined_df['VM_WOL'] ** 2)

                        # Compute friction velocity (Reibungsgeschwindigkeit)
                        combined_df['u_star_WL'] = np.sqrt(combined_df['tau_w_WL'] / rho)
                        combined_df['u_star_WOL'] = np.sqrt(combined_df['tau_w_WOL'] / rho)

                        # Compute thickness of laminar sublayer (Dicke der laminaren Unterschicht)
                        combined_df['delta_WL'] = (5 * nu) / combined_df['u_star_WL']
                        combined_df['delta_WOL'] = (5 * nu) / combined_df['u_star_WOL']

                        # Compute head loss h_f using Darcy-Weisbach equation
                        # Ensure RORL is in meters
                        combined_df['h_f_WL'] = combined_df['f_WL'] * (combined_df['RORL'] / combined_df['DM']) * \
                                                (combined_df['VM_WL'] ** 2) / (2 * g)
                        combined_df['h_f_WOL'] = combined_df['f_WOL'] * (combined_df['RORL'] / combined_df['DM']) * \
                                                 (combined_df['VM_WOL'] ** 2) / (2 * g)

                        # Compute energy gradient S
                        combined_df['S_WL'] = combined_df['h_f_WL'] / combined_df['RORL']
                        combined_df['S_WOL'] = combined_df['h_f_WOL'] / combined_df['RORL']

                        # Compute flow regime indicator
                        combined_df['flow_regime_WL'] = np.where(
                            combined_df['Re_WL'] < 2000, 0,
                            np.where(combined_df['Re_WL'] <= 4000, 1, 2)
                        )
                        combined_df['flow_regime_WOL'] = np.where(
                            combined_df['Re_WOL'] < 2000, 0,
                            np.where(combined_df['Re_WOL'] <= 4000, 1, 2)
                        )

                        # Compute Reibungsverlust pro Kilometer (mbar/km)
                        # Formel: ΔP(mbar/km) = (f * rho * v^2 * 5) / D
                        combined_df['Reibungsverlust_mbar_km_WL'] = (combined_df['f_WL'] * rho * (combined_df['VM_WL'] ** 2) * 5) / combined_df['DM']
                        combined_df['Reibungsverlust_mbar_km_WOL'] = (combined_df['f_WOL'] * rho * (combined_df['VM_WOL'] ** 2) * 5) / combined_df['DM']

                        # Ensure no infinite or NaN values in the calculations
                        calc_columns = ['Re_WL', 'Re_WOL', 'f_WL', 'f_WOL', 'tau_w_WL', 'tau_w_WOL',
                                        'u_star_WL', 'u_star_WOL', 'delta_WL', 'delta_WOL',
                                        'h_f_WL', 'h_f_WOL', 'S_WL', 'S_WOL',
                                        'Reibungsverlust_mbar_km_WL', 'Reibungsverlust_mbar_km_WOL']
                        for col in calc_columns:
                            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

                        # Convert flow regime indicators to integer type
                        combined_df['flow_regime_WL'] = combined_df['flow_regime_WL'].astype(int)
                        combined_df['flow_regime_WOL'] = combined_df['flow_regime_WOL'].astype(int)

                        # **Calculate the variables with highest correlation to RAU (excluding DPREL)**

                        # 1. tau_w_WL_square
                        combined_df['tau_w_WL_square'] = combined_df['tau_w_WL'] ** 2

                        # 2. S_WL_square
                        combined_df['S_WL_square'] = combined_df['S_WL'] ** 2

                        # 3. Reibungsverlust_mbar_km_WL_square
                        combined_df['Reibungsverlust_mbar_km_WL_square'] = combined_df['Reibungsverlust_mbar_km_WL'] ** 2

                        # 4. tau_w_WOL_square
                        combined_df['tau_w_WOL_square'] = combined_df['tau_w_WOL'] ** 2

                        # 5. Reibungsverlust_mbar_km_WOL_square
                        combined_df['Reibungsverlust_mbar_km_WOL_square'] = combined_df['Reibungsverlust_mbar_km_WOL'] ** 2

                        # 6. S_WOL_square
                        combined_df['S_WOL_square'] = combined_df['S_WOL'] ** 2

                        # 7. f_WL_sqrt
                        combined_df['f_WL_sqrt'] = np.sqrt(combined_df['f_WL'].clip(lower=0))

                        # 8. u_star_WL_square
                        combined_df['u_star_WL_square'] = combined_df['u_star_WL'] ** 2

                        # 9. u_star_WOL_square
                        combined_df['u_star_WOL_square'] = combined_df['u_star_WOL'] ** 2

                        # 10. h_f_WL_square
                        combined_df['h_f_WL_square'] = combined_df['h_f_WL'] ** 2

                        # Ensure no infinite or NaN values in the new calculations
                        new_calc_columns = ['tau_w_WL_square', 'S_WL_square', 'Reibungsverlust_mbar_km_WL_square',
                                            'tau_w_WOL_square', 'Reibungsverlust_mbar_km_WOL_square', 'S_WOL_square',
                                            'f_WL_sqrt', 'u_star_WL_square', 'u_star_WOL_square', 'h_f_WL_square']
                        for col in new_calc_columns:
                            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

                    elif file_type == 'Node':
                        # Columns for 'Node' files
                        key_columns = ['KNAM', 'GEOH', 'XRECHTS', 'YHOCH']
                        self.logger.debug(f"Key columns for 'Node': {key_columns}")

                        # Create a new dataframe with the key columns from 'with_load'
                        combined_df = df_wl[key_columns].copy()

                        # Add PRECH, HP, and ZUFLUSS columns from both with_load and without_load
                        combined_df['PRECH_WL'] = pd.to_numeric(df_wl['PRECH'], errors='coerce')
                        combined_df['HP_WL'] = pd.to_numeric(df_wl['HP'], errors='coerce')
                        combined_df['ZUFLUSS_WL'] = pd.to_numeric(df_wl['ZUFLUSS'], errors='coerce')
                        combined_df['PRECH_WOL'] = pd.to_numeric(df_wol['PRECH'], errors='coerce')
                        combined_df['HP_WOL'] = pd.to_numeric(df_wol['HP'], errors='coerce')
                        combined_df['ZUFLUSS_WOL'] = pd.to_numeric(df_wol['ZUFLUSS'], errors='coerce')

                        # Add dp column for Node files as PRECH_WOL - PRECH_WL
                        combined_df['dp'] = combined_df['PRECH_WOL'] - combined_df['PRECH_WL']
                        self.logger.debug(f"Added dp column as PRECH_WOL - PRECH_WL for Node")

                        # Compute change in hydraulic head (delta_H)
                        combined_df['delta_H'] = combined_df['HP_WL'] - combined_df['HP_WOL']

                        # Ensure no infinite or NaN values in the calculations
                        node_calc_columns = ['PRECH_WL', 'HP_WL', 'ZUFLUSS_WL', 'PRECH_WOL',
                                             'HP_WOL', 'ZUFLUSS_WOL', 'dp', 'delta_H']
                        for col in node_calc_columns:
                            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

                    # Extract the base file name without "_with" and define output file name
                    base_name = re.sub(rf'_with_load_\d+_({file_type})\.csv', '', os.path.basename(wl_file))
                    output_file = os.path.join(zwischenspeicher_dir, f"{base_name}_{number}_combined_{file_type}.csv")

                    # Save the combined dataframe to a new CSV file
                    combined_df.to_csv(output_file, index=False, sep=';', decimal='.')
                    self.logger.info(f"Combined CSV file saved successfully to: {output_file}")

        except Exception as e:
            self.logger.error(f"Error combining files: {e}", exc_info=True)





