import os
import pandas as pd
import numpy as np
import logging
from scipy.stats import skew, kurtosis
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Correct import
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to set up logging
def setup_logging(directory):
    log_file = os.path.join(directory, 'analysis.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all previous handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a FileHandler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Define the format
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger

# Function to load and analyze data
def load_and_analyze_data(directory, logger):
    master_data = []

    # Define the relevant features for Pipes and Nodes, including the new physical variables
    pipe_features = [
        'DM', 'RAU', 'RAISE', 'VM_WL', 'VM_WOL', 'FLUSS_WL', 'FLUSS_WOL', 'RORL',
        'RE_WL', 'RE_WOL'
    ]

    node_features = [
        'PRECH_WL', 'PRECH_WOL', 'HP_WL', 'HP_WOL', 'ZUFLUSS_WL', 'ZUFLUSS_WOL',
        'dp'
    ]

    # Use glob to find all relevant files
    node_pattern = os.path.join(directory, '*_combined_Node.csv')
    pipe_pattern = os.path.join(directory, '*_combined_Pipes.csv')

    node_files = glob.glob(node_pattern)
    pipe_files = glob.glob(pipe_pattern)

    logger.info(f"Found Node files: {len(node_files)}")
    logger.info(f"Found Pipe files: {len(pipe_files)}")

    # Extract base names to pair Node and Pipe files
    node_bases = set([os.path.basename(f).replace('_combined_Node.csv', '') for f in node_files])
    pipe_bases = set([os.path.basename(f).replace('_combined_Pipes.csv', '') for f in pipe_files])

    # Find common bases
    common_bases = node_bases.intersection(pipe_bases)
    logger.info(f"Found matching pairs: {len(common_bases)}")

    # Iterate through common bases and load corresponding files
    for base in common_bases:
        node_file = os.path.join(directory, f"{base}_combined_Node.csv")
        pipe_file = os.path.join(directory, f"{base}_combined_Pipes.csv")

        try:
            # Load Pipes data
            pipes_df = pd.read_csv(pipe_file, delimiter=';', decimal='.')
            logger.info(f"Loaded file {pipe_file}")

            # Check if all required pipe features are present
            missing_pipe_cols = [col for col in pipe_features if col not in pipes_df.columns]
            if missing_pipe_cols:
                logger.error(f"Missing Pipe columns in {pipe_file}: {missing_pipe_cols}.")
                continue

            # Extract relevant Pipe features
            pipes_data = pipes_df[pipe_features].copy()

            # Load Nodes data
            nodes_df = pd.read_csv(node_file, delimiter=';', decimal='.')
            logger.info(f"Loaded file {node_file}")

            # Check if all required node features are present
            missing_node_cols = [col for col in node_features if col not in nodes_df.columns]
            if missing_node_cols:
                logger.error(f"Missing Node columns in {node_file}: {missing_node_cols}.")
                continue

            # Extract relevant Node features
            nodes_data = nodes_df[node_features].copy()

            # Combine the data
            combined_data = pd.concat([pipes_data, nodes_data], axis=1)

            # Append the combined data to the list
            master_data.append(combined_data)

        except Exception as e:
            logger.error(f"Error processing {base}: {e}")
            continue

    # Combine all data into a single DataFrame
    if not master_data:
        logger.error("No data found for analysis.")
        return None

    master_df = pd.concat(master_data, ignore_index=True)
    logger.info(f"Total data loaded: {master_df.shape[0]} rows.")

    # Select only numerical columns for analysis
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()

    # Drop columns with more than 50% missing values
    master_df = master_df.dropna(axis=1, thresh=int(0.5 * master_df.shape[0]))
    logger.info(f"Columns after dropping those with >50% missing values: {master_df.shape[1]}")

    # Fill remaining missing values with mean of the column
    master_df = master_df.fillna(master_df.mean())

    # Ensure no inf values are present
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if master_df.isnull().values.any():
        master_df = master_df.fillna(master_df.mean())
        logger.warning("Replaced inf values with NaN and filled remaining NaNs with column means.")

    return master_df

# Function to create the analysis report
def save_analysis_report(df, directory, logger):
    report_path = os.path.join(directory, 'RAU_Analysis_Report.txt')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        # Descriptive statistics for all features
        report_file.write("### Descriptive Statistics for All Numerical Features:\n")
        desc_stats = df.describe().transpose()
        report_file.write(str(desc_stats) + "\n\n")

        # Skewness and kurtosis for all features
        report_file.write("### Skewness and Kurtosis for All Numerical Features:\n")
        skew_kurt = pd.DataFrame({
            'Skewness': df.apply(lambda x: skew(x.dropna())),
            'Kurtosis': df.apply(lambda x: kurtosis(x.dropna()))
        })
        report_file.write(str(skew_kurt) + "\n\n")

        # Correlation matrix focused on RAU
        report_file.write("### Correlation Matrix with RAU:\n")
        corr_matrix = df.corr()
        rau_corr = corr_matrix[['RAU']].sort_values(by='RAU', ascending=False)
        report_file.write(str(rau_corr) + "\n\n")

        # Non-linear transformations and their correlations with RAU
        report_file.write("### Non-linear Transformations and Correlations with RAU:\n")
        transformations = ['square', 'sqrt', 'log']
        transformed_corrs = pd.DataFrame()

        for col in df.columns:
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
                    logger.warning(f"Transformation {trans} could not be applied to column {col}: {e}")

        # Sort correlations by absolute value
        transformed_corrs = transformed_corrs.sort_values(by='Correlation', key=lambda x: x.abs(), ascending=False)

        # Write transformed correlations to the report
        report_file.write("Correlation coefficients of transformed variables with RAU:\n")
        if not transformed_corrs.empty:
            report_file.write(f"{'Variable':<30} {'Correlation':>12}\n")
            report_file.write(f"{'-'*30} {'-'*12}\n")
            for index, row in transformed_corrs.iterrows():
                report_file.write(f"{index:<30} {row['Correlation']:>12.4f}\n")
        else:
            report_file.write("No transformed variables found.\n")
        report_file.write("\n")

        # Outlier detection
        report_file.write("### Outlier Detection:\n")
        outliers = detect_outliers(df, logger)
        report_file.write(f"Number of outliers detected: {len(outliers)}\n")
        report_file.write("\n")

        # Multicollinearity analysis
        report_file.write("### Multicollinearity Analysis (VIF):\n")
        # Clean data for VIF: replace inf with NaN and fill remaining NaNs with mean
        clean_df = df.drop(columns=['RAU']).replace([np.inf, -np.inf], np.nan)
        if clean_df.isnull().values.any():
            clean_df = clean_df.fillna(clean_df.mean())
            logger.warning("Filled remaining NaNs after cleaning for VIF calculation.")

        vif_data = calculate_vif(clean_df)
        report_file.write(str(vif_data) + "\n\n")

        # Recommendations
        report_file.write("### Recommendations:\n")
        recommendations = provide_recommendations(df, transformed_corrs, vif_data)
        for rec in recommendations:
            report_file.write(f"- {rec}\n")

    logger.info(f"Analysis report saved at: {report_path}")

    # Generate plots
    generate_plots(df, directory, logger)

# Function to detect outliers using the IQR method
def detect_outliers(df, logger):
    outlier_indices = []
    for col in df.columns:
        if col == 'RAU':
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_col = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].index
        outlier_indices.extend(outliers_col)
    outlier_indices = list(set(outlier_indices))
    logger.info(f"Detected {len(outlier_indices)} outliers.")
    return outlier_indices

# Function to calculate Variance Inflation Factor (VIF) for multicollinearity analysis
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

# Function to generate plots for visual analysis
def generate_plots(df, directory, logger):
    plots_dir = os.path.join(directory, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Histograms for RAU and features
    for col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(os.path.join(plots_dir, f'histogram_{col}.png'))
        plt.close()

    # Scatter plots of features vs RAU
    for col in df.columns:
        if col == 'RAU':
            continue
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[col], y=df['RAU'])
        plt.title(f'{col} vs RAU')
        plt.xlabel(col)
        plt.ylabel('RAU')
        plt.savefig(os.path.join(plots_dir, f'scatter_{col}_vs_RAU.png'))
        plt.close()

    logger.info(f"Generated plots saved in {plots_dir}")

# Function to generate recommendations
def provide_recommendations(df, transformed_corrs, vif_data):
    recommendations = []

    # Skewness of RAU
    skew_val = skew(df['RAU'].dropna())
    if abs(skew_val) > 1:
        recommendations.append("The RAU distribution is highly skewed. Consider applying a transformation.")
    elif abs(skew_val) > 0.5:
        recommendations.append("The RAU distribution is moderately skewed. A transformation might be helpful.")
    else:
        recommendations.append("The RAU distribution is approximately symmetric.")

    # High correlation features with RAU
    corr_matrix = df.corr()
    rau_corr = corr_matrix['RAU'].drop('RAU')
    high_corr_features = rau_corr[abs(rau_corr) >= 0.3].index.tolist()
    if high_corr_features:
        recommendations.append(f"The following features have a high correlation with RAU: {', '.join(high_corr_features)}")
    else:
        recommendations.append("No features with high correlation to RAU in the original variables found.")

    # High correlation transformed features
    high_corr_transformed = transformed_corrs[abs(transformed_corrs['Correlation']) >= 0.3].index.tolist()
    if high_corr_transformed:
        recommendations.append(f"After applying non-linear transformations, the following variables have a high correlation with RAU: {', '.join(high_corr_transformed)}")
    else:
        recommendations.append("No strong correlations found after non-linear transformations.")

    # Multicollinearity check
    high_vif = vif_data[vif_data['VIF'] > 5]['Feature'].tolist()
    if high_vif:
        recommendations.append(f"The following features show multicollinearity (VIF > 5): {', '.join(high_vif)}. Consider removing or combining these features.")
    else:
        recommendations.append("No significant multicollinearity detected among features.")

    # Outlier presence
    outlier_count = len(detect_outliers(df, logging.getLogger()))
    if outlier_count > 0:
        recommendations.append(f"Outliers detected in the dataset ({outlier_count} outliers). Consider handling them appropriately.")
    else:
        recommendations.append("No significant outliers detected in the dataset.")

    # Feature selection suggestion
    if not high_corr_features and not high_corr_transformed:
        recommendations.append("Consider adding more features or performing feature engineering to improve the model.")
    else:
        recommendations.append("Proceed with modeling using the identified significant features.")

    return recommendations

# Main function
def main():
    # Adjust the path to your data directory
    directory = 'C:\\Users\\D.Muehlfeld\\Documents\Synthetic_Data\\Synthetic_Data_Roughness_Simplification^3(0;100)\\Zwischenspeicher'

    # Set up logging
    logger = setup_logging(directory)

    # Load and analyze data
    df = load_and_analyze_data(directory, logger)
    if df is None:
        logger.error("Analysis aborted.")
        return

    # Save analysis report
    save_analysis_report(df, directory, logger)

    logger.info("Analysis completed.")

if __name__ == "__main__":
    main()
