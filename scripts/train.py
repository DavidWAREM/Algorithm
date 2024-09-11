import logging
import argparse
import yaml
import os
from src.logging_config import setup_logging  # Custom logging setup
from src.data.data_load import CSVDataLoader  # Custom class for loading CSV data
from src.data.data_preprocess import FeatureEngineer  # Feature engineering utility for data preprocessing
from src.models.train_GBR import GradientBoostingModel  # Gradient Boosting model class for training
from src.evaluation.evaluation_GBR import GBRModelEvaluator  # Model evaluator for Gradient Boosting
from src.models.train_ANN import ANNModel  # ANN model class for training
from src.evaluation.evaluation_ANN import ANNModelEvaluator  # Model evaluator for ANN
from src.data.datapreperation_GNN import GraphDataset  # Graph data preparation for GNN
from src.models.train_GNN import GNNModel  # GNN model class for training
from src.models.train_XGB import XGBoostModel  # XGBoost model class for training
from src.evaluation.evaluaten_XGB import XGBoostModelEvaluator  # XGBoost model evaluator


def main():
    """
    Main function for running the training process for various machine learning models.

    This function sets up logging, loads the configuration file, and parses the command-line arguments
    to select which algorithm (e.g., Gradient Boosting, ANN, GNN, XGBoost) will be trained and evaluated.
    """
    setup_logging()  # Initialize logging configuration
    logger = logging.getLogger(__name__)  # Create a logger for this script

    # Get the absolute path to the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))  # Move up one level to the project root
    config_path = os.path.join(project_root, 'config', 'config.yaml')  # Path to the configuration file

    # Load the YAML configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Parse command-line arguments to choose the algorithm for training
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--algorithm', type=str, required=True,
                        help='Algorithm to train (e.g., gradient_boosting, XGB, XGB Hyperparameter)')
    args = parser.parse_args()

    # Log the algorithm selected for training
    logger.info(f"Training with algorithm: {args.algorithm}")

    # Conditional logic based on the selected algorithm from the command-line arguments
    if args.algorithm == 'gradient_boosting':
        # Load and preprocess data for Gradient Boosting
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Perform feature engineering
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Initialize and train the Gradient Boosting model
        model = GradientBoostingModel()
        model.train(X_train, y_train)
        model.save_model()  # Save the trained model

        # Evaluate the Gradient Boosting model and visualize results
        GBRModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'ANN':
        # Load and preprocess data for Artificial Neural Network (ANN)
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Perform feature engineering
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Initialize and train the ANN model
        model = ANNModel(input_shape=X_train.shape[1])  # Input shape based on the number of features
        model.train(X_train, y_train)
        model.save_model()  # Save the trained model

        # Evaluate the ANN model and visualize results
        ANNModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'GNN':
        # Train a Graph Neural Network (GNN)
        folder_path_data = config['paths']['folder_path_data']
        folder_path_data_GNN_dataset = config['paths']['folder_path_data_GNN_dataset']

        # Prepare the graph dataset
        graph_dataset = GraphDataset(folder_path_data, folder_path_data_GNN_dataset)

        # Initialize, train, and evaluate the GNN model
        gnn_model = GNNModel(folder_path=folder_path_data, save_path=folder_path_data_GNN_dataset)
        gnn_model.run_training()
        gnn_model.evaluate()
        gnn_model.save_model()

    elif args.algorithm == 'XGB':
        # Load and preprocess data for XGBoost
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Perform feature engineering
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Initialize and train the XGBoost model
        model = XGBoostModel()
        model.train(X_train, y_train)
        model.save_model()  # Save the trained model

        # Evaluate the XGBoost model and visualize results
        XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'XGB Hyperparameter':
        # Load and preprocess data for XGBoost with hyperparameter tuning
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Perform feature engineering
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Perform hyperparameter tuning, train, and evaluate the XGBoost model
        model = XGBoostModel()
        model.hyperparameter_tuning(X_train, y_train)  # Tune the hyperparameters
        model.train(X_train, y_train)
        model.save_model()  # Save the trained model

        # Evaluate the XGBoost model with tuned hyperparameters
        XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)

    # Log the completion of training and evaluation
    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()  # Entry point: run the main function if this script is executed directly
