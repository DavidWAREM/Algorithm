import logging
import argparse
import yaml
import os
import torch
import optuna
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.logging_config import setup_logging
from src.data.data_load import CSVDataLoader
from src.data.data_preprocess import FeatureEngineer
from src.models.train_GBR import GradientBoostingModel
from src.evaluation.evaluation_GBR import GBRModelEvaluator
from src.models.train_ANN import ANNModel
from src.evaluation.evaluation_ANN import ANNModelEvaluator
from src.models.train_XGB import XGBoostModel
from src.evaluation.evaluation_XGB import XGBoostModelEvaluator
from src.models.train_GAT import main as gat_main


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Verwende einen einzigen Argumentparser
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to train (e.g., GCN, XGB)')
    parser.add_argument('--hyperparameter_search', action='store_true', help='Run hyperparameter search with Optuna')
    args = parser.parse_args()

    # Log the algorithm selected for training
    logger.info(f"Training with algorithm: {args.algorithm}")

    # Conditional logic based on the selected algorithm from the command-line arguments
    if args.algorithm == 'GBR':
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

    elif args.algorithm == "GAT":
        gat_main()


    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()
