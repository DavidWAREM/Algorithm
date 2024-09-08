import logging
import argparse
import yaml
import os
from src.logging_config import setup_logging
from src.data.data_load import CSVDataLoader
from src.data.data_preprocess import FeatureEngineer
from src.models.train_GBR import GradientBoostingModel
from src.evaluation.evaluation_GBR import GBRModelEvaluator
from src.models.train_ANN import ANNModel
from src.evaluation.evaluation_ANN import ANNModelEvaluator
from src.data.datapreperation_GNN import GraphDataset
from src.models.train_GNN import GNNModel
from src.models.train_XGB import XGBoostModel  # Import für XGBoost-Modell
from src.evaluation.evaluate_XGB import XGBoostModelEvaluator  # Import für XGBoost-Modellevaluierung

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Absolute path to the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    # Load configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to train (e.g., gradient_boosting, XGB, XGB Hyperparameter)')
    args = parser.parse_args()

    logger.info(f"Training with algorithm: {args.algorithm}")

    if args.algorithm == 'gradient_boosting':
        # Load and preprocess data
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        model = GradientBoostingModel()
        model.train(X_train, y_train)
        model.save_model()

        # Evaluate the model
        GBRModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'ANN':
        # Load and preprocess data
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        model = ANNModel(input_shape=X_train.shape[1])
        model.train(X_train, y_train)
        model.save_model()

        # Evaluate the model
        ANNModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'GNN':
        folder_path_data = config['paths']['folder_path_data']
        folder_path_data_GNN_dataset = config['paths']['folder_path_data_GNN_dataset']
        graph_dataset = GraphDataset(folder_path_data, folder_path_data_GNN_dataset)
        gnn_model = GNNModel(folder_path=folder_path_data, save_path=folder_path_data_GNN_dataset)
        gnn_model.run_training()
        gnn_model.evaluate()
        gnn_model.save_model()

    elif args.algorithm == 'XGB':
        # Load and preprocess data for XGBoost
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Train the XGBoost model
        model = XGBoostModel()
        model.train(X_train, y_train)
        model.save_model()

        # Evaluate the XGBoost model
        XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'XGB Hyperparameter':
        # Load and preprocess data for XGBoost with hyperparameter tuning
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Perform hyperparameter tuning and train the XGBoost model
        model = XGBoostModel()
        model.hyperparameter_tuning(X_train, y_train)
        model.train(X_train, y_train)
        model.save_model()

        # Evaluate the XGBoost model
        XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)

    logger.info("Training and evaluation completed successfully")

if __name__ == '__main__':
    main()
