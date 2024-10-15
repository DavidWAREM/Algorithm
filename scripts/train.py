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

    elif args.algorithm == 'XGB_Hyperparameter':
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

    elif args.algorithm == "GCN":
        folder_path_data = config['paths']['folder_path_data']

        # Initialize data loader and load all datasets
        data_loader = GCNDataLoader(folder_path_data, num_datasets=10)
        datasets = data_loader.load_all_data()

        # Initialize the preprocessor (with scaling of both node and edge data)
        preprocessor = GCNDataPreprocessor()
        train_data, test_data = preprocessor.split_data(datasets, test_size=0.2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Debugging: print structure of train_data
        print(f"Structure of train_data[0]: {train_data[0]}")




        if args.hyperparameter_search:
            def objective(trial):
                hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
                lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
                weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

                # Initialize model with trial parameters and pass num_edge_features
                model = GCNModel(num_node_features=train_data[0][0].x.shape[1], num_edge_features=num_edge_features,
                                 output_dim=1, hidden_dim=hidden_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                # Train and test model
                trainer = GCNTrainer(model=model, optimizer=optimizer, device=device)
                trainer.train_model(train_data)
                tester = GCNTester(model=model, device=device)
                total_mse, rmse, r2 = tester.test_model(test_data)
                return total_mse

            # Run hyperparameter tuning with Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)

            logger.info(f"Best trial: {study.best_params}")
            best_params = study.best_params

            # Use the best hyperparameters for final evaluation
            model = GCNModel(num_node_features=train_data[0][0].x.shape[1], num_edge_features=num_edge_features,
                             output_dim=1, hidden_dim=best_params['hidden_dim'])
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'],
                                         weight_decay=best_params['weight_decay'])

        else:

            num_edge_features = train_data[0][1]# Use default hyperparameters
            model = GCNModel(num_node_features=train_data[0][0].x.shape[1], num_edge_features=num_edge_features,
                             output_dim=1, hidden_dim=27)
            optimizer = torch.optim.Adam(model.parameters(), lr=9.63e-05, weight_decay=6.81e-05)

        # Train the model
        trainer = GCNTrainer(model=model, optimizer=optimizer, device=device)
        trainer.train_model(train_data)

        # Test and evaluate the model
        tester = GCNTester(model=model, device=device)
        total_mse, rmse, r2 = tester.test_model(test_data)
        logger.info(f'Total MSE: {total_mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()
