import logging
import argparse
import yaml
import os
from src.logging_config import setup_logging
from src.data.data_load import CSVDataLoader
from src.data.data_preprocess import FeatureEngineer
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
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm to train (e.g., GCN, XGB, ANN)')
    parser.add_argument('--hyperparameter_search', action='store_true', help='Run hyperparameter search with Optuna')
    args = parser.parse_args()

    # Log the algorithm selected for training
    logger.info(f"Training with algorithm: {args.algorithm}")

    # Conditional logic based on the selected algorithm from the command-line arguments
    if args.algorithm == 'ANN':
        # Lade und verarbeite Daten für Artificial Neural Network (ANN)
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Feature Engineering durchführen
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Lade Hyperparameter aus der Konfigurationsdatei
        ann_config = config.get('ANN', {})
        learning_rate = ann_config.get('learning_rate', 0.001)
        epochs = ann_config.get('epochs', 1000)
        batch_size = ann_config.get('batch_size', 32)
        patience = ann_config.get('patience', 10)

        logger.info(f"ANN Hyperparameters: Learning Rate={learning_rate}, Epochs={epochs}, Batch Size={batch_size}, Patience={patience}")

        # Initialisiere und trainiere das ANN-Modell
        model = ANNModel(input_shape=X_train.shape[1], learning_rate=learning_rate)  # Input shape basierend auf der Anzahl der Features
        history = model.train(
            X_train, y_train,
            validation_split=ann_config.get('validation_split', 0.2),
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        model.save_model(file_name=ann_config.get('model_filename', "ann_model_complex.h5"))  # Modell speichern

        # Evaluieren und visualisieren des ANN-Modells
        ANNModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'XGB':
        # Lade und verarbeite Daten für XGBoost
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Feature Engineering durchführen
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Initialisiere und trainiere das XGBoost-Modell
        model = XGBoostModel()
        model.train(X_train, y_train)
        model.save_model()  # Modell speichern

        # Evaluieren und visualisieren des XGBoost-Modells
        XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == 'XGB_Hyperparameter':
        # Lade und verarbeite Daten für XGBoost mit Hyperparameter-Tuning
        data_loader = CSVDataLoader(config_file=config_path)
        all_data = data_loader.get_data()

        # Feature Engineering durchführen
        feature_engineer = FeatureEngineer(all_data)
        feature_engineer.process_features()
        X_train, X_test, y_train, y_test = feature_engineer.get_processed_data()

        # Führe Hyperparameter-Tuning, Training und Evaluierung des XGBoost-Modells durch
        model = XGBoostModel()
        if args.hyperparameter_search:
            model.hyperparameter_tuning(X_train, y_train)  # Hyperparameter tunen
        model.train(X_train, y_train)
        model.save_model()  # Modell speichern

        # Evaluieren und visualisieren des XGBoost-Modells mit getunten Hyperparametern
        XGBoostModelEvaluator.evaluate_and_visualize(X_test, y_test)

    elif args.algorithm == "GAT":
        gat_main()

    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()
