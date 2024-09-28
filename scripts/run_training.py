# scripts/run_training.py
import logging
import os
import torch
from torch_geometric.loader import DataLoader

from src.logging_config import setup_logging
from src.utils.config_loader import load_config
from src.data.data_preparation import DataModule
from src.models.model_setup import initialize_model, initialize_optimizer
from src.train.trainer import Trainer
from src.evaluation.evaluation import Evaluator
from sklearn.model_selection import train_test_split

def main():
    """
    Main function to orchestrate data loading, model training, evaluation, and saving.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting training process.")

    # Determine project root and config path
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        logger.info(f"Configuration path set to {config_path}.")
    except Exception as e:
        logger.error(f"Error determining project directories: {e}")
        return

    # Load configuration
    config = load_config(config_path, logger)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        return

    # Extract data path from config
    try:
        directory = config['paths']['folder_path_data']
        logger.info(f"Data directory set to {directory}.")
    except KeyError:
        logger.error("The key 'paths.folder_path_data' is missing in the configuration file.")
        return

    # Extract hyperparameters
    try:
        batch_size = int(config['hyperparameters']['batch_size'])
        hidden_dim = int(config['hyperparameters']['hidden_dim'])
        dropout = float(config['hyperparameters']['dropout'])
        learning_rate = float(config['hyperparameters']['learning_rate'])
        weight_decay = float(config['hyperparameters']['weight_decay'])
        num_epochs = int(config['hyperparameters']['num_epochs'])
        patience = int(config['hyperparameters']['patience'])
        lr_scheduler_factor = float(config['hyperparameters']['lr_scheduler_factor'])
        lr_scheduler_patience = int(config['hyperparameters']['lr_scheduler_patience'])
        logger.info("Hyperparameters extracted from configuration.")
    except KeyError as e:
        logger.error(f"Missing hyperparameter in configuration: {e}")
        return
    except ValueError as e:
        logger.error(f"Error converting hyperparameters: {e}")
        return

    # Extract node lists
    try:
        included_nodes = config['nodes']['included_nodes']
        zfluss_wl_nodes = config['nodes']['zfluss_wl_nodes']
        logger.info("Node lists extracted from configuration.")
    except KeyError as e:
        logger.error(f"Missing node parameter in configuration: {e}")
        return

    # Define feature columns
    geo_columns = ['XRECHTS', 'YHOCH', 'GEOH']
    adjusted_physical_columns = ['PRECH_WOL', 'PRECH_WL', 'HP_WL', 'HP_WOL', 'dp']
    additional_physical_columns = ['ZUFLUSS_WL']
    all_physical_columns = adjusted_physical_columns + additional_physical_columns

    # Initialize DataModule and prepare data
    try:
        data_module = DataModule(
            directory, all_physical_columns, geo_columns, included_nodes, zfluss_wl_nodes
        )
        nodes_df_first, edges_df_first, edge_columns = data_module.prepare()
        physical_scaler, geo_scaler, edge_scaler = data_module.get_scalers()
        logger.info("DataModule prepared the first dataset and fitted scalers.")
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        return

    # Load all datasets
    try:
        datasets = data_module.load_datasets(num_valves=108)
        if not datasets:
            logger.error("No valid datasets available. Please check the data.")
            return
        logger.info(f"Loaded {len(datasets)} datasets successfully.")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return

    # Split into training and validation sets
    try:
        train_data, val_data = train_test_split(datasets, test_size=0.2, random_state=42)
        logger.info(f"Data split into {len(train_data)} training and {len(val_data)} validation samples.")
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return

    # Check for positive samples
    try:
        train_positive = sum([data.y.sum().item() for data in train_data])
        val_positive = sum([data.y.sum().item() for data in val_data])
        if train_positive == 0 or val_positive == 0:
            logger.warning("No positive examples in training or validation sets. Please check the data split.")
            return
        logger.info(f"Training positive samples: {train_positive}, Validation positive samples: {val_positive}.")
    except Exception as e:
        logger.error(f"Error checking positive samples: {e}")
        return

    # Initialize model, optimizer, criterion, and scheduler
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = initialize_model(
            num_node_features=datasets[0].x.shape[1],
            num_edge_features=datasets[0].edge_attr.shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device
        )
        optimizer = initialize_optimizer(model, learning_rate, weight_decay)
        pos_weight = torch.tensor([len(train_data[0].edge_index[0]) - 1], dtype=torch.float).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
            # verbose=True  # Removed to avoid deprecated warning
        )
        logger.info("Model, optimizer, criterion, and scheduler initialized.")
    except Exception as e:
        logger.error(f"Error during model setup: {e}")
        return

    # Create DataLoaders
    try:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        logger.info("DataLoaders created successfully.")
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {e}")
        return

    # Initialize and run Trainer
    try:
        trainer = Trainer(
            model, optimizer, criterion, scheduler,
            device, num_epochs, patience, logger
        )
        model = trainer.train_model(train_loader, val_loader)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    # Initialize and run Evaluator
    try:
        evaluator = Evaluator(model, device, logger)
        test_loader = DataLoader(datasets, batch_size=batch_size, shuffle=False)
        auc_score, accuracy = evaluator.evaluate(test_loader)
        logger.info(f"Evaluation completed. AUC: {auc_score}, Accuracy: {accuracy}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

    # Save the trained model
    try:
        model_path = os.path.join(directory, 'edge_gat_model_classification.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved successfully at {model_path}.")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")

if __name__ == "__main__":
    main()
