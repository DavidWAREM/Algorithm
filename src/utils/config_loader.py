# src/utils/config_loader.py
import yaml
import logging

def load_config(config_path, logger):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        logger (logging.Logger): Logger instance for logging errors.

    Returns:
        dict: Configuration parameters if successful, None otherwise.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}.")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}.")
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}.")
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}.")
    return None
