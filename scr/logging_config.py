import os
import logging.config
import yaml

def load_main_config(config_file='config/config.yaml'):
    """
    Load the main configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    config_path = os.path.join(project_root, config_file)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(
    default_path='config/logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """
    Setup logging configuration.

    Parameters:
    default_path (str): The default path to the logging configuration file.
    default_level (int): The default logging level if no configuration file is found.
    env_key (str): The environment variable that can override the default path to the logging configuration file.
    """
    config = load_main_config()
    log_file_path = config['paths']['log_file']
    log_dir = os.path.dirname(log_file_path)

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            logging_config = yaml.safe_load(f.read())
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Update the handlers to use the log file path from the main config
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.baseFilename = log_file_path

# Initialize logging as soon as this module is imported
setup_logging()
