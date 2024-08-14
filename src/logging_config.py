import os
import logging.config
import yaml
import inspect


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
        default_level=logging.DEBUG,
        env_key='LOG_CFG'
):
    """
    Setup logging configuration.

    Parameters:
    default_path (str): The default path to the logging configuration file.
    default_level (int): The default logging level if no configuration file is found.
    env_key (str): The environment variable that can override the default path to the logging configuration file.
    """
    # Get the name of the calling file
    calling_file = inspect.stack()[1].filename
    calling_file_name = os.path.splitext(os.path.basename(calling_file))[0]

    # Create the log file path
    log_file_path = os.path.join(os.path.dirname(calling_file), f'{calling_file_name}_logging.log')

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Prevent double logging setup
    if len(logging.getLogger().handlers) > 0:
        return

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            logging_config = yaml.safe_load(f.read())
        logging.config.dictConfig(logging_config)

        # Update the handlers to use the log file path
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.baseFilename = log_file_path
    else:
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


# Initialize logging as soon as this module is imported
setup_logging()
