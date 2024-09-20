import os
import logging.config
import yaml

def setup_logging(
        default_path='config/logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """
    Setup logging configuration.
    This function reads the logging configuration from a YAML file and applies it.
    """
    # Prevent double logging setup
    if len(logging.getLogger().handlers) > 0:
        return

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    # Load YAML logging configuration
    if os.path.exists(path):
        with open(path, 'rt') as f:
            logging_config = yaml.safe_load(f.read())
        logging.config.dictConfig(logging_config)
    else:
        # Default logging setup if the YAML config file is missing
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize logging as soon as this module is imported
setup_logging()
