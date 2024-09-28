# src/utils/config_loader.py
import yaml
import logging

def load_config(config_path, logger):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            logger.error(f"Fehler beim Laden der Konfigurationsdatei: {exc}")
            return None
