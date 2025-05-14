# src/core/config.py
import yaml
import os
import logging # Import logging
from .exceptions import ConfigurationError # Import our custom exception

logger = logging.getLogger(__name__) # Get a logger for this module

class SimpleConfigLoader:
    def __init__(self, config_file_path: str = "config/config.yaml"):
        self._config = {}
        self._config_file_path = config_file_path
        try:
            self._load_config()
        except ConfigurationError as e:
            # Log the configuration error and re-raise it or handle as appropriate
            # For now, we'll let it propagate, which will likely stop the app
            # if config is essential.
            logger.critical(f"Failed to initialize SimpleConfigLoader: {e}")
            raise # Re-raise the caught ConfigurationError

    def _load_config(self):
        if not os.path.exists(self._config_file_path):
            msg = f"Configuration file '{self._config_file_path}' not found."
            logger.error(msg)
            raise ConfigurationError(msg) # Raise our custom error

        try:
            with open(self._config_file_path, 'r') as f:
                self._config = yaml.safe_load(f)
            if self._config is None: # Handle empty YAML file
                self._config = {}
                logger.warning(f"Configuration file '{self._config_file_path}' is empty.")
        except yaml.YAMLError as e:
            msg = f"Error parsing YAML file '{self._config_file_path}': {e}"
            logger.error(msg)
            raise ConfigurationError(msg) from e # Chain the original YAMLError
        except Exception as e: # Catch any other unexpected errors during file loading
            msg = f"An unexpected error occurred while loading config file '{self._config_file_path}': {e}"
            logger.error(msg)
            raise ConfigurationError(msg) from e

    def get(self, key: str, default: any = None) -> any:
        # ... (get method remains the same)
        try:
            value = self._config
            for part in key.split('.'):
                if isinstance(value, dict):
                    value = value[part]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, key: str) -> dict:
        # ... (get_section method remains the same)
        section = self.get(key, {})
        return section if isinstance(section, dict) else {}

    def reload_config(self):
        logger.info(f"Attempting to reload configuration from {self._config_file_path}")
        try:
            self._load_config()
            logger.info(f"Successfully reloaded configuration from {self._config_file_path}")
        except ConfigurationError as e:
            logger.error(f"Failed to reload configuration: {e}")
            # Decide if reload failure should re-raise or just log
            # For now, just logging the error during reload.
