# src/core/config.py
import yaml
import logging 
from typing import Any, Dict, Optional

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class SimpleConfigLoader:
    def __init__(self, config_file_path: str):
        self.config_file_path: str = config_file_path
        self._config_data: Dict[str, Any] = {}
        self._load_config()
        logger.info(f"ConfigLoader initialized. Loaded data from '{config_file_path}'.")

    def _load_config(self):
        try:
            with open(self.config_file_path, 'r') as f:
                self._config_data = yaml.safe_load(f)
            if self._config_data is None: 
                self._config_data = {}
                logger.warning(f"Configuration file '{self.config_file_path}' is empty or contains no valid YAML. Using empty configuration.")
            logger.info(f"Full configuration data loaded by SimpleConfigLoader: {self._config_data}") 
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file_path}")
            raise ConfigurationError(f"Configuration file not found: {self.config_file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file '{self.config_file_path}': {e}")
            raise ConfigurationError(f"Error parsing YAML configuration file: {e}")
        except Exception as e: 
            logger.error(f"An unexpected error occurred while loading config file '{self.config_file_path}': {e}")
            raise ConfigurationError(f"Unexpected error loading configuration: {e}")

    def get(self, config_key: str, default_value: Any = None) -> Any:
        # THIS IS THE MOST IMPORTANT LOG LINE TO LOOK FOR
        logger.info(f"[[[[[ SimpleConfigLoader.get() CALLED for key: '{config_key}' (type: {type(config_key)}) ]]]]]")

        if self._config_data is None: 
            logger.warning("SimpleConfigLoader.get(): _config_data is None. Returning default_value.")
            return default_value
        
        # Log all top-level keys available at the time of the call for this specific key
        logger.info(f"SimpleConfigLoader.get(): Available top-level keys in _config_data: {list(self._config_data.keys())}")

        try:
            if '.' not in config_key:
                logger.info(f"SimpleConfigLoader.get(): Processing as TOP-LEVEL key: '{config_key}'")
                
                # Explicitly check if the key exists
                if config_key in self._config_data:
                    value = self._config_data[config_key] # Direct access since we know it exists
                    logger.info(f"SimpleConfigLoader.get(): Key '{config_key}' FOUND in _config_data. Value type: {type(value)}. Returning value.")
                    return value
                else:
                    logger.info(f"SimpleConfigLoader.get(): Top-level key '{config_key}' NOT IN _config_data. Returning default_value.")
                    return default_value
            else: 
                logger.info(f"SimpleConfigLoader.get(): Processing as NESTED key: '{config_key}'")
                keys = config_key.split('.')
                current_level_data = self._config_data
                for i, key_part in enumerate(keys):
                    if isinstance(current_level_data, dict):
                        if key_part in current_level_data:
                            current_level_data = current_level_data[key_part]
                        else:
                            logger.info(f"SimpleConfigLoader.get(): Nested key part '{key_part}' (index {i} of '{config_key}') not found. Returning default_value.")
                            return default_value
                    else: 
                        logger.info(f"SimpleConfigLoader.get(): Expected dict for key part '{key_part}' in '{config_key}', but found {type(current_level_data)}. Returning default_value.")
                        return default_value
                logger.info(f"SimpleConfigLoader.get(): Successfully retrieved nested key '{config_key}'. Value type: {type(current_level_data)}")
                return current_level_data
        except Exception as e: 
            logger.error(f"SimpleConfigLoader.get(): Unexpected error accessing config key '{config_key}': {e}", exc_info=True)
            return default_value # Ensure default is returned on any unexpected error

    def get_all_config(self) -> Dict[str, Any]:
        return self._config_data

    def reload_config(self):
        logger.info(f"Reloading configuration from '{self.config_file_path}'...")
        self._load_config()
        logger.info("Configuration reloaded.")

