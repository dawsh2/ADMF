# src/core/config.py
import yaml
import logging 
from typing import Any, Dict, Optional
import copy # Import copy module

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class SimpleConfigLoader:
    def __init__(self, config_file_path: str):
        self.config_file_path: str = config_file_path
        self._raw_config_data: Optional[Dict[str, Any]] = None # Store raw load
        self._config_data: Dict[str, Any] = {} # This will be the shielded deep copy
        
        # Only use debug level for detailed information to avoid cluttering console
        logger.debug(f"SimpleConfigLoader instance {id(self)} creating for {config_file_path}.")
        self._load_and_shield_config() # Changed method name
        logger.debug(f"ConfigLoader {id(self)} initialized. Shielded config ready. Loaded from '{config_file_path}'.")

    def _load_and_shield_config(self): # Renamed
        """Loads the configuration and immediately creates a deep copy for internal use."""
        raw_data_before_load: Optional[Dict[str, Any]] = None
        try:
            with open(self.config_file_path, 'r') as f:
                raw_data_before_load = yaml.safe_load(f) # Load into a temporary variable
            
            if raw_data_before_load is None: 
                self._raw_config_data = {} # Store the (potentially empty) raw data
                logger.warning(f"Configuration file '{self.config_file_path}' is empty or contains no valid YAML. Raw config is empty.")
            else:
                self._raw_config_data = raw_data_before_load # Store the raw data
            
            # Create a deep copy for all internal operations of this loader
            self._config_data = copy.deepcopy(self._raw_config_data) 
            
            # Use debug level for detailed config dumps to avoid cluttering console
            logger.debug(f"ConfigLoader {id(self)}: Raw configuration data loaded successfully")
            logger.debug(f"ConfigLoader {id(self)}: Shielded _config_data created successfully")

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file_path}")
            self._raw_config_data = {} # Ensure raw_config_data is an empty dict on error
            self._config_data = {} # Ensure _config_data is an empty dict on error
            raise ConfigurationError(f"Configuration file not found: {self.config_file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file '{self.config_file_path}': {e}")
            self._raw_config_data = {}
            self._config_data = {}
            raise ConfigurationError(f"Error parsing YAML configuration file: {e}")
        except Exception as e: 
            logger.error(f"An unexpected error occurred while loading config file '{self.config_file_path}': {e}")
            self._raw_config_data = {}
            self._config_data = {}
            raise ConfigurationError(f"Unexpected error loading configuration: {e}")

    def get(self, config_key: str, default_value: Any = None) -> Any:
        # This log will now show the state of the *shielded* _config_data
        logger.debug(f"ConfigLoader {id(self)}: SimpleConfigLoader.get() called for key: '{config_key}'")

        if self._config_data is None: 
            logger.warning(f"ConfigLoader {id(self)}: SimpleConfigLoader.get(): _config_data (shielded copy) is None. This should not happen. Returning default_value.")
            return default_value # Should not be None if __init__ is robust
        
        logger.debug(f"ConfigLoader {id(self)}: Accessing _config_data with available top-level keys: {list(self._config_data.keys())}")

        try:
            if '.' not in config_key:
                logger.debug(f"ConfigLoader {id(self)}: Processing as top-level key: '{config_key}'")
                if config_key in self._config_data: # Check in the shielded copy
                    value = self._config_data[config_key]
                    logger.debug(f"ConfigLoader {id(self)}: Key '{config_key}' found. Returning value.")
                    if isinstance(value, (dict, list)):
                        return copy.deepcopy(value) # Return a copy to the caller
                    return value
                else:
                    logger.debug(f"ConfigLoader {id(self)}: Key '{config_key}' not found. Returning default_value.")
                    return default_value
            else: 
                logger.debug(f"ConfigLoader {id(self)}: Processing as nested key: '{config_key}'")
                keys = config_key.split('.')
                current_level_data = self._config_data # Start with the shielded copy
                for i, key_part in enumerate(keys):
                    if isinstance(current_level_data, dict):
                        if key_part in current_level_data:
                            current_level_data = current_level_data[key_part]
                        else:
                            logger.debug(f"ConfigLoader {id(self)}: Nested key part '{key_part}' not found. Returning default_value.")
                            return default_value
                    else: 
                        logger.debug(f"ConfigLoader {id(self)}: Expected dict for key part '{key_part}', but found {type(current_level_data)}. Returning default_value.")
                        return default_value
                logger.debug(f"ConfigLoader {id(self)}: Successfully retrieved nested key '{config_key}'")
                if isinstance(current_level_data, (dict, list)):
                    return copy.deepcopy(current_level_data) # Return a copy to the caller
                return current_level_data
        except Exception as e: 
            logger.error(f"ConfigLoader {id(self)}: Unexpected error accessing config key '{config_key}': {e}", exc_info=True)
            return default_value

    def get_all_config(self) -> Dict[str, Any]:
        """Returns a deep copy of the entire (shielded) configuration data."""
        if self._config_data is None:
            return {}
        return copy.deepcopy(self._config_data) # Return a copy of the shielded copy

    def reload_config(self):
        logger.debug(f"Reloading configuration from '{self.config_file_path}'...")
        self._load_and_shield_config() # Use the shielding method on reload too
        logger.debug("Configuration reloaded.")