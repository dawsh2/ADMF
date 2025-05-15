# src/core/component.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger_component_base = logging.getLogger(__name__) # Module-level logger for this file

class BaseComponent(ABC):
    STATE_CREATED = "CREATED"
    STATE_INITIALIZED = "INITIALIZED"
    STATE_STARTED = "STARTED"
    STATE_STOPPED = "STOPPED"
    STATE_FAILED = "FAILED"

    def __init__(self, 
                 instance_name: str, 
                 config_loader: Any, 
                 component_config_key: Optional[str] = None):
        self.name: str = instance_name
        self.state: str = BaseComponent.STATE_CREATED
        self.logger = logging.getLogger(f"component.{self.name}") 

        self._config_loader = config_loader
        self._component_config_key: Optional[str] = component_config_key
        self.component_specific_config: Dict[str, Any] = {}

        if self._config_loader and self._component_config_key:
            # --- ADD INFO LOGGING FOR THE KEY BEING USED ---
            self.logger.info(f"Attempting to load specific config for '{self.name}' using key: '{self._component_config_key}'")
            # -------------------------------------------------
            loaded_config = self._config_loader.get(self._component_config_key)
            
            if isinstance(loaded_config, dict):
                self.component_specific_config = loaded_config
                self.logger.info(f"Specific configuration loaded for '{self.name}' using key '{self._component_config_key}'. Content: {self.component_specific_config}")
            elif loaded_config is not None:
                self.logger.warning(
                    f"Specific configuration for key '{self._component_config_key}' for component '{self.name}' "
                    f"is not a dictionary (found type: {type(loaded_config)}, value: {loaded_config}). Using empty specific config."
                )
            else: 
                self.logger.warning(
                    f"No specific configuration found for key '{self._component_config_key}' for component '{self.name}'. Using empty config. `get` returned None."
                )
        elif self._config_loader and not self._component_config_key:
            self.logger.debug(f"No component_config_key provided for '{self.name}'. Using empty specific config.")
        elif not self._config_loader:
            self.logger.warning(f"No config_loader provided for component '{self.name}'. Component may not be configurable.")
            
        self.logger.info(f"Component '{self.name}' created. State: {self.state}")

    def get_specific_config(self, key: str, default: Any = None) -> Any:
        if not isinstance(self.component_specific_config, dict):
            self.logger.warning(f"Component specific config for '{self.name}' is not a dictionary. Cannot get key '{key}'. Returning default.")
            return default
        return self.component_specific_config.get(key, default)

    def get_state(self) -> str:
        return self.state

    @abstractmethod
    def setup(self):
        self.logger.info(f"Setting up component '{self.name}'...")
        pass

    @abstractmethod
    def start(self):
        self.logger.info(f"Starting component '{self.name}'...")
        pass

    @abstractmethod
    def stop(self):
        self.logger.info(f"Stopping component '{self.name}'...")
        pass
