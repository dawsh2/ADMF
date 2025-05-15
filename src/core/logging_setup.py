# src/core/logging_setup.py
import logging
import sys
# We'll need SimpleConfigLoader to get the config, so an import will be needed
# from .config import SimpleConfigLoader # Assuming SimpleConfigLoader is accessible

LOG_LEVEL_STRINGS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}

def setup_logging(config_loader): # We'll pass the config_loader instance
    """
    Configures basic logging for the application.

    Reads the logging level from the configuration and sets up
    a basic console logger.
    """
    log_level_str = config_loader.get('logging.level', 'INFO').upper()
    log_level = LOG_LEVEL_STRINGS.get(log_level_str, logging.INFO)

    # Basic configuration logs to stderr by default.
    # We can specify stream=sys.stdout if preferred for INFO/DEBUG.
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout # Log to standard output
    )

    # You can get a root logger instance and test it
    # logging.info("Logging system initialized.")
    # logging.debug("This is a debug message (will not show if level is INFO).")



# Possibly an enhancement?

# # src/core/logging_setup.py
# import logging
# import logging.config
# import os
# from typing import Any, Dict
# import copy # Import copy

# # Default logging configuration (used if config file doesn't specify or is problematic)
# DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
#     'version': 1,
#     'disable_existing_loggers': False, 
#     'formatters': {
#         'standard': {
#             'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             'datefmt': '%Y-%m-%d %H:%M:%S'
#         },
#         'component_formatter': { 
#             'format': '%(asctime)s - component.%(component_name)s - %(levelname)s - %(message)s',
#             'datefmt': '%Y-%m-%d %H:%M:%S'
#         }
#     },
#     'handlers': {
#         'console': {
#             'class': 'logging.StreamHandler',
#             'formatter': 'standard',
#             'level': 'INFO', 
#             'stream': 'ext://sys.stdout'
#         },
#     },
#     'root': { 
#         'handlers': ['console'], 
#         'level': 'INFO', 
#     },
# }

# # Get a logger for this module itself
# module_logger = logging.getLogger(__name__)

# def setup_logging(config_loader: Any, default_level: int = logging.INFO):
#     """
#     Sets up logging configuration for the application.
#     Uses configuration from the config_loader if available, otherwise uses DEFAULT_LOGGING_CONFIG.
#     """
#     logging_config_to_use: Optional[Dict[str, Any]] = None
    
#     if config_loader:
#         loaded_logging_section = config_loader.get('logging')

#         if isinstance(loaded_logging_section, dict) and loaded_logging_section:
#             # --- MAKE AN EXPLICIT DEEP COPY HERE ---
#             logging_config_to_use = copy.deepcopy(loaded_logging_section)
#             module_logger.info(f"Using logging configuration from file (after deepcopy): {logging_config_to_use}")
#             # -----------------------------------------

#             if 'version' not in logging_config_to_use:
#                 logging_config_to_use['version'] = 1
#             if 'disable_existing_loggers' not in logging_config_to_use:
#                 logging_config_to_use['disable_existing_loggers'] = False

#             # Simplified default handler/formatter setup if basics are missing
#             if 'formatters' not in logging_config_to_use or not logging_config_to_use['formatters']:
#                 logging_config_to_use.setdefault('formatters', {}).update(DEFAULT_LOGGING_CONFIG['formatters'])
            
#             if 'handlers' not in logging_config_to_use or not logging_config_to_use['handlers']:
#                  # Ensure console handler exists if no handlers are defined
#                 logging_config_to_use.setdefault('handlers', {}).update({'console': DEFAULT_LOGGING_CONFIG['handlers']['console']})


#             if 'root' not in logging_config_to_use and 'loggers' not in logging_config_to_use:
#                 logging_config_to_use['root'] = DEFAULT_LOGGING_CONFIG['root']
#             elif 'root' in logging_config_to_use:
#                 logging_config_to_use.setdefault('root', {}).setdefault('level', 'INFO')
#                 if not logging_config_to_use['root'].get('handlers'):
#                     logging_config_to_use['root']['handlers'] = ['console'] # Ensure console is a default for root if no handlers specified
            
#         else:
#             module_logger.info("Using DEFAULT_LOGGING_CONFIG because 'logging' section not found or invalid in config file.")
#             logging_config_to_use = copy.deepcopy(DEFAULT_LOGGING_CONFIG) # Also copy default
#     else:
#         module_logger.info("Using DEFAULT_LOGGING_CONFIG because no config_loader provided.")
#         logging_config_to_use = copy.deepcopy(DEFAULT_LOGGING_CONFIG) # Also copy default

#     try:
#         if logging_config_to_use:
#             logging.config.dictConfig(logging_config_to_use)
#             # Use a logger obtained *after* dictConfig might have reconfigured things
#             logging.getLogger("src.core.logging_setup").info("Logging system configured successfully using dictConfig.")
#         else: # Should not happen if logic above is correct
#             raise ValueError("logging_config_to_use is None, cannot configure logging.")
            
#     except Exception as e:
#         logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         logging.getLogger("src.core.logging_setup").error(f"Failed to apply logging configuration from dictConfig: {e}. Falling back to basicConfig.", exc_info=True)

