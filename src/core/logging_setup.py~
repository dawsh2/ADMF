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
