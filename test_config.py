# test_config.py
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.exceptions import ConfigurationError # Import the exception
import logging

logger = logging.getLogger(__name__)

def main():
    print("--- Test Script Start ---")
    config_loader = None # Initialize to None

    # 1. Load Configuration (with error handling)
    print("\nAttempting to load configuration...")
    try:
        # Assuming you modified the default path in SimpleConfigLoader itself
        # or you provide it here:
        config_loader = SimpleConfigLoader(config_file_path="config/config.yaml")
        system_name_from_config = config_loader.get("system.name")
        # The print below would require logging to be setup first to see its logger output
        # logger.info(f"Successfully loaded config. System name: {system_name_from_config}")
        print(f"Successfully loaded config. System name: {system_name_from_config}")

    except ConfigurationError as e:
        # Logging is not set up yet if config loading fails.
        # So, we might just print the error here or use a pre-setup basic logger.
        # For now, print is fine for this test script's early failure.
        print(f"CRITICAL: Failed to load configuration: {e}. Exiting test script.")
        return # Exit if config fails, as logging setup depends on it.
    except Exception as e: # Catch any other unexpected error during config load
        print(f"CRITICAL: An unexpected error occurred during config load: {e}. Exiting.")
        return


    # 2. Setup Logging
    print("\nSetting up logging...")
    setup_logging(config_loader) # Pass the config_loader instance
    logger.info("Logging has been set up by test_config.py.")
    # ... (rest of the main function from previous version)

    # ... (rest of the script) ...
    logger.debug(f"This is a DEBUG message from {__name__}. It will only appear if log level is DEBUG.")
    logger.info(f"This is an INFO message from {__name__}.")
    logger.warning(f"This is a WARNING message from {__name__}.")
    logger.error(f"This is an ERROR message from {__name__}.")
    logger.critical(f"This is a CRITICAL message from {__name__}.")

    print("\n--- Original Config Value Tests (from previous step) ---")
    print(f"System Name: {config_loader.get('system.name', 'Default System Name')}")
    # ... (rest of print statements)

    print("\n--- Test Script End ---")


if __name__ == "__main__":
    main()
