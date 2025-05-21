"""
Debug logging facility for the ADMF framework.

This module provides functions to set up a dedicated file logger with flexible
configuration options, allowing components to perform detailed debug logging
during development and testing of complex components like regime-adaptive strategies.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Union

# Define log level constants for convenience
LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}

# Define formatter types for different use cases
FORMATTERS = {
    # Standard formatter - concise but with enough context
    'standard': logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%H:%M:%S'),
    
    # Verbose formatter - includes module name for debugging
    'verbose': logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S'),
    
    # Detailed formatter - includes thread and function information
    'detailed': logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S'),
    
    # Minimal formatter - just the message
    'minimal': logging.Formatter('%(message)s')
}

def setup_debug_logger(
    logger_name: str, 
    log_level: Union[str, int] = "DEBUG", 
    file_path: Optional[str] = None,
    formatter: str = "detailed",
    propagate: bool = False,
    rotate: bool = False,
    max_size_mb: int = 10,
    backup_count: int = 3,
    overwrite: bool = True
) -> logging.Logger:
    """
    Sets up a specialized debug logger that writes to a file.
    
    Args:
        logger_name: Name for the logger (usually module or component name)
        log_level: Logging level to use (DEBUG, INFO, etc. or logging level constant)
        file_path: Path to the log file. If None, creates a file in the logs directory
        formatter: Name of the formatter to use (standard, verbose, detailed, minimal)
        propagate: Whether to propagate messages to parent loggers
        rotate: Whether to use a rotating file handler
        max_size_mb: Maximum size in MB for log file before rotation
        backup_count: Number of backup files to keep when rotating
        overwrite: Whether to overwrite existing log file (if not rotating)
    
    Returns:
        A configured logger instance
    """
    # Ensure we have a valid log level
    if isinstance(log_level, str):
        level = LOG_LEVELS.get(log_level.upper(), logging.DEBUG)
    else:
        level = log_level
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Determine the log file path
    if file_path is None:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        file_path = os.path.join(logs_dir, f"{logger_name.replace('.', '_')}.log")
    
    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Get formatter
    fmt = FORMATTERS.get(formatter, FORMATTERS['detailed'])
    
    # Set up file handler
    if rotate:
        # Use rotating file handler
        handler = RotatingFileHandler(
            file_path,
            maxBytes=max_size_mb * 1024 * 1024,  # Convert to bytes
            backupCount=backup_count,
            delay=False  # Create file immediately
        )
    else:
        # Use standard file handler
        if overwrite and os.path.exists(file_path):
            mode = 'w'  # Overwrite
        else:
            mode = 'a'  # Append
        handler = logging.FileHandler(file_path, mode=mode)
    
    # Configure handler
    handler.setLevel(level)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    
    # Add a console handler that only shows warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(FORMATTERS['standard'])
    logger.addHandler(console_handler)
    
    # Log initialization
    logger.info(f"Debug logger '{logger_name}' initialized, writing to {file_path}")
    logger.info(f"Log level: {logging.getLevelName(level)}, Format: {formatter}, Propagate: {propagate}")
    
    return logger

def add_file_handler(
    logger: logging.Logger,
    file_path: str,
    log_level: Union[str, int] = "DEBUG",
    formatter: str = "detailed",
    rotate: bool = False,
    max_size_mb: int = 10,
    backup_count: int = 3,
    overwrite: bool = True
) -> logging.Logger:
    """
    Adds an additional file handler to an existing logger.
    
    Args:
        logger: The logger to add a handler to
        file_path: Path to the log file
        log_level: Logging level for this handler
        formatter: Name of the formatter to use
        rotate: Whether to use a rotating file handler 
        max_size_mb: Maximum size in MB for log file before rotation
        backup_count: Number of backup files to keep when rotating
        overwrite: Whether to overwrite existing log file (if not rotating)
    
    Returns:
        The logger with the additional handler
    """
    # Ensure we have a valid log level
    if isinstance(log_level, str):
        level = LOG_LEVELS.get(log_level.upper(), logging.DEBUG)
    else:
        level = log_level
        
    # Get formatter
    fmt = FORMATTERS.get(formatter, FORMATTERS['detailed'])
    
    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Set up file handler
    if rotate:
        # Use rotating file handler
        handler = RotatingFileHandler(
            file_path,
            maxBytes=max_size_mb * 1024 * 1024,  # Convert to bytes
            backupCount=backup_count
        )
    else:
        # Use standard file handler
        if overwrite and os.path.exists(file_path):
            mode = 'w'  # Overwrite
        else:
            mode = 'a'  # Append
        handler = logging.FileHandler(file_path, mode=mode)
    
    # Configure handler
    handler.setLevel(level)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    
    logger.info(f"Added file handler to '{logger.name}', writing to {file_path}")
    
    return logger

def log_component_lifecycle(
    logger: logging.Logger,
    component_name: str,
    action: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Logs a component lifecycle event with a standardized format.
    
    Args:
        logger: The logger to use
        component_name: Name of the component
        action: What is happening (start, stop, reset, etc.)
        details: Optional additional details to include
    """
    event = f"LIFECYCLE: Component '{component_name}' - {action}"
    
    if details:
        # Format details
        details_str = ", ".join(f"{k}={v}" for k, v in details.items())
        event += f" - {details_str}"
    
    logger.info(event)

def log_runtime_value(
    logger: logging.Logger,
    name: str,
    value: Any,
    component: str = "",
    category: str = "VALUE"
) -> None:
    """
    Logs a runtime value with standardized format for easy filtering.
    
    Args:
        logger: The logger to use
        name: Name of the value being logged
        value: The value to log
        component: Optional component name for context
        category: Category for grouping values (default: VALUE)
    """
    msg = f"{category}: "
    if component:
        msg += f"[{component}] "
    
    # Add special formatting for different value types
    if isinstance(value, dict):
        msg += f"{name} = {{ {', '.join(f'{k}: {v}' for k, v in value.items())} }}"
    elif isinstance(value, (list, tuple)):
        msg += f"{name} = [{', '.join(str(v) for v in value)}]"
    else:
        msg += f"{name} = {value}"
    
    logger.debug(msg)