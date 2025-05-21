# src/core/logging_setup.py
import logging
import sys
import os

LOG_LEVEL_STRINGS = {
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
    
    # Debug formatter - detailed with timestamp and call location
    'debug': logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S'),
    
    # Minimal formatter - just the message for optimization progress
    'minimal': logging.Formatter('%(message)s')
}

def create_optimization_logger(name="optimization"):
    """
    Creates a specialized logger for optimization runs that:
    - Uses a minimal format without timestamps or level indicators
    - Always shows at INFO level regardless of root logger
    - Doesn't propagate to parent loggers (avoiding duplication)
    
    Args:
        name: Name for the logger, defaults to "optimization"
        
    Returns:
        A configured logger instance for optimization output
    """
    opt_logger = logging.getLogger(f"admf.{name}")
    opt_logger.setLevel(logging.INFO)  # Always show optimization progress
    opt_logger.propagate = False  # Don't propagate to root logger
    
    # Clear existing handlers
    for handler in opt_logger.handlers[:]:  
        opt_logger.removeHandler(handler)
        
    # Create a stdout handler with minimal formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTERS['minimal'])
    opt_logger.addHandler(handler)
    
    return opt_logger

def create_debug_file_logger(debug_file='adaptive_test_debug.log'):
    """
    Creates a debug file logger that:
    - Logs at DEBUG level regardless of root logger setting
    - Overwrites the file on each run
    - Uses a more detailed formatter
    
    Args:
        debug_file: Path to the debug log file, default is 'adaptive_test_debug.log'
    """
    # Make sure directory exists
    debug_dir = os.path.dirname(debug_file)
    if debug_dir and not os.path.exists(debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
    
    # Create a file handler that overwrites on each run
    debug_handler = logging.FileHandler(debug_file, mode='w', encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)  # Always log at DEBUG level
    debug_handler.setFormatter(FORMATTERS['debug'])
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(debug_handler)
    
    # Create a dedicated logger for enhanced optimizer
    adaptive_logger = logging.getLogger('src.strategy.optimization.enhanced_optimizer')
    adaptive_logger.setLevel(logging.DEBUG)
    
    # Also create loggers for related components
    for module in ['src.strategy.regime_detector', 'src.strategy.regime_adaptive_strategy',
                  'src.core.event_bus', 'src.core.container']:
        component_logger = logging.getLogger(module)
        component_logger.setLevel(logging.DEBUG)
    
    return debug_handler

def setup_logging(config_loader, cmd_log_level=None, optimization_mode=False, debug_file=None):
    """
    Configures basic logging for the application.

    Reads the logging level from the configuration and sets up
    a basic console logger. Command-line log level override takes precedence.
    
    Args:
        config_loader: The config loader instance
        cmd_log_level: Optional command-line log level override
        optimization_mode: If True, configures for optimization (concise format)
        debug_file: Optional path to debug log file, if provided enables detailed DEBUG logging to file
    """
    # Create logs directory if it doesn't exist
    import datetime
    from pathlib import Path
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not debug_file:
        debug_file = logs_dir / f"admf_{timestamp}.log"
    
    # Command-line log level override takes precedence if provided
    if cmd_log_level is not None:
        log_level_str = cmd_log_level.upper()
        # Only print override message if we're not in optimization mode
        if not optimization_mode:
            print(f"Overriding log level from config with command-line value: {log_level_str}")
    else:
        # Default to WARNING instead of INFO for cleaner startup
        log_level_str = config_loader.get('logging.level', 'WARNING').upper()
        
    log_level = LOG_LEVEL_STRINGS.get(log_level_str, logging.WARNING)  # Default to WARNING

    # Update root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Choose formatter based on mode
    if optimization_mode:
        formatter = FORMATTERS['standard']
    else:
        formatter = FORMATTERS['verbose']
    
    # Clear existing handlers if any
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
    else:
        # Basic configuration logs to stdout and file
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
        
        # Always add a file handler for all logs
        file_handler = logging.FileHandler(debug_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log at DEBUG level to file
        file_handler.setFormatter(FORMATTERS['debug'])
        root_logger.addHandler(file_handler)
        print(f"Logging to file: {debug_file}")
    
    # Setup additional debug file logging if requested specifically
    if debug_file and debug_file != str(debug_file):
        debug_handler = create_debug_file_logger(debug_file)
        print(f"Additional detailed DEBUG logging enabled to file: {debug_file}")
    
    # Set specific loggers to DEBUG level to capture all events
    for module in ['src.strategy.regime_detector', 'src.strategy.regime_adaptive_strategy',
                  'src.core.event_bus', 'src.strategy.ma_strategy']:
        component_logger = logging.getLogger(module)
        component_logger.setLevel(logging.DEBUG)
    
    # Only log the level change if not in optimization mode
    if not optimization_mode:
        setup_logger = logging.getLogger('src.core.logging_setup')
        setup_logger.debug(f"Logging level set to: {log_level_str}")  # Debug not Info