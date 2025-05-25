#!/usr/bin/env python3
"""
Refactored main.py using the new Bootstrap system.

This demonstrates the clean separation of concerns where main.py is just
a thin orchestration layer that delegates all setup to Bootstrap.
"""

import argparse
import logging
import sys
from typing import Optional, Any

# Core imports
from src.core.config import Config
from src.core.bootstrap import Bootstrap, RunMode
from src.core.exceptions import ConfigurationError, ADMFTraderError


logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ADMF-Trader Application")
    
    # Basic configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml", 
        help="Path to the configuration YAML file."
    )
    
    # Run mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=['production', 'backtest', 'optimization', 'test'],
        default='backtest',
        help="Run mode for the application"
    )
    
    # Optimization options (for backward compatibility)
    parser.add_argument(
        "--optimize", 
        action="store_true", 
        help="Run in optimization mode (same as --mode optimization)"
    )
    parser.add_argument(
        "--genetic-optimize", 
        action="store_true", 
        help="Use genetic algorithm optimization"
    )
    parser.add_argument(
        "--random-search", 
        action="store_true", 
        help="Use random search optimization"
    )
    
    # Data options
    parser.add_argument(
        "--bars", 
        type=int, 
        default=None, 
        help="Number of bars to process. Default is all."
    )
    
    # Logging options
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Override the log level defined in the config file."
    )
    parser.add_argument(
        "--debug-log", 
        type=str, 
        default=None, 
        help="Enable detailed DEBUG logging to the specified file."
    )
    
    args = parser.parse_args()
    
    # Handle legacy --optimize flag
    if args.optimize:
        args.mode = 'optimization'
    
    return args


def configure_logging(args, config: Config):
    """Configure logging based on arguments and config."""
    # Get log level from args or config
    log_level = args.log_level or config.get("logging.level", "INFO")
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    # Add debug file handler if requested
    if args.debug_log:
        debug_handler = logging.FileHandler(args.debug_log, mode='w')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(debug_handler)
        logger.info(f"Debug logging enabled to file: {args.debug_log}")


def update_config_for_args(config: Config, args):
    """Update configuration based on command line arguments."""
    # Update max bars if specified
    if args.bars is not None:
        config.set("components.data_handler_csv.max_bars", args.bars)
        logger.info(f"Set max_bars to {args.bars}")
    
    # Store optimization preferences in config
    if args.mode == 'optimization':
        if args.genetic_optimize:
            config.set("components.optimizer.use_genetic", True)
        if args.random_search:
            config.set("components.optimizer.use_random_search", True)


def main():
    """Main entry point using Bootstrap system."""
    # Parse arguments
    args = parse_arguments()
    
    # Minimal logging until config is loaded
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = Config(args.config)
        
        # Configure logging
        configure_logging(args, config)
        logger.info(f"ADMF-Trader starting in {args.mode} mode")
        
        # Update config based on arguments
        update_config_for_args(config, args)
        
        # Determine run mode
        run_mode = RunMode(args.mode)
        
        # Use Bootstrap context manager for automatic cleanup
        with Bootstrap() as bootstrap:
            # Initialize system
            logger.info("Initializing system context...")
            context = bootstrap.initialize(
                config=config,
                run_mode=run_mode
            )
            
            # Set up all managed components
            logger.info("Setting up managed components...")
            components = bootstrap.setup_managed_components(
                search_paths=["src/"]  # Search for component_meta.yaml files
            )
            logger.info(f"Created {len(components)} components")
            
            # Start all components
            logger.info("Starting components...")
            bootstrap.start_components()
            
            # Execute the entrypoint component based on run mode
            logger.info(f"Executing {run_mode.value} entrypoint...")
            try:
                result = bootstrap.execute_entrypoint()
                
                # Handle results based on run mode
                if run_mode == RunMode.OPTIMIZATION and result:
                    log_optimization_results(result)
                elif run_mode == RunMode.BACKTEST and result:
                    log_backtest_results(result)
                    
            except AttributeError as e:
                # Entrypoint doesn't have execute method
                logger.warning(f"Entrypoint component doesn't have execute() method: {e}")
                logger.info("Running in compatibility mode...")
                run_legacy_logic(bootstrap, run_mode)
                
        # Bootstrap automatically handles cleanup on exit
        logger.info("ADMF-Trader finished successfully")
        
    except ConfigurationError as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
    except ADMFTraderError as e:
        logger.critical(f"Application error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def log_optimization_results(results: Any):
    """Log optimization results."""
    if isinstance(results, dict):
        logger.info("=== OPTIMIZATION RESULTS ===")
        if "best_parameters" in results:
            logger.info(f"Best parameters: {results['best_parameters']}")
        if "best_metric_value" in results:
            logger.info(f"Best metric value: {results['best_metric_value']}")
        if "test_metric_value" in results:
            logger.info(f"Test set metric: {results['test_metric_value']}")
            
        # Log per-regime results if available
        if "best_weights_per_regime" in results:
            logger.info("\nPer-regime optimized weights:")
            for regime, weight_data in results["best_weights_per_regime"].items():
                weights = weight_data.get("weights", {})
                fitness = weight_data.get("fitness", "N/A")
                logger.info(f"  {regime}: {weights} (fitness: {fitness})")


def log_backtest_results(results: Any):
    """Log backtest results."""
    if isinstance(results, dict):
        logger.info("=== BACKTEST RESULTS ===")
        metrics = results.get("metrics", {})
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value}")


def run_legacy_logic(bootstrap: Bootstrap, run_mode: RunMode):
    """
    Run legacy application logic for components that don't have execute() method.
    This provides backward compatibility during migration.
    """
    logger.info("Running legacy application logic...")
    
    # Get key components
    context = bootstrap.get_context()
    
    if run_mode == RunMode.OPTIMIZATION:
        # Legacy optimization logic
        optimizer = context.container.get("optimizer")
        if optimizer and hasattr(optimizer, "run_grid_search"):
            logger.info("Running legacy grid search optimization...")
            results = optimizer.run_grid_search()
            if results:
                log_optimization_results(results)
    else:
        # Legacy backtest logic
        data_handler = context.container.get("data_handler")
        if data_handler:
            # The components are already started, just let them run
            logger.info("Data handler streaming events...")
            # Events will flow automatically through the event bus
            
        # Close positions at end
        portfolio = context.container.get("portfolio")
        if portfolio and hasattr(portfolio, "close_all_positions"):
            logger.info("Closing all positions...")
            portfolio.close_all_positions()


if __name__ == "__main__":
    main()