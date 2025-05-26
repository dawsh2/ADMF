#!/usr/bin/env python3
"""
Application Launcher - handles argument parsing and Bootstrap initialization.

This is the bridge between the minimal main.py and the Bootstrap system.
It parses command line args but doesn't determine application behavior -
that's left to the configuration and Bootstrap.
"""

import argparse
import logging
from typing import List, Dict, Any, Optional

from .config import SimpleConfigLoader
from .bootstrap import Bootstrap, RunMode
from .exceptions import ConfigurationError, ADMFTraderError


class ApplicationLauncher:
    """
    Launches the ADMF application by:
    1. Parsing command line arguments
    2. Loading configuration
    3. Setting up Bootstrap with AppRunner as the entrypoint
    4. Running the application
    """
    
    def __init__(self, argv: List[str]):
        """
        Initialize with command line arguments.
        
        Args:
            argv: Command line arguments (without script name)
        """
        self.argv = argv
        self.logger = logging.getLogger(__name__)
        
    def run(self) -> int:
        """
        Run the application.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Parse command line arguments
            args = self._parse_arguments()
            
            # Set up minimal logging for bootstrap process
            self._setup_bootstrap_logging(args)
            
            # Load configuration
            config_loader = SimpleConfigLoader(args.config)
            config = config_loader  # For now, use the loader directly
            self.logger.info(f"Configuration loaded from: {args.config}")
            
            # Determine run mode from CLI args first, then config
            run_mode = self._determine_run_mode(config, args)
            self.logger.info(f"Run mode: {run_mode.value}")
            
            # Prepare metadata with CLI args for AppRunner
            metadata = {
                'cli_args': vars(args)  # Convert Namespace to dict
            }
            
            # Run using Bootstrap
            with Bootstrap() as bootstrap:
                # Initialize system
                context = bootstrap.initialize(
                    config=config,
                    run_mode=run_mode,
                    metadata=metadata
                )
                
                # Ensure AppRunner is registered as a component
                self._register_app_runner(bootstrap)
                
                # Set AppRunner as the entrypoint for this run mode
                self._configure_entrypoint(bootstrap, run_mode)
                
                # Determine component overrides based on run mode
                component_overrides = self._get_component_overrides(run_mode, args)
                
                # Set up all components with overrides
                bootstrap.setup_managed_components(component_overrides=component_overrides)
                
                # Start all components
                bootstrap.start_components()
                
                # Execute the entrypoint (AppRunner)
                result = bootstrap.execute_entrypoint()
                
                if result:
                    self.logger.info(f"Execution completed successfully")
                    
            return 0  # Success
            
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")
            return 1
        except ADMFTraderError as e:
            self.logger.error(f"Application error: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1
            
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="ADMF-Trader Application",
            epilog="Application behavior is determined by configuration, not command line args."
        )
        
        # Configuration file (only required argument)
        parser.add_argument(
            "--config", "-c",
            type=str,
            default="config/config.yaml",
            help="Path to configuration file"
        )
        
        # Optional overrides that get passed to AppRunner
        parser.add_argument("--bars", type=int, help="Override max bars to process")
        parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        parser.add_argument("--debug-log", type=str, help="Debug log file path")
        parser.add_argument("--dataset", choices=['full', 'train', 'test'], 
                          help="Select dataset to use (requires train_test_split_ratio in config)")
        
        # Legacy optimization flags (for compatibility)
        parser.add_argument("--optimize", action="store_true", help="Run optimization")
        parser.add_argument("--optimize-ma", action="store_true", help="Optimize MA parameters")
        parser.add_argument("--optimize-rsi", action="store_true", help="Optimize RSI parameters")
        parser.add_argument("--optimize-seq", action="store_true", help="Sequential optimization")
        parser.add_argument("--optimize-joint", action="store_true", help="Joint optimization")
        parser.add_argument("--genetic-optimize", action="store_true", help="Use genetic algorithm")
        parser.add_argument("--random-search", action="store_true", help="Use random search")
        
        return parser.parse_args(self.argv)
        
    def _setup_bootstrap_logging(self, args: argparse.Namespace):
        """Set up logging for the bootstrap process including file logging."""
        from .logging_setup import setup_logging
        
        # Use the full setup_logging function which creates log files
        cmd_log_level = args.log_level.upper() if args.log_level else None
        
        # Load config to get logging settings
        config_loader = SimpleConfigLoader(args.config)
        
        # Call the proper setup_logging function which creates timestamped log files
        setup_logging(
            config_loader, 
            cmd_log_level=cmd_log_level, 
            optimization_mode=False,
            debug_file=args.debug_log
        )
            
    def _determine_run_mode(self, config: SimpleConfigLoader, args) -> RunMode:
        """
        Determine run mode from CLI args first, then configuration.
        
        Priority order:
        1. CLI --optimize flag (overrides config)
        2. Config file run_mode setting
        3. Default to backtest
        """
        # Check CLI args first - --optimize flag takes precedence
        if hasattr(args, 'optimize') and args.optimize:
            self.logger.info("Optimization mode enabled via --optimize flag")
            return RunMode.OPTIMIZATION
            
        # Check if config explicitly sets the run mode (top-level)
        run_mode = config.get("run_mode")
        if run_mode:
            try:
                return RunMode(run_mode)
            except ValueError:
                self.logger.warning(f"Invalid run_mode '{run_mode}' in config")
                
        # Check if config explicitly sets the application mode (legacy)
        app_mode = config.get("system.application_mode")
        if app_mode:
            try:
                return RunMode(app_mode)
            except ValueError:
                self.logger.warning(f"Invalid application_mode '{app_mode}' in config")
                
        # For backward compatibility, check for optimization section
        if config.get("optimization.enabled", False):
            return RunMode.OPTIMIZATION
            
        # Default to backtest
        return RunMode.BACKTEST
        
    def _get_component_overrides(self, run_mode: RunMode, args) -> Optional[Dict[str, str]]:
        """
        Get component overrides based on run mode and CLI args.
        
        This allows us to use different strategy implementations based on
        whether we're optimizing or doing verification.
        
        Args:
            run_mode: The current run mode
            args: CLI arguments
            
        Returns:
            Dictionary of component name -> class name overrides
        """
        # No overrides needed - we'll handle this in the config loading instead
        return None
        
    def _register_app_runner(self, bootstrap: Bootstrap):
        """Register AppRunner as a component."""
        bootstrap.component_definitions['app_runner'] = {
            'class': 'AppRunner',
            'module': 'core.app_runner',
            'dependencies': ['event_bus', 'container'],
            'config_key': 'components.app_runner',
            'required': True
        }
        
    def _configure_entrypoint(self, bootstrap: Bootstrap, run_mode: RunMode):
        """Configure AppRunner as the entrypoint for the current run mode."""
        # Since SimpleConfigLoader is read-only, we can't modify config at runtime
        # The entrypoint should be configured in the YAML file or handled differently
        # For now, we'll skip this and rely on Bootstrap to handle the entrypoint
        pass