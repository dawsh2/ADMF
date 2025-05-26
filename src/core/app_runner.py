#!/usr/bin/env python3
"""
AppRunner - the main application component that orchestrates execution.

This component inherits from ComponentBase and handles the actual
application logic that was previously in main.py. It's managed by
Bootstrap like any other component, which gives us proper lifecycle
management and dependency injection.
"""

import datetime
from typing import Optional, Dict, Any

from .component_base import ComponentBase
from .bootstrap import RunMode
from .exceptions import ComponentError, DependencyNotFoundError


class AppRunner(ComponentBase):
    """
    Main application runner component.
    
    This component is set as the entrypoint by ApplicationLauncher and
    handles the high-level application flow based on the run mode.
    """
    
    def _initialize(self) -> None:
        """
        Initialize AppRunner with CLI arguments and configuration.
        
        The CLI args are passed via context.metadata by ApplicationLauncher.
        """
        # Get CLI arguments from context metadata
        cli_args = self.context.metadata.get('cli_args', {})
        
        # Store relevant arguments
        self.max_bars = cli_args.get('bars')
        self.optimize_flags = {
            'optimize': cli_args.get('optimize', False),
            'optimize_ma': cli_args.get('optimize_ma', False),
            'optimize_rsi': cli_args.get('optimize_rsi', False),
            'optimize_seq': cli_args.get('optimize_seq', False),
            'optimize_joint': cli_args.get('optimize_joint', False),
            'genetic_optimize': cli_args.get('genetic_optimize', False),
            'random_search': cli_args.get('random_search', False)
        }
        
        self.logger.info(
            f"AppRunner initialized for {self.context.run_mode.value} mode"
        )
        
        if self.max_bars:
            self.logger.info(f"Max bars override: {self.max_bars}")
            
    def execute(self) -> Optional[Dict[str, Any]]:
        """
        Main execution method called by Bootstrap.execute_entrypoint().
        
        Delegates to specific methods based on run mode.
        """
        if not self.initialized or not self.running:
            raise ComponentError(
                f"AppRunner not ready: initialized={self.initialized}, "
                f"running={self.running}"
            )
            
        self.logger.info(f"Executing {self.context.run_mode.value} mode")
        
        # Delegate based on run mode
        if self.context.run_mode == RunMode.OPTIMIZATION:
            return self._run_optimization()
        elif self.context.run_mode == RunMode.BACKTEST:
            return self._run_backtest()
        elif self.context.run_mode == RunMode.PRODUCTION:
            return self._run_production()
        elif self.context.run_mode == RunMode.TEST:
            return self._run_test()
        else:
            raise ComponentError(f"Unknown run mode: {self.context.run_mode}")
            
    def _run_optimization(self) -> Dict[str, Any]:
        """Run optimization mode."""
        self.logger.info("Starting optimization run")
        
        # Update data handler with max bars if specified
        if self.max_bars:
            data_handler = self.container.get('data_handler')
            if data_handler and hasattr(data_handler, 'set_max_bars'):
                data_handler.set_max_bars(self.max_bars)
        
        # Check if we have a workflow orchestrator configured
        workflow_orchestrator = self.container.get('workflow_orchestrator')
        if workflow_orchestrator:
            # Use the new config-driven workflow system
            self.logger.info("Using workflow orchestrator for config-driven optimization")
            
            # Get required components
            data_handler = self.container.get('data_handler')
            portfolio_manager = self.container.get('portfolio_manager')
            strategy = self.container.get('strategy')
            risk_manager = self.container.get('risk_manager')
            execution_handler = self.container.get('execution_handler')
            
            # Get date ranges from config
            opt_config = self._context.config.get("optimization", {})
            train_dates = opt_config.get("train_date_range", ["2023-01-01", "2023-06-30"])
            test_dates = opt_config.get("test_date_range", ["2023-07-01", "2023-12-31"])
            
            # Run the workflow
            results = workflow_orchestrator.run_optimization_workflow(
                data_handler=data_handler,
                portfolio_manager=portfolio_manager,
                strategy=strategy,
                risk_manager=risk_manager,
                execution_handler=execution_handler,
                train_dates=tuple(train_dates),
                test_dates=tuple(test_dates)
            )
            
            return results
        
        # Fall back to legacy optimizer for backward compatibility
        optimizer = self.container.get('optimizer')
        if not optimizer:
            raise DependencyNotFoundError("Neither workflow_orchestrator nor optimizer component found")
            
        self.logger.info("Using legacy optimizer (configure workflow_orchestrator for new features)")
        
        # Run grid search
        self.logger.info("Starting grid search optimization")
        results = optimizer.run_grid_search()
        
        if not results:
            self.logger.error("Grid search returned no results")
            return {'status': 'failed', 'reason': 'No optimization results'}
            
        # Run additional optimization if requested
        if self.optimize_flags['genetic_optimize']:
            self.logger.info("Running genetic optimization")
            if hasattr(optimizer, 'run_per_regime_genetic_optimization'):
                results = optimizer.run_per_regime_genetic_optimization(results)
                
        elif self.optimize_flags['random_search']:
            self.logger.info("Running random search optimization")
            if hasattr(optimizer, 'run_per_regime_random_search_optimization'):
                results = optimizer.run_per_regime_random_search_optimization(results)
                
        # Run adaptive test if available
        if hasattr(optimizer, 'run_adaptive_test'):
            self.logger.info("Running adaptive test")
            optimizer.run_adaptive_test(results)
            
        return results
        
    def _run_backtest(self) -> Dict[str, Any]:
        """Run standard backtest mode."""
        self.logger.info("Starting backtest run")
        
        # Get required components
        data_handler = self.container.get('data_handler')
        portfolio = self.container.get('portfolio_manager')
        
        if not data_handler or not portfolio:
            raise DependencyNotFoundError("Required components not found")
            
        # Set dataset based on configuration
        if hasattr(data_handler, 'set_active_dataset'):
            # Use test set if available (for validation)
            if (hasattr(data_handler, 'test_df_exists_and_is_not_empty') and 
                data_handler.test_df_exists_and_is_not_empty):
                self.logger.info("Using test dataset")
                data_handler.set_active_dataset('test')
            else:
                self.logger.info("Using full dataset")
                data_handler.set_active_dataset('full')
                
        # Apply max bars if specified
        if self.max_bars and hasattr(data_handler, 'set_max_bars'):
            data_handler.set_max_bars(self.max_bars)
            
        # The event-driven backtest runs automatically once components are started
        # We just need to wait for completion and collect results
        
        self.logger.info("Backtest event flow active")
        
        # After data handler completes, close positions
        last_timestamp = self._get_last_timestamp(data_handler, portfolio)
        
        if portfolio and hasattr(portfolio, 'close_all_positions'):
            self.logger.info(f"Closing all positions at {last_timestamp}")
            portfolio.close_all_positions(last_timestamp)
            
        # Get performance summary
        results = {}
        if hasattr(portfolio, 'get_performance_summary'):
            results = portfolio.get_performance_summary()
            
        # Log final performance
        if hasattr(portfolio, 'log_final_performance_summary'):
            portfolio.log_final_performance_summary()
            
        return results
        
    def _run_production(self) -> Dict[str, Any]:
        """Run production/live trading mode."""
        self.logger.info("Starting production run")
        
        # Production mode would connect to live data feeds and brokers
        # For now, this is a placeholder
        return {
            'status': 'not_implemented',
            'message': 'Production mode requires live data/broker connections'
        }
        
    def _run_test(self) -> Dict[str, Any]:
        """Run test mode."""
        self.logger.info("Starting test run")
        
        # Test mode could run integration tests, unit tests, etc.
        # For now, just run a simple backtest with limited data
        if not self.max_bars:
            self.max_bars = 100  # Default to small dataset for tests
            
        return self._run_backtest()
        
    def _get_last_timestamp(self, data_handler, portfolio) -> datetime.datetime:
        """Get the last timestamp for closing positions."""
        last_timestamp = None
        
        # Try data handler first
        if data_handler and hasattr(data_handler, 'get_last_timestamp'):
            last_timestamp = data_handler.get_last_timestamp()
            
        # Fall back to portfolio
        if not last_timestamp and portfolio and hasattr(portfolio, 'get_last_processed_timestamp'):
            last_timestamp = portfolio.get_last_processed_timestamp()
            
        # Final fallback
        if not last_timestamp:
            last_timestamp = datetime.datetime.now(datetime.timezone.utc)
            self.logger.warning(f"Using current time as fallback: {last_timestamp}")
            
        return last_timestamp
        
    def initialize_event_subscriptions(self) -> None:
        """
        AppRunner doesn't need to subscribe to events directly.
        It orchestrates other components that handle events.
        """
        pass
        
    def _validate_configuration(self) -> None:
        """Validate AppRunner configuration if needed."""
        # AppRunner doesn't have specific configuration requirements
        pass