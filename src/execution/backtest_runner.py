#!/usr/bin/env python3
"""
BacktestRunner - dedicated component for running backtests.

This component handles the backtest execution flow, properly separating
this concern from other execution modes.
"""

import datetime
from typing import Optional, Dict, Any

from ..core.component_base import ComponentBase
from ..core.exceptions import ComponentError, DependencyNotFoundError


class BacktestRunner(ComponentBase):
    """
    Dedicated component for running backtests.
    
    This component:
    1. Orchestrates the backtest flow
    2. Manages data streaming
    3. Handles position closure at end
    4. Returns performance metrics
    """
    
    def _initialize(self) -> None:
        """Initialize the backtest runner."""
        # Get CLI overrides from context
        cli_args = self.context.metadata.get('cli_args', {})
        self.max_bars = cli_args.get('bars')
        
        # Get backtest-specific configuration
        self.use_test_dataset = self.component_config.get('use_test_dataset', False)
        self.close_positions_at_end = self.component_config.get('close_positions_at_end', True)
        
        self.logger.info(
            f"BacktestRunner initialized. Max bars: {self.max_bars}, "
            f"Use test dataset: {self.use_test_dataset}"
        )
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute the backtest.
        
        This is the main entry point called by Bootstrap.
        
        Returns:
            Dictionary containing backtest results and metrics
        """
        self.logger.info("Starting backtest execution")
        
        # Validate state
        if not self.initialized or not self.running:
            raise ComponentError("BacktestRunner not properly initialized")
            
        # Get required components
        data_handler = self._get_required_component('data_handler')
        portfolio = self._get_required_component('portfolio_manager')
        strategy = self._get_required_component('strategy')
        
        # Configure data handler
        self._configure_data_handler(data_handler)
        
        # The actual backtest runs via event flow
        # Components are already started by Bootstrap
        self.logger.info("Backtest event flow is active")
        
        # Note: In a real implementation, we might need to:
        # 1. Subscribe to a completion event from data_handler
        # 2. Or have data_handler.execute() block until complete
        # For now, we assume the event flow completes
        
        # Close positions if configured
        if self.close_positions_at_end:
            self._close_all_positions(data_handler, portfolio)
            
        # Collect and return results
        results = self._collect_results(portfolio, strategy)
        
        self.logger.info(f"Backtest completed. Total return: {results.get('total_return', 'N/A')}")
        
        return results
        
    def _get_required_component(self, name: str) -> Any:
        """Get a required component from the container."""
        component = self.container.get(name)
        if not component:
            raise DependencyNotFoundError(f"Required component '{name}' not found")
        return component
        
    def _configure_data_handler(self, data_handler) -> None:
        """Configure the data handler for backtest."""
        # Set active dataset
        if hasattr(data_handler, 'set_active_dataset'):
            if self.use_test_dataset and hasattr(data_handler, 'test_df_exists_and_is_not_empty'):
                if data_handler.test_df_exists_and_is_not_empty:
                    self.logger.info("Using test dataset for backtest")
                    data_handler.set_active_dataset('test')
                else:
                    self.logger.warning("Test dataset requested but not available")
                    data_handler.set_active_dataset('full')
            else:
                self.logger.info("Using full dataset for backtest")
                data_handler.set_active_dataset('full')
                
        # Apply max bars override
        if self.max_bars and hasattr(data_handler, 'set_max_bars'):
            self.logger.info(f"Limiting backtest to {self.max_bars} bars")
            data_handler.set_max_bars(self.max_bars)
            
    def _close_all_positions(self, data_handler, portfolio) -> None:
        """Close all open positions at end of backtest."""
        # Get last timestamp
        last_timestamp = None
        
        if hasattr(data_handler, 'get_last_timestamp'):
            last_timestamp = data_handler.get_last_timestamp()
            
        if not last_timestamp and hasattr(portfolio, 'get_last_processed_timestamp'):
            last_timestamp = portfolio.get_last_processed_timestamp()
            
        if not last_timestamp:
            last_timestamp = datetime.datetime.now(datetime.timezone.utc)
            self.logger.warning(f"Using current time for position closure: {last_timestamp}")
            
        # Close positions
        if hasattr(portfolio, 'close_all_positions'):
            self.logger.info(f"Closing all positions at {last_timestamp}")
            portfolio.close_all_positions(last_timestamp)
        else:
            self.logger.warning("Portfolio doesn't support close_all_positions")
            
    def _collect_results(self, portfolio, strategy) -> Dict[str, Any]:
        """Collect results from the backtest."""
        results = {
            'mode': 'backtest',
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Get portfolio metrics
        if hasattr(portfolio, 'get_performance_summary'):
            results.update(portfolio.get_performance_summary())
        else:
            # Fallback to basic metrics
            if hasattr(portfolio, 'get_total_return'):
                results['total_return'] = portfolio.get_total_return()
            if hasattr(portfolio, 'get_sharpe_ratio'):
                results['sharpe_ratio'] = portfolio.get_sharpe_ratio()
            if hasattr(portfolio, 'get_max_drawdown'):
                results['max_drawdown'] = portfolio.get_max_drawdown()
                
        # Get strategy info
        if hasattr(strategy, 'get_name'):
            results['strategy'] = strategy.get_name()
        else:
            results['strategy'] = strategy.__class__.__name__
            
        # Log final performance
        if hasattr(portfolio, 'log_final_performance_summary'):
            portfolio.log_final_performance_summary()
            
        return results
        
    def initialize_event_subscriptions(self) -> None:
        """
        BacktestRunner doesn't need direct event subscriptions.
        It orchestrates other components that handle events.
        """
        pass