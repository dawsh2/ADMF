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
            
    def _get_component(self, name: str):
        """Helper to get component from container."""
        if hasattr(self.container, 'get'):
            return self.container.get(name)
        elif hasattr(self.container, 'resolve'):
            return self.container.resolve(name)
        else:
            raise AttributeError(f"Container has neither 'get' nor 'resolve' method")
    
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
        workflow_orchestrator = self._get_component('workflow_orchestrator')
        if workflow_orchestrator:
            # Use the new config-driven workflow system
            self.logger.info("Using workflow orchestrator for config-driven optimization")
            
            # Get required components
            data_handler = self._get_component('data_handler')
            portfolio_manager = self._get_component('portfolio_manager')
            strategy = self._get_component('strategy')
            risk_manager = self._get_component('risk_manager')
            execution_handler = self._get_component('execution_handler')
            
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
        optimizer = self._get_component('optimizer')
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
        strategy = self.container.get('strategy')
        
        if not data_handler or not portfolio:
            raise DependencyNotFoundError("Required components not found")
            
        # Check if we should load optimized parameters for test set
        cli_args = self._context.metadata.get('cli_args', {})
        dataset_arg = cli_args.get('dataset', '')
        
        # Set dataset based on configuration
        if hasattr(data_handler, 'set_active_dataset'):
            # Use test set if available (for validation)
            if dataset_arg == 'test' or (hasattr(data_handler, 'test_df_exists_and_is_not_empty') and 
                data_handler.test_df_exists_and_is_not_empty):
                self.logger.info("Using test dataset")
                data_handler.set_active_dataset('test')
                
                # If using test dataset and we have regime parameters, load them
                if strategy and dataset_arg == 'test':
                    self.logger.warning("\n" + "="*80)
                    self.logger.warning("🧪 RUNNING TEST DATASET WITH OPTIMIZED PARAMETERS 🧪")
                    self.logger.warning("="*80)
                    self._load_optimized_parameters_for_test(strategy)
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
        elif hasattr(portfolio, 'get_performance_metrics'):
            results = portfolio.get_performance_metrics()
        elif hasattr(portfolio, 'get_performance'):
            results = portfolio.get_performance()
            
        # Log final performance - ALWAYS show for regular runs
        self.logger.warning("\n" + "="*60)
        self.logger.warning("BACKTEST COMPLETE - FINAL PERFORMANCE")
        self.logger.warning("="*60)
        if results:
            final_value = results.get('final_value', results.get('final_portfolio_value', 'N/A'))
            initial_value = results.get('initial_value', 100000)
            total_return = results.get('total_return_pct', results.get('total_return', 0) * 100)
            num_trades = results.get('num_trades', results.get('total_trades', 0))
            sharpe = results.get('portfolio_sharpe_ratio', 'N/A')
            win_rate = results.get('win_rate', 'N/A')
            
            self.logger.warning(f"Initial Portfolio Value: ${initial_value:,.2f}")
            self.logger.warning(f"Final Portfolio Value: ${final_value:,.2f}")
            self.logger.warning(f"Total Return: {total_return:.2f}%")
            self.logger.warning(f"Number of Trades: {num_trades}")
            self.logger.warning(f"Sharpe Ratio: {sharpe}")
            self.logger.warning(f"Win Rate: {win_rate}")
            
            # Display regime performance if available
            regime_perf = results.get('performance_metrics', {}).get('regime_performance', {})
            if not regime_perf and 'regime_performance' in results:
                regime_perf = results['regime_performance']
                
            if regime_perf and any(k != '_boundary_trades_summary' for k in regime_perf.keys()):
                self.logger.warning("\nREGIME PERFORMANCE:")
                for regime, perf in regime_perf.items():
                    if regime != '_boundary_trades_summary' and isinstance(perf, dict):
                        self.logger.warning(f"  {regime.upper()}: PnL=${perf.get('pnl', 0):,.2f}, "
                                          f"Trades={perf.get('count', 0)}, "
                                          f"Sharpe={perf.get('sharpe_ratio', 0):.2f}")
        else:
            self.logger.warning("No performance results available")
        self.logger.warning("="*60 + "\n")
        
        # If running test dataset, also print a summary at the very end
        if dataset_arg == 'test':
            import sys
            print("\n" + "="*80, file=sys.stderr)
            print("📊 TEST DATASET RUN COMPLETE - SUMMARY 📊", file=sys.stderr)
            print("="*80, file=sys.stderr)
            if results:
                print(f"Final Portfolio Value: ${final_value:,.2f}", file=sys.stderr)
                print(f"Total Return: {total_return:.2f}%", file=sys.stderr)
                print(f"Sharpe Ratio: {sharpe}", file=sys.stderr)
                print(f"Number of Trades: {num_trades}", file=sys.stderr)
                print(f"Win Rate: {win_rate}", file=sys.stderr)
            else:
                print("No performance results available", file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            
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
        
    def _load_optimized_parameters_for_test(self, strategy) -> None:
        """
        Load optimized parameters when running with --dataset test.
        
        This enables running test verification using the same config without --optimize flag,
        getting the same results as the test phase during optimization.
        """
        import json
        import os
        from pathlib import Path
        
        self.logger.info("Loading optimized parameters for test dataset run")
        
        # Get optimization config
        opt_config = self._context.config.get("optimization", {})
        results_dir = opt_config.get("results_directory", "optimization_results")
        
        # Look for the most recent regime optimized parameters file
        regime_params_file = Path(results_dir) / "regime_optimized_parameters.json"
        
        if not regime_params_file.exists():
            self.logger.warning(f"No optimized parameters found at {regime_params_file}")
            return
            
        try:
            with open(regime_params_file, 'r') as f:
                data = json.load(f)
                
            self.logger.info(f"Loaded regime parameters from {regime_params_file}")
            
            # Extract regime parameters from the nested structure
            if 'regimes' in data:
                regime_params = data['regimes']
            else:
                regime_params = data
            
            # Apply the parameters to the strategy
            if hasattr(strategy, '_regime_specific_params') and hasattr(strategy, '_overall_best_params'):
                # For RegimeAdaptiveEnsembleComposed strategy
                # Load regime-specific parameters
                for regime, params in regime_params.items():
                    if isinstance(params, dict) and regime != 'metadata':
                        strategy._regime_specific_params[regime] = params
                        
                # Also set overall best if available
                if 'overall_best' in regime_params:
                    strategy._overall_best_params = regime_params['overall_best']
                elif 'default' in regime_params:
                    # Use default as fallback
                    strategy._overall_best_params = regime_params['default']
                    
                self.logger.info("Loaded regime-specific parameters into strategy")
                
                # Log what was loaded for verification
                self.logger.info("Regime-specific parameters loaded:")
                for regime, params in regime_params.items():
                    if isinstance(params, dict) and regime != 'metadata':
                        # Show a summary of the parameters
                        param_count = len(params)
                        weight_params = [k for k in params.keys() if k.endswith('.weight')]
                        if weight_params:
                            weights_str = ", ".join([f"{k.split('.')[0]}={params[k]:.2f}" for k in weight_params])
                            self.logger.info(f"  {regime}: {param_count} parameters, weights: {weights_str}")
                        else:
                            self.logger.info(f"  {regime}: {param_count} parameters")
                            
                # Enable regime switching for test mode
                if hasattr(strategy, '_enable_regime_switching'):
                    strategy._enable_regime_switching = True
                    self.logger.info("Enabled regime switching for test phase")
            else:
                self.logger.warning("Strategy does not support regime-specific parameter loading")
                
        except Exception as e:
            self.logger.error(f"Failed to load optimized parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())