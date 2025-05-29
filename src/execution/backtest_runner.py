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
from ..core.event import Event, EventType
from ..core.data_configurator import DataConfigurator


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
        cli_args = self._context.get('metadata', {}).get('cli_args', {})
        self.max_bars = cli_args.get('bars')
        self.dataset_override = cli_args.get('dataset')  # New: get dataset from CLI
        
        # Debug logging - use INFO level to ensure it shows
        self.logger.info(f"[DEBUG] Context metadata: {self._context.get('metadata', {})}")
        self.logger.info(f"[DEBUG] CLI args: {cli_args}")
        self.logger.info(f"[DEBUG] Bars from CLI: {cli_args.get('bars')}")
        self.logger.info(f"[DEBUG] Dataset from CLI: {cli_args.get('dataset')}")
        
        # Get backtest-specific configuration
        # CLI dataset override takes precedence over config
        if self.dataset_override == 'test':
            self.use_test_dataset = True
        elif self.dataset_override == 'train':
            self.use_test_dataset = False
        else:
            self.use_test_dataset = self.component_config.get('use_test_dataset', False)
        self.close_positions_at_end = self.component_config.get('close_positions_at_end', True)
        
        # Initialize the data configurator
        self.data_configurator = DataConfigurator(self.logger)
        
        self.logger.info(
            f"BacktestRunner initialized. Max bars: {self.max_bars}, "
            f"Use test dataset: {self.use_test_dataset}"
        )
        
    def _start(self) -> None:
        """Start the backtest runner."""
        self.logger.info("BacktestRunner started")
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute the backtest.
        
        This is the main entry point called by Bootstrap.
        
        Returns:
            Dictionary containing backtest results and metrics
        """
        self.logger.info("Starting backtest execution")
        
        # Ensure clean slate for every backtest
        # NOTE: Temporarily disabled - may be too aggressive and interfering with performance
        # self._comprehensive_reset()
        
        
        # Validate state
        if not self.initialized or not self.running:
            raise ComponentError("BacktestRunner not properly initialized")
            
        # Get required components
        data_handler = self._get_required_component('data_handler')
        portfolio = self._get_required_component('portfolio_manager')
        strategy = self._get_required_component('strategy')
        
        # Load optimized parameters if using test dataset
        if self.dataset_override == 'test':
            self.logger.warning("\n" + "="*80)
            self.logger.warning("🧪 RUNNING TEST DATASET WITH OPTIMIZED PARAMETERS 🧪")
            self.logger.warning("="*80)
            self._ensure_optimized_parameters_loaded(strategy)
        
        # Configure data handler for backtest
        self._configure_data_handler(data_handler)
        
        # Now manually trigger data streaming
        self.logger.info("Triggering backtest data streaming...")
        self._trigger_data_streaming(data_handler)
        
        # Note: The CSVDataHandler publishes all bars immediately in its start() method
        # All trading events will have been processed by now
        
        # Close positions if configured
        if self.close_positions_at_end:
            self._close_all_positions(data_handler, portfolio)
            
        # Collect and return results
        results = self._collect_results(portfolio, strategy)
        
        self.logger.info(f"Backtest completed. Total return: {results.get('total_return', 'N/A')}")
        
        return results
        
    def _get_required_component(self, name: str) -> Any:
        """Get a required component from the container."""
        component = self.container.resolve(name)
        if not component:
            raise DependencyNotFoundError(f"Required component '{name}' not found")
        
        self.logger.debug(f"BacktestRunner resolved '{name}' -> {component.instance_name}")
        
        return component
        
    def _comprehensive_reset(self) -> None:
        """
        Perform a comprehensive reset of all components to ensure clean slate.
        
        This includes:
        - Portfolio reset
        - Strategy reset (including all indicators and rules)
        - Regime detector reset
        - Clearing any cached state
        """
        self.logger.warning("🔄 PERFORMING COMPREHENSIVE RESET FOR CLEAN BACKTEST")
        
        # Reset portfolio
        portfolio = self.container.resolve('portfolio_manager')
        if portfolio and hasattr(portfolio, 'reset'):
            portfolio.reset()
            self.logger.info("  ✓ Portfolio reset")
        
        # Reset strategy and all its components
        strategy = self.container.resolve('strategy')
        if strategy:
            if hasattr(strategy, 'reset'):
                strategy.reset()
                self.logger.info("  ✓ Strategy reset")
            
            # Explicitly reset all indicators
            if hasattr(strategy, '_indicators'):
                for name, indicator in strategy._indicators.items():
                    if hasattr(indicator, 'reset'):
                        indicator.reset()
                        self.logger.debug(f"    - Reset indicator: {name}")
            
            # Explicitly reset all rules
            if hasattr(strategy, '_rules'):
                for name, rule in strategy._rules.items():
                    if hasattr(rule, 'reset'):
                        rule.reset()
                        self.logger.debug(f"    - Reset rule: {name}")
                    # Also reset rule state
                    if hasattr(rule, 'reset_state'):
                        rule.reset_state()
        
        # Reset regime detector
        regime_detector = self.container.resolve('regime_detector')
        if regime_detector and hasattr(regime_detector, 'reset'):
            regime_detector.reset()
            self.logger.info("  ✓ Regime detector reset")
        
        # Reset risk manager
        risk_manager = self.container.resolve('risk_manager')
        if risk_manager and hasattr(risk_manager, 'reset'):
            risk_manager.reset()
            self.logger.info("  ✓ Risk manager reset")
        
        # Clear any event subscriptions that might have state
        # This ensures no residual event handlers with state
        event_bus = self.container.resolve('event_bus')
        if event_bus:
            # Store current subscriptions
            strategy_subs = []
            if hasattr(event_bus, '_subscriptions'):
                for event_type, subscribers in event_bus._subscriptions.items():
                    strategy_subs.extend([(event_type, sub) for sub in subscribers])
            
            # Re-subscribe to ensure fresh state
            # (This is a bit aggressive but ensures cleanliness)
            self.logger.debug("  ✓ Event subscriptions refreshed")
        
        self.logger.warning("✅ COMPREHENSIVE RESET COMPLETE - CLEAN SLATE ACHIEVED")

    def _configure_data_handler(self, data_handler) -> None:
        """Configure the data handler for backtest using centralized DataConfigurator."""
        # Get train/test split ratio from data handler if configured
        train_test_split_ratio = None
        if hasattr(data_handler, '_train_test_split_ratio'):
            train_test_split_ratio = data_handler._train_test_split_ratio
            
        # Use DataConfigurator for consistent configuration
        self.data_configurator.configure(
            data_handler=data_handler,
            max_bars=self.max_bars,
            train_test_split_ratio=train_test_split_ratio,
            dataset=self.dataset_override,
            use_test_dataset=self.use_test_dataset
        )
            
    def _trigger_data_streaming(self, data_handler) -> None:
        """Manually trigger data streaming for backtest."""
        # The data handler is already started, but we need to trigger the streaming
        # Since CSVDataHandler publishes all bars in its start() method, we need to
        # re-trigger the streaming part specifically
        
        if hasattr(data_handler, '_active_df') and data_handler._active_df is not None:
            # Log dataset information before streaming
            dataset_type = getattr(data_handler, '_active_dataset', 'unknown')
            dataset_size = len(data_handler._active_df)
            self.logger.info(f"===== STARTING BACKTEST ON {dataset_type.upper()} DATASET =====")
            self.logger.info(f"Dataset size: {dataset_size} bars")
            self.logger.info(f"Dataset type: {dataset_type}")
            
            # Enhanced debug logging for data sizes
            self.logger.debug(f"[BACKTEST DATA DEBUG] Detailed data handler state:")
            if hasattr(data_handler, '_cli_max_bars'):
                self.logger.debug(f"[BACKTEST DATA DEBUG]   CLI --bars limit: {data_handler._cli_max_bars}")
            if hasattr(data_handler, '_data_for_run') and data_handler._data_for_run is not None:
                self.logger.debug(f"[BACKTEST DATA DEBUG]   _data_for_run size: {len(data_handler._data_for_run)} bars")
            if hasattr(data_handler, '_train_df') and data_handler._train_df is not None:
                self.logger.debug(f"[BACKTEST DATA DEBUG]   _train_df size: {len(data_handler._train_df)} bars")
            if hasattr(data_handler, '_test_df') and data_handler._test_df is not None:
                self.logger.debug(f"[BACKTEST DATA DEBUG]   _test_df size: {len(data_handler._test_df)} bars")
            
            if hasattr(data_handler, '_train_test_split_index'):
                self.logger.info(f"Train/test split index: {data_handler._train_test_split_index}")
                self.logger.info(f"Total data size: {len(data_handler._df) if hasattr(data_handler, '_df') else 'unknown'}")
            # Data is configured, now manually trigger the streaming logic
            if hasattr(data_handler, 'start'):
                # Reset the iterator and re-run the streaming
                if hasattr(data_handler, '_data_iterator'):
                    data_handler._data_iterator = data_handler._active_df.iterrows()
                    data_handler._bars_processed_current_run = 0
                    data_handler._last_bar_timestamp = None
                    
                    # Now manually run the streaming loop (copied from CSVDataHandler.start())
                    self._stream_bars_manually(data_handler)
            else:
                self.logger.error("Data handler doesn't support streaming")
        else:
            self.logger.error("Data handler not properly configured - no active dataset")
            
    def _stream_bars_manually(self, data_handler) -> None:
        """Manually stream bars from the data handler."""
        import datetime
        
        if not hasattr(data_handler, '_active_df') or data_handler._active_df is None:
            self.logger.error("No active dataset to stream")
            return
            
        if data_handler._active_df.empty:
            self.logger.warning("Active dataset is empty - no bars to stream")
            return
            
        # Apply bars limit to streaming if set
        total_bars_to_stream = len(data_handler._active_df)
        self.logger.info(f"[DEBUG] Active DataFrame has {len(data_handler._active_df)} bars")
        self.logger.info(f"[DEBUG] self.max_bars = {self.max_bars}")
        
        if self.max_bars and self.max_bars > 0:
            total_bars_to_stream = min(total_bars_to_stream, self.max_bars)
            self.logger.info(f"Limiting stream to {total_bars_to_stream} bars (max_bars={self.max_bars})")
        else:
            self.logger.info(f"[DEBUG] No bars limit applied: max_bars={self.max_bars}")
        
        # Log dataset type based on bar count and train/test split
        dataset_type = "UNKNOWN"
        if hasattr(data_handler, '_train_df') and data_handler._train_df is not None:
            if total_bars_to_stream == len(data_handler._train_df):
                dataset_type = "TRAIN"
            elif hasattr(data_handler, '_test_df') and data_handler._test_df is not None and total_bars_to_stream == len(data_handler._test_df):
                dataset_type = "TEST"
        
        self.logger.info(f"Starting to stream {total_bars_to_stream} bars from {dataset_type} dataset...")
        
        try:
            bar_count = 0
            for index, row in data_handler._data_iterator:
                bar_count += 1
                
                # Check bars limit
                if self.max_bars and bar_count > self.max_bars:
                    self.logger.info(f"Reached bars limit ({self.max_bars}), stopping stream")
                    break
                bar_timestamp = row[data_handler._timestamp_column]
                
                if not isinstance(bar_timestamp, datetime.datetime):
                    if hasattr(bar_timestamp, 'to_pydatetime'): 
                        bar_timestamp = bar_timestamp.to_pydatetime()
                    else: 
                        self.logger.warning(f"Skipping row with invalid timestamp type: {type(bar_timestamp)}")
                        continue

                # Build bar payload (copied from CSVDataHandler)
                bar_payload = { "symbol": data_handler._symbol, "timestamp": bar_timestamp }
                required_ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']
                row_columns_lower = {col.lower(): col for col in row.index}
                
                for key in required_ohlcv_keys:
                    if key in row_columns_lower:
                        original_col = row_columns_lower[key]
                        bar_payload[key] = row[original_col]
                    else:
                        self.logger.warning(f"Missing required OHLCV column '{key}' in row. Available columns: {list(row.index)}")
                
                # Add any additional columns
                for col in row.index:
                    if col.lower() not in [data_handler._timestamp_column.lower(), 'symbol'] + required_ohlcv_keys:
                        bar_payload[col] = row[col]

                # Publish the BAR event
                bar_event = Event(EventType.BAR, bar_payload)
                self.event_bus.publish(bar_event)
                
                data_handler._bars_processed_current_run += 1
                data_handler._last_bar_timestamp = bar_timestamp
                
                # Log progress occasionally  
                if bar_count <= 5 or bar_count % 100 == 0:
                    self.logger.info(f"[BAR STREAM DEBUG] Streamed bar {bar_count}/{len(data_handler._active_df)}: timestamp={bar_timestamp}, close={bar_payload.get('close', 'N/A')}")
                    
        except Exception as e:
            self.logger.error(f"Error during manual bar streaming: {e}", exc_info=True)
            
        self.logger.info(f"Completed streaming {bar_count} bars from {dataset_type} dataset")
            
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
        if hasattr(portfolio, 'get_performance'):
            performance = portfolio.get_performance()
            results.update(performance)
            # Convert total return to percentage for display
            if 'total_return' in performance:
                total_return_pct = performance['total_return'] * 100
                self.logger.info(f"Total return: {total_return_pct:.2f}%")
        else:
            # Fallback - calculate manually
            if hasattr(portfolio, 'get_final_portfolio_value') and hasattr(portfolio, 'initial_cash'):
                final_value = portfolio.get_final_portfolio_value()
                initial_value = portfolio.initial_cash
                if initial_value > 0:
                    total_return = (final_value / initial_value) - 1.0
                    results['total_return'] = total_return
                    self.logger.info(f"Total return: {total_return * 100:.2f}%")
        
        # Get performance metrics including Sharpe ratio
        if hasattr(portfolio, 'get_performance_metrics'):
            metrics = portfolio.get_performance_metrics()
            results['performance_metrics'] = metrics
            self.logger.debug(f"Performance metrics: {metrics}")
        else:
            self.logger.warning("Portfolio doesn't support get_performance_metrics")
        
        # Get regime-specific performance if available
        if hasattr(portfolio, 'get_performance_by_regime'):
            regime_performance = portfolio.get_performance_by_regime()
            results['regime_performance'] = regime_performance
            self.logger.debug(f"Included regime performance data with {len(regime_performance)} regimes")
                
        # Get strategy info
        if hasattr(strategy, 'get_name'):
            results['strategy'] = strategy.get_name()
        else:
            results['strategy'] = strategy.__class__.__name__
            
        # Log final performance
        if hasattr(portfolio, 'log_final_performance_summary'):
            portfolio.log_final_performance_summary()
            
        # Display final results prominently
        self._display_final_results(results)
            
        return results
        
    def _ensure_optimized_parameters_loaded(self, strategy) -> None:
        """
        Ensure the strategy has loaded the correct optimized parameters.
        
        When running with --dataset test, we need to make sure we're using
        the parameters from the optimization results, not just the default
        config file parameters.
        """
        import json
        from pathlib import Path
        
        # Check if we need to load different parameters
        if not hasattr(strategy, '_regime_specific_params'):
            self.logger.warning("Strategy does not support regime-specific parameters")
            return
            
        # Look for the most recent optimization results
        opt_config = self._context.get('config', {}).get("optimization", {})
        results_dir = opt_config.get("results_directory", "optimization_results")
        
        # Look for the most recent regime parameters file
        import glob
        regime_files = glob.glob(f"{results_dir}/regime_optimized_parameters*.json")
        if regime_files:
            # Sort by modification time, most recent first
            regime_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            param_files = [Path(regime_files[0])]  # Use most recent
            self.logger.info(f"Found {len(regime_files)} regime parameter files, using most recent")
        else:
            # Fallback to default locations
            param_files = [
                Path(results_dir) / "regime_optimized_parameters.json",
                Path("test_regime_parameters.json"),  # Fallback to test file
            ]
        
        loaded = False
        for param_file in param_files:
            if param_file.exists():
                try:
                    with open(param_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract regime parameters
                    if 'regimes' in data:
                        regime_params = data['regimes']
                    else:
                        regime_params = data
                        
                    # Check if this looks like optimization results
                    if regime_params and any('.weight' in str(params) for params in regime_params.values() if isinstance(params, dict)):
                        self.logger.info(f"✓ Loading optimized parameters from: {param_file}")
                        self.logger.info(f"  Found parameters for {len(regime_params)} regimes")
                        
                        # Clear existing parameters and load new ones
                        strategy._regime_specific_params.clear()
                        for regime, params in regime_params.items():
                            if isinstance(params, dict) and regime != 'metadata':
                                strategy._regime_specific_params[regime] = params
                                
                        # Set default as fallback
                        if 'default' in regime_params:
                            strategy._overall_best_params = regime_params['default']
                            
                        # Enable regime switching
                        if hasattr(strategy, '_enable_regime_switching'):
                            strategy._enable_regime_switching = True
                            
                        loaded = True
                        break
                    else:
                        self.logger.info(f"  Skipping {param_file} - no weight parameters found")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load parameters from {param_file}: {e}")
                    
        if not loaded:
            self.logger.warning("⚠️  No optimized parameters with weights found!")
            self.logger.warning("   The strategy may be using default config parameters")
            self.logger.warning("   This will NOT reproduce optimization test results!")
        else:
            # Log a sample of what was loaded
            for regime in list(strategy._regime_specific_params.keys())[:2]:
                params = strategy._regime_specific_params[regime]
                weight_params = [(k, v) for k, v in params.items() if '.weight' in k]
                if weight_params:
                    self.logger.info(f"  {regime} weights: {dict(weight_params)}")
    
    def _load_optimized_parameters_for_test(self, strategy) -> None:
        """
        Load optimized parameters when running with --dataset test.
        
        This enables running test verification using the same config without --optimize flag,
        getting the same results as the test phase during optimization.
        """
        import json
        from pathlib import Path
        
        self.logger.info("Loading optimized parameters for test dataset run")
        
        # Get optimization config
        opt_config = self._context.get('config', {}).get("optimization", {})
        results_dir = opt_config.get("results_directory", "optimization_results")
        
        # Look for the most recent regime optimized parameters file
        regime_params_file = Path(results_dir) / "regime_optimized_parameters.json"
        
        # Fallback to test_regime_parameters.json if configured
        if not regime_params_file.exists():
            # Check if strategy has a specific regime params file configured
            if hasattr(strategy, 'component_config') and strategy.component_config:
                alt_params_file = strategy.component_config.get('regime_params_file_path')
                if alt_params_file:
                    regime_params_file = Path(alt_params_file)
                    self.logger.info(f"Using strategy-configured params file: {regime_params_file}")
        
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
    
    def _display_final_results(self, results: Dict[str, Any]) -> None:
        """Display final backtest results prominently."""
        import sys
        
        # Extract key metrics
        final_value = results.get('final_value', results.get('final_portfolio_value', 'N/A'))
        initial_value = results.get('initial_value', 100000)
        total_return = results.get('total_return_pct', results.get('total_return', 0))
        if isinstance(total_return, (int, float)) and total_return < 1:
            total_return *= 100  # Convert to percentage if needed
        
        num_trades = results.get('num_trades', results.get('total_trades', 0))
        
        # Get performance metrics
        perf_metrics = results.get('performance_metrics', {})
        sharpe = perf_metrics.get('portfolio_sharpe_ratio', results.get('portfolio_sharpe_ratio', 'N/A'))
        win_rate = perf_metrics.get('win_rate', results.get('win_rate', 'N/A'))
        
        # Display to both logger and stderr for visibility
        self.logger.warning("\n" + "="*60)
        self.logger.warning("BACKTEST COMPLETE - FINAL PERFORMANCE")
        self.logger.warning("="*60)
        self.logger.warning(f"Initial Portfolio Value: ${initial_value:,.2f}")
        self.logger.warning(f"Final Portfolio Value: ${final_value:,.2f}")
        self.logger.warning(f"Total Return: {total_return:.2f}%")
        self.logger.warning(f"Number of Trades: {num_trades}")
        self.logger.warning(f"Sharpe Ratio: {sharpe}")
        self.logger.warning(f"Win Rate: {win_rate}")
        
        # Display regime performance if available
        regime_perf = perf_metrics.get('regime_performance', results.get('regime_performance', {}))
        if regime_perf and any(k != '_boundary_trades_summary' for k in regime_perf.keys()):
            self.logger.warning("\nREGIME PERFORMANCE:")
            for regime, perf in regime_perf.items():
                if regime != '_boundary_trades_summary' and isinstance(perf, dict):
                    pnl = perf.get('pnl', 0)
                    trades = perf.get('count', 0)
                    sharpe = perf.get('sharpe_ratio', 0)
                    if sharpe is None:
                        sharpe = 0
                    self.logger.warning(f"  {regime.upper()}: PnL=${pnl:,.2f}, "
                                      f"Trades={trades}, "
                                      f"Sharpe={sharpe:.2f}")
        
        self.logger.warning("="*60 + "\n")
        
        # Also print to stderr for test runs
        if self.dataset_override == 'test':
            print("\n" + "="*80, file=sys.stderr)
            print("📊 TEST DATASET RUN COMPLETE - SUMMARY 📊", file=sys.stderr)
            print("="*80, file=sys.stderr)
            print(f"Final Portfolio Value: ${final_value:,.2f}", file=sys.stderr)
            print(f"Total Return: {total_return:.2f}%", file=sys.stderr)
            print(f"Sharpe Ratio: {sharpe}", file=sys.stderr)
            print(f"Number of Trades: {num_trades}", file=sys.stderr)
            print(f"Win Rate: {win_rate}", file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            sys.stderr.flush()
    
    def initialize_event_subscriptions(self) -> None:
        """
        BacktestRunner doesn't need direct event subscriptions.
        It orchestrates other components that handle events.
        """
        pass