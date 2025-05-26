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
        
        # Validate state
        if not self.initialized or not self.running:
            raise ComponentError("BacktestRunner not properly initialized")
            
        # Get required components
        data_handler = self._get_required_component('data_handler')
        portfolio = self._get_required_component('portfolio_manager')
        strategy = self._get_required_component('strategy')
        
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
        return component
        
    def _configure_data_handler(self, data_handler) -> None:
        """Configure the data handler for backtest."""
        # IMPORTANT: Apply max_bars BEFORE train/test split to match optimization behavior
        # This ensures we split the same limited dataset
        if self.max_bars and hasattr(data_handler, 'set_max_bars'):
            self.logger.info(f"Limiting to {self.max_bars} bars BEFORE train/test split")
            data_handler.set_max_bars(self.max_bars)
        
        # Now check if we need to apply train/test split
        if self.dataset_override in ['train', 'test']:
            # User wants train or test data, ensure split is applied
            if hasattr(data_handler, '_train_test_split_ratio') and data_handler._train_test_split_ratio:
                # Split ratio already configured, apply it
                if hasattr(data_handler, 'apply_train_test_split'):
                    self.logger.info(f"Applying train/test split with ratio {data_handler._train_test_split_ratio}")
                    data_handler.apply_train_test_split(data_handler._train_test_split_ratio)
            else:
                self.logger.warning("Dataset 'train' or 'test' requested but no train_test_split_ratio configured")
        
        # Set active dataset
        if hasattr(data_handler, 'set_active_dataset'):
            # Check if we have train/test split
            has_train_test_split = (hasattr(data_handler, 'train_df_exists_and_is_not_empty') and 
                                   data_handler.train_df_exists_and_is_not_empty and
                                   hasattr(data_handler, 'test_df_exists_and_is_not_empty') and 
                                   data_handler.test_df_exists_and_is_not_empty)
            
            if has_train_test_split:
                if self.use_test_dataset:
                    self.logger.info("Using test dataset for backtest")
                    data_handler.set_active_dataset('test')
                else:
                    self.logger.info("Using train dataset for backtest")
                    data_handler.set_active_dataset('train')
            else:
                if self.dataset_override:
                    self.logger.info(f"Requested dataset '{self.dataset_override}' but no train/test split available, using full dataset")
                else:
                    self.logger.info("No train/test split available, using full dataset")
                data_handler.set_active_dataset('full')
            
    def _trigger_data_streaming(self, data_handler) -> None:
        """Manually trigger data streaming for backtest."""
        # The data handler is already started, but we need to trigger the streaming
        # Since CSVDataHandler publishes all bars in its start() method, we need to
        # re-trigger the streaming part specifically
        
        if hasattr(data_handler, '_active_df') and data_handler._active_df is not None:
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
        
        self.logger.info(f"Starting to stream {total_bars_to_stream} bars...")
        
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
                if bar_count <= 3 or bar_count % 100 == 0:
                    self.logger.debug(f"Streamed bar {bar_count}/{len(data_handler._active_df)}: {bar_timestamp}")
                    
        except Exception as e:
            self.logger.error(f"Error during manual bar streaming: {e}", exc_info=True)
            
        self.logger.info(f"Completed streaming {bar_count} bars")
            
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
            
        return results
        
    def initialize_event_subscriptions(self) -> None:
        """
        BacktestRunner doesn't need direct event subscriptions.
        It orchestrates other components that handle events.
        """
        pass