# src/data/csv_data_handler.py
import logging
import pandas as pd
import datetime
from typing import Optional, Iterator, Tuple, Any

from src.core.component_base import ComponentBase
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError

class CSVDataHandler(ComponentBase):
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Initialize internal state (no external dependencies)
        self._cli_max_bars = None
        
        # Configuration parameters (will be set in initialize)
        self._symbol: Optional[str] = None
        self._csv_file_path: Optional[str] = None
        self._timestamp_column: str = "timestamp"
        
        self._train_test_split_ratio: Optional[float] = None
        self._regime_column_to_filter: Optional[str] = None
        self._target_regime_value: Optional[str] = None

        # Data state
        self._data_for_run: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._active_df: Optional[pd.DataFrame] = None
        self._data_iterator: Optional[Iterator[Tuple[Any, pd.Series]]] = None
        self._bars_processed_current_run = 0
        self._last_bar_timestamp: Optional[datetime.datetime] = None
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Load configuration
        self._symbol = self.get_specific_config("symbol")
        self._csv_file_path = self.get_specific_config("csv_file_path")
        self._timestamp_column = self.get_specific_config("timestamp_column", "timestamp")
        
        # Debug logging
        if hasattr(self, 'logger') and self.logger:
            self.logger.debug(f"CSVDataHandler._initialize: symbol={self._symbol}, csv_file_path={self._csv_file_path}")
        
        if not self._symbol or not self._csv_file_path:
            raise ConfigurationError(f"Missing 'symbol' or 'csv_file_path' for {self.instance_name}")
        
        # Get CLI parameter if provided
        if hasattr(self, 'context') and self.context:
            self._cli_max_bars = self.context.get("max_bars", None)
        
        self._train_test_split_ratio = self.get_specific_config("train_test_split_ratio", None)
        if self._train_test_split_ratio is not None and not (0 < self._train_test_split_ratio < 1):
            raise ConfigurationError(
                f"'{self.instance_name}': 'train_test_split_ratio' must be between 0 and 1 (exclusive). "
                f"Got {self._train_test_split_ratio}"
            )
            
        self._regime_column_to_filter = self.get_specific_config("regime_column_to_filter", None)
        self._target_regime_value = self.get_specific_config("target_regime_value", None)
        
        self.logger.info(self._build_config_log_message())
    
    def get_specific_config(self, key: str, default=None):
        """Helper method to get configuration values."""
        # First try component_config set by ComponentBase
        if hasattr(self, 'component_config') and self.component_config:
            value = self.component_config.get(key, None)
            if value is not None:
                return value
        
        # Fall back to config_loader
        if not self.config_loader:
            return default
        config_key = self.config_key or self.instance_name
        config = self.config_loader.get_component_config(config_key)
        return config.get(key, default) if config else default

    def _build_config_log_message(self) -> str:
        parts = [
            f"CSVDataHandler '{self.instance_name}' configured for symbol '{self._symbol}' using file '{self._csv_file_path}'.",
            f"Timestamp column: '{self._timestamp_column}'."
        ]
        if self._cli_max_bars is not None:
            parts.append(f"CLI --bars: Total data for this run initially limited to first {self._cli_max_bars} bars.")
        if self._regime_column_to_filter and self._target_regime_value:
            parts.append(f"Regime pre-filtering: for '{self._target_regime_value}' in column '{self._regime_column_to_filter}' (applied after --bars).")
        if self._train_test_split_ratio is not None:
            parts.append(f"Train/test split ratio: {self._train_test_split_ratio*100:.0f}% train (applied to the --bars limited and/or regime-filtered data).")
        return " ".join(parts)

    def _start(self):
        self.logger.info(f"Setting up CSVDataHandler '{self.instance_name}'...")
        try:
            df_loaded = pd.read_csv(self._csv_file_path)
            self.logger.info(f"Successfully loaded CSV file: {self._csv_file_path}. Initial full shape: {df_loaded.shape}")

            if self._timestamp_column not in df_loaded.columns:
                raise ConfigurationError(f"Timestamp column '{self._timestamp_column}' not found in CSV.")

            if not pd.api.types.is_datetime64_any_dtype(df_loaded[self._timestamp_column]):
                try:
                    df_loaded[self._timestamp_column] = pd.to_datetime(df_loaded[self._timestamp_column])
                except Exception as e:
                    self.logger.error(f"Could not parse timestamp column '{self._timestamp_column}'. Error: {e}", exc_info=True)
                    raise ConfigurationError(f"Timestamp parsing failed for '{self._timestamp_column}'.") from e
            
            if df_loaded[self._timestamp_column].dt.tz is None:
                df_loaded[self._timestamp_column] = df_loaded[self._timestamp_column].dt.tz_localize('UTC')
            else:
                df_loaded[self._timestamp_column] = df_loaded[self._timestamp_column].dt.tz_convert('UTC')
            
            df_loaded = df_loaded.sort_values(by=self._timestamp_column).reset_index(drop=True)
            self.logger.info(f"Data sorted by timestamp column '{self._timestamp_column}'.")

            # Step 1: Apply --bars limit to the initially loaded data
            self._data_for_run = df_loaded # Start with the full loaded data
            if self._cli_max_bars is not None and self._cli_max_bars != 0:
                if self._cli_max_bars > 0:
                    # Positive value: take first N bars
                    if len(self._data_for_run) > self._cli_max_bars:
                        self.logger.info(f"Applying --bars limit: Using first {self._cli_max_bars} bars from loaded data (was {len(self._data_for_run)}).")
                        self._data_for_run = self._data_for_run.head(self._cli_max_bars)
                    else:
                        self.logger.info(f"Dataset length ({len(self._data_for_run)}) is within or equal to --bars limit ({self._cli_max_bars}). Using all {len(self._data_for_run)} bars.")
                else:
                    # Negative value: take last N bars
                    abs_bars = abs(self._cli_max_bars)
                    if len(self._data_for_run) > abs_bars:
                        self.logger.info(f"Applying --bars limit: Using last {abs_bars} bars from loaded data (was {len(self._data_for_run)}).")
                        self._data_for_run = self._data_for_run.tail(abs_bars)
                    else:
                        self.logger.info(f"Dataset length ({len(self._data_for_run)}) is within or equal to --bars limit ({abs_bars}). Using all {len(self._data_for_run)} bars.")
            
            # Step 2: Optional Regime Filtering (applies to the now --bars limited self._data_for_run)
            if self._regime_column_to_filter and self._target_regime_value:
                if self._regime_column_to_filter not in self._data_for_run.columns:
                    raise ConfigurationError(
                        f"Regime filter column '{self._regime_column_to_filter}' not found in effective dataset (shape: {self._data_for_run.shape})."
                    )
                self.logger.info(
                    f"Applying regime filter to current dataset ({len(self._data_for_run)} bars): keeping rows where '{self._regime_column_to_filter}' is '{self._target_regime_value}'."
                )
                original_rows = len(self._data_for_run)
                self._data_for_run = self._data_for_run[self._data_for_run[self._regime_column_to_filter] == self._target_regime_value].copy()
                self.logger.info(
                    f"Regime filter applied. Rows in dataset reduced from {original_rows} to {len(self._data_for_run)}."
                )
                if self._data_for_run.empty:
                    self.logger.warning(f"Dataset is empty after applying regime filter for '{self._target_regime_value}'.")

            # Step 3: Train/Test Split on self._data_for_run (which is already limited by --bars and/or regime)
            if self._train_test_split_ratio is not None and not self._data_for_run.empty:
                split_point = int(len(self._data_for_run) * self._train_test_split_ratio)
                self._train_df = self._data_for_run.iloc[:split_point].copy()
                self._test_df = self._data_for_run.iloc[split_point:].copy()
                
                # DEBUG: Log split details to trace the 74% vs 80% issue
                self.logger.warning(f"SPLIT_DEBUG: Using ratio {self._train_test_split_ratio}, split_point={split_point}")
                if not self._test_df.empty:
                    first_test_date = self._test_df.iloc[0]['timestamp'] if 'timestamp' in self._test_df.columns else 'unknown'
                    self.logger.warning(f"SPLIT_DEBUG: First test timestamp: {first_test_date}")
                
                self.logger.info(f"Current dataset (size {len(self._data_for_run)}) split into: Train ({len(self._train_df)} bars), Test ({len(self._test_df)} bars).")
            else: 
                self._train_df = self._data_for_run.copy() if self._data_for_run is not None else pd.DataFrame()
                self._test_df = pd.DataFrame() 
                self.logger.info("Using entire current dataset as training data (no test split defined or data was empty).")

            self._active_df = None 
            self._data_iterator = iter([])
            self._bars_processed_current_run = 0
            self._last_bar_timestamp = None 

            # Component is now initialized
            self.logger.info(f"CSVDataHandler '{self.instance_name}' setup complete. Data loaded and splits prepared. Call set_active_dataset() to begin.")

        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {self._csv_file_path}")
            # Mark as failed
            raise ConfigurationError(f"CSVDataHandler: File not found {self._csv_file_path}")
        except KeyError as e: 
            self.logger.error(f"KeyError during CSVDataHandler setup (likely missing column '{e}' in CSV).", exc_info=True)
            # Mark as failed
            raise ConfigurationError(f"CSVDataHandler: Missing expected column in CSV - {e}")
        except Exception as e:
            self.logger.error(f"Error loading or processing CSV {self._csv_file_path}: {e}", exc_info=True)
            # Mark as failed
            raise ComponentError(f"CSVDataHandler failed setup: {e}") from e

    def set_max_bars(self, max_bars: int):
        """Set the maximum number of bars to stream."""
        self._cli_max_bars = max_bars
        self.logger.info(f"CSVDataHandler max bars set to: {max_bars}")
        
        # If we already have an active dataset, we need to re-limit it
        if self._active_df is not None and max_bars:
            original_size = len(self._active_df)
            if max_bars > 0 and original_size > max_bars:
                self._active_df = self._active_df.head(max_bars)
                self._data_iterator = self._active_df.iterrows()
                self.logger.info(f"Active dataset re-limited from {original_size} to {len(self._active_df)} bars")
    
    def set_active_dataset(self, dataset_type: str = "full"):
        """
        Sets the active dataset for iteration from the pre-processed and pre-split DataFrames.
        """
        self.logger.info(f"=== SWITCHING TO {dataset_type.upper()} DATASET ===")
        
        # DEBUG: Log all available dataset sizes
        self.logger.debug(f"Available datasets - train: {len(self._train_df) if self._train_df is not None else 'None'}, test: {len(self._test_df) if self._test_df is not None else 'None'}")
        
        selected_df: Optional[pd.DataFrame] = None
        if dataset_type.lower() == "train":
            selected_df = self._train_df
            self.logger.debug(f"Selected train_df with {len(selected_df)} rows")
        elif dataset_type.lower() == "test":
            selected_df = self._test_df
            self.logger.debug(f"Selected test_df with {len(selected_df)} rows")
        elif dataset_type.lower() == "full": # "full" now refers to the --bars limited, regime-filtered data
            selected_df = self._data_for_run 
            self.logger.debug(f"Selected data_for_run with {len(selected_df)} rows")
        else:
            self.logger.warning(f"Unknown dataset_type '{dataset_type}'. Defaulting to 'full' effective dataset.")
            selected_df = self._data_for_run

        if selected_df is None or selected_df.empty:
            self.logger.warning(f"Dataset '{dataset_type}' is empty or not available. No bars will be processed.")
            self._active_df = pd.DataFrame(columns=self._data_for_run.columns if self._data_for_run is not None and not self._data_for_run.empty else [])
            self._data_iterator = iter([])
        else:
            self._active_df = selected_df.copy() # Use a copy so iterator changes don't affect _train_df etc.
            self._data_iterator = self._active_df.iterrows()
            self.logger.debug(f"Set active_df with {len(self._active_df)} rows")

        self._bars_processed_current_run = 0
        self._last_bar_timestamp = None
        self.logger.debug(f"Active dataset '{dataset_type}' ready with {len(self._active_df)} bars.")

    # start(), stop(), get_last_timestamp() methods remain the same.
    # The bar payload construction in start() also remains the same.
    def start(self):
        """Start publishing bar events."""
        super().start()
        
        if self._active_df is None: # Check if set_active_dataset was called
             self.logger.error(f"No active dataset selected for {self.instance_name}. Call set_active_dataset() after setup.")
             # Cannot proceed without dataset
             return
        
        if self._active_df.empty:
            # DEBUG: Add detailed logging to understand why dataset is empty
            self.logger.warning(f"Active dataset is empty!")
            self.logger.debug(f"train_df size: {len(self._train_df) if self._train_df is not None else 'None'}")
            self.logger.debug(f"test_df size: {len(self._test_df) if self._test_df is not None else 'None'}")
            self.logger.debug(f"data_for_run size: {len(self._data_for_run) if self._data_for_run is not None else 'None'}")
            self.logger.info(f"CSVDataHandler '{self.instance_name}' active dataset is empty. No BAR events will be published.")
            self.logger.info(f"CSVDataHandler '{self.instance_name}' completed data streaming (0 bars).")
            return

        # IMPORTANT: Reset the data iterator each time we start to ensure fresh iteration
        self.logger.debug(f"Resetting data iterator for fresh streaming")
        self.logger.debug(f"Creating fresh iterator from {len(self._active_df)} rows")
        self._data_iterator = self._active_df.iterrows()
        self._bars_processed_current_run = 0
        self._last_bar_timestamp = None
        self.logger.debug(f"Reset complete - bars_processed: {self._bars_processed_current_run}")

        self.logger.debug(f"Starting to publish {len(self._active_df)} BAR events...")
        # Component is now running
        
        try:
            self.logger.debug(f"About to iterate through {len(self._active_df)} rows")
            row_count = 0
            for index, row in self._data_iterator:
                row_count += 1
                if row_count <= 3 or row_count % 50 == 0:  # Log first few and every 50th
                    pass  # Remove verbose row processing logs
                bar_timestamp = row[self._timestamp_column]
                if not isinstance(bar_timestamp, datetime.datetime):
                    if hasattr(bar_timestamp, 'to_pydatetime'): bar_timestamp = bar_timestamp.to_pydatetime()
                    else: 
                        self.logger.warning(f"Skipping row with invalid timestamp type: {type(bar_timestamp)}")
                        continue

                bar_payload = { "symbol": self._symbol, "timestamp": bar_timestamp }
                required_ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']
                row_columns_lower = {col.lower(): col for col in row.index}

                valid_bar = True
                for key in required_ohlcv_keys:
                    actual_col_name = row_columns_lower.get(key)
                    if actual_col_name and actual_col_name in row:
                        try:
                            bar_payload[key] = float(row[actual_col_name])
                        except (ValueError, TypeError):
                            self.logger.error(f"Could not convert {key} ('{row[actual_col_name]}') to float for bar at {bar_timestamp}. Setting to NaN.")
                            bar_payload[key] = float('nan') # Or skip bar
                            valid_bar = False; break 
                    else: 
                         self.logger.warning(f"BAR event for '{self._symbol}' at {bar_timestamp} is missing standard key '{key}' in CSV headers. Setting to NaN.")
                         bar_payload[key] = float('nan')
                         valid_bar = False; break
                if not valid_bar:
                    self.logger.warning(f"Skipping bar at {bar_timestamp} due to missing/invalid essential OHLCV data.")
                    continue
                
                for original_col_name, value in row.items():
                    lower_col_name = original_col_name.lower()
                    if lower_col_name != self._timestamp_column.lower() and lower_col_name not in bar_payload: # Avoid overwriting already set/processed fields
                        bar_payload[lower_col_name] = value
                
                bar_event = Event(EventType.BAR, bar_payload)
                if (self._bars_processed_current_run + 1) % 50 == 0:  # Log every 50 bars
                    self.logger.debug(f"Published BAR event {self._bars_processed_current_run + 1}/{len(self._active_df)}")
                self._event_bus.publish(bar_event)
                self._bars_processed_current_run += 1
                self._last_bar_timestamp = bar_timestamp
            
            self.logger.info(f"Finished publishing {self._bars_processed_current_run} BAR events for '{self.instance_name}'.")

        except Exception as e:
            self.logger.error(f"Error during BAR event publishing for '{self.instance_name}': {e}", exc_info=True)
            # Mark as failed
        finally:
            self.logger.info(f"CSVDataHandler '{self.instance_name}' completed data streaming for active dataset.")

    def _stop(self):
        """Stop the data handler."""
        self.logger.info(f"Stopping CSVDataHandler '{self.instance_name}'...")
        self._data_iterator = None
        self.logger.info(f"CSVDataHandler '{self.instance_name}' stopped.")

    def teardown(self):
        """Clean up resources."""
        super().teardown()
        self._data_for_run = None
        self._train_df = None
        self._test_df = None
        self._active_df = None
        self._data_iterator = None
    
    def get_last_timestamp(self) -> Optional[datetime.datetime]:
        return self._last_bar_timestamp

    @property
    def test_df_exists_and_is_not_empty(self) -> bool:
        return self._test_df is not None and not self._test_df.empty
