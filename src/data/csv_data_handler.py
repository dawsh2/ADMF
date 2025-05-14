# src/data/csv_data_handler.py
import logging
import pandas as pd
from pathlib import Path
from typing import Optional # Make sure Optional is imported

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ComponentError, ConfigurationError

class CSVDataHandler(BaseComponent):
    def __init__(self, 
                 instance_name: str, 
                 config_loader, 
                 event_bus, 
                 component_config_key: str,
                 max_bars: Optional[int] = None): # NEW: max_bars parameter
        super().__init__(instance_name, config_loader, component_config_key) 
        
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error("EventBus instance is required for CSVDataHandler.")
            raise ValueError("EventBus instance is required for CSVDataHandler.")

        self._file_path_str: str = self.get_specific_config("file_path")
        self._symbol: str = self.get_specific_config("symbol")
        self._timestamp_col = self.get_specific_config("timestamp_column", "timestamp")
        self._open_col = self.get_specific_config("open_column", "Open")
        self._high_col = self.get_specific_config("high_column", "High")
        self._low_col = self.get_specific_config("low_column", "Low")
        self._close_col = self.get_specific_config("close_column", "Close")
        self._volume_col = self.get_specific_config("volume_column", "Volume")

        self._max_bars = max_bars # NEW: Store max_bars

        self._data_frame: Optional[pd.DataFrame] = None # Type hint for clarity
        self._data_iterator = None

        if not self._file_path_str:
            raise ConfigurationError(f"Missing 'file_path' in configuration for {self.name}")
        if not self._symbol:
            raise ConfigurationError(f"Missing 'symbol' in configuration for {self.name}")

        project_root = Path().resolve() 
        self._absolute_file_path = project_root / self._file_path_str
        
        log_msg = (
            f"CSVDataHandler '{self.name}' configured for symbol '{self._symbol}' "
            f"using file '{self._absolute_file_path}'. Timestamp column: '{self._timestamp_col}'."
        )
        if self._max_bars is not None and self._max_bars > 0:
            log_msg += f" Processing up to {self._max_bars} bars."
        self.logger.info(log_msg)


    def setup(self):
        self.logger.info(f"Setting up CSVDataHandler '{self.name}'...")
        try:
            if not self._absolute_file_path.exists():
                self.logger.error(f"CSV file not found at path: {self._absolute_file_path}")
                self.state = BaseComponent.STATE_FAILED
                raise ComponentError(f"CSV file not found: {self._absolute_file_path}")

            dtypes = {
                self._open_col: float, self._high_col: float, self._low_col: float,
                self._close_col: float, self._volume_col: float, 
            }
            
            self._data_frame = pd.read_csv(
                self._absolute_file_path, 
                dtype=dtypes,
                parse_dates=[self._timestamp_col]
            )
            self.logger.info(f"Successfully loaded CSV file: {self._absolute_file_path}. Full shape: {self._data_frame.shape}")

            # NEW: Slice DataFrame if max_bars is set
            if self._max_bars is not None and self._max_bars > 0 and len(self._data_frame) > self._max_bars:
                self._data_frame = self._data_frame.head(self._max_bars)
                self.logger.info(f"Data sliced to first {self._max_bars} bars. New shape: {self._data_frame.shape}")
            
            if self._timestamp_col not in self._data_frame.columns:
                raise ConfigurationError(f"Timestamp column '{self._timestamp_col}' not found in CSV after parsing.")

            if self._data_frame[self._timestamp_col].isnull().any():
                num_failed = self._data_frame[self._timestamp_col].isnull().sum()
                self.logger.error(f"{num_failed} rows failed datetime parsing in column '{self._timestamp_col}'. Please check CSV format.")
                self.state = BaseComponent.STATE_FAILED
                raise ComponentError(f"Datetime parsing failed for {num_failed} rows in column '{self._timestamp_col}' from {self._absolute_file_path}")
            
            self._data_iterator = self._data_frame.iterrows()
            
            self.state = BaseComponent.STATE_INITIALIZED
            self.logger.info(f"CSVDataHandler '{self.name}' setup complete. {len(self._data_frame)} bars loaded for processing. State: {self.state}")

        # ... (exception handling in setup as before) ...
        except FileNotFoundError:
            self.logger.error(f"CSV file not found at path: {self._absolute_file_path}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            raise ComponentError(f"CSV file not found during setup: {self._absolute_file_path}")
        except KeyError as e: 
            self.logger.error(f"Missing expected column in CSV: {e}. Check config and CSV header.", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            raise ConfigurationError(f"Missing column {e} in CSV {self._absolute_file_path}")
        except Exception as e:
            self.logger.error(f"Error during CSVDataHandler setup for '{self.name}': {e}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            raise ComponentError(f"Failed to setup CSVDataHandler '{self.name}': {e}") from e


    def start(self):
        # ... (start method remains largely the same, it will now iterate over the potentially sliced _data_iterator)
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start CSVDataHandler '{self.name}' from state '{self.state}'. Expected INITIALIZED.")
            return

        self.logger.info(f"CSVDataHandler '{self.name}' starting to publish BAR events...")
        self.state = BaseComponent.STATE_STARTED

        if self._data_iterator is None: 
            self.logger.error("Data iterator not initialized. Setup might have failed or not been called.")
            self.state = BaseComponent.STATE_FAILED
            return

        bars_published = 0
        try:
            for index, row in self._data_iterator: # This now iterates over the (potentially sliced) DataFrame
                payload = {
                    "symbol": self._symbol,
                    "timestamp": row[self._timestamp_col], 
                    "open": float(row[self._open_col]),
                    "high": float(row[self._high_col]),
                    "low": float(row[self._low_col]),
                    "close": float(row[self._close_col]),
                    "volume": int(row[self._volume_col]) 
                }
                bar_event = Event(EventType.BAR, payload)
                self._event_bus.publish(bar_event)
                bars_published += 1
            self.logger.info(f"Finished publishing {bars_published} BAR events for '{self.name}'.")
        except Exception as e:
            self.logger.error(f"Error during BAR event publishing for '{self.name}': {e}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
        finally:
            if self.state == BaseComponent.STATE_STARTED : 
                self.state = BaseComponent.STATE_STOPPED 
                self.logger.info(f"CSVDataHandler '{self.name}' completed data streaming. State: {self.state}")


    def stop(self):
        # ... (stop method remains the same)
        self.logger.info(f"Stopping CSVDataHandler '{self.name}'...")
        self._data_frame = None
        self._data_iterator = None
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"CSVDataHandler '{self.name}' stopped. State: {self.state}")
