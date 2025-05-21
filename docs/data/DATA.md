# Data Module Documentation

## Overview

The Data module is responsible for loading, processing, and providing market data to the ADMF-Trader system. It manages data isolation between training and testing periods, ensures proper data propagation through the event system, and handles the efficient processing of market data.

## Key Components

### 1. Data Models

- **Bar Class**: Represents OHLCV (Open, High, Low, Close, Volume) market data with standardized fields and conversion methods
- **Timeframe Utilities**: Tools for handling different time frames and resolutions
- **Standardized Data Formats**: Consistent representation of market data across the system

### 2. Data Handler Interface

The DataHandler is the core abstraction for loading and managing market data:

```python
class DataHandler(Component):
    """
    Abstract interface for data management components.
    
    Responsible for loading market data, providing access to that data,
    and publishing data events to the system.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        self.symbols = []
        self.current_bar_index = 0
        
    def load_data(self, symbols):
        """
        Load data for specified symbols.
        
        Args:
            symbols: List of symbols to load
            
        Returns:
            bool: Success or failure
        """
        raise NotImplementedError
        
    def update_bars(self):
        """
        Update and emit the next bar.
        
        Returns:
            bool: True if more bars available, False otherwise
        """
        raise NotImplementedError
        
    def get_latest_bar(self, symbol):
        """
        Get the latest bar for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Bar: Latest bar or None
        """
        raise NotImplementedError
        
    def get_latest_bars(self, symbol, N=1):
        """
        Get the last N bars for a symbol.
        
        Args:
            symbol: Instrument symbol
            N: Number of bars
            
        Returns:
            list: List of bars
        """
        raise NotImplementedError
        
    def setup_train_test_split(self, method='ratio', train_ratio=0.7, split_date=None):
        """
        Set up train/test data split for optimization.
        
        Args:
            method: Split method ('ratio', 'date')
            train_ratio: Proportion for training data
            split_date: Date to split on
            
        Returns:
            bool: Success or failure
        """
        raise NotImplementedError
        
    def set_active_split(self, split_name):
        """
        Set the active data split.
        
        Args:
            split_name: Split name ('train', 'test', None for full dataset)
            
        Returns:
            bool: Success or failure
        """
        raise NotImplementedError
```

### 3. Historical Data Handler

The HistoricalDataHandler implements the DataHandler interface for backtesting:

```python
class HistoricalDataHandler(DataHandler):
    """
    Historical data handler implementation.
    
    Manages historical market data with proper train/test isolation.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize data containers
        self.data = {}  # Full dataset
        self.splits = {
            'train': {},
            'test': {}
        }
        self.bar_indices = {}
        self.active_split = None
        
        # Create CSV loader
        self.loader = CSVLoader()
        
        # Create train/test splitter
        self.splitter = TrainTestSplitter()
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Auto-load data if configured
        data_config = self.parameters.get('data', {})
        if 'symbols' in data_config and data_config.get('auto_load', False):
            self.load_data(data_config['symbols'])
            
    def load_data(self, symbols):
        """Load data for specified symbols."""
        self.symbols = symbols
        self.data = {}
        self.bar_indices = {symbol: 0 for symbol in symbols}
        
        # Get data directory
        data_dir = self.parameters.get('data_dir', 'data')
        
        # Load data for each symbol
        success = True
        for symbol in symbols:
            try:
                file_path = os.path.join(data_dir, f"{symbol}.csv")
                df = self.loader.load_file(file_path, symbol)
                self.data[symbol] = df
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
                success = False
                
        # Reset state
        self.current_bar_index = 0
        
        # Create default train/test split
        if success:
            split_config = self.parameters.get('train_test_split', {})
            self.setup_train_test_split(
                method=split_config.get('method', 'ratio'),
                train_ratio=split_config.get('train_ratio', 0.7),
                split_date=split_config.get('split_date')
            )
            
        return success
        
    def setup_train_test_split(self, method='ratio', train_ratio=0.7, split_date=None):
        """Set up train/test data split."""
        self.splits = {
            'train': {},
            'test': {}
        }
        
        # Split each symbol's data
        for symbol, df in self.data.items():
            train_df, test_df = self.splitter.split(
                df, method, train_ratio, split_date
            )
            
            self.splits['train'][symbol] = train_df
            self.splits['test'][symbol] = test_df
            
        return True
        
    def set_active_split(self, split_name):
        """Set active data split."""
        if split_name not in ['train', 'test', None]:
            raise ValueError(f"Invalid split name: {split_name}")
            
        self.active_split = split_name
        self.current_bar_index = 0
        self.bar_indices = {symbol: 0 for symbol in self.symbols}
        
        return True
        
    def update_bars(self):
        """Update and emit the next bar."""
        # Get active dataset
        dataset = self._get_active_dataset()
        
        # Check if we have data
        if not dataset:
            return False
            
        # Find earliest timestamp
        next_bars = []
        for symbol in self.symbols:
            if symbol not in dataset:
                continue
                
            df = dataset[symbol]
            idx = self.bar_indices[symbol]
            
            if idx >= len(df):
                continue
                
            timestamp = df.index[idx]
            next_bars.append((timestamp, symbol))
            
        if not next_bars:
            return False
            
        # Get earliest timestamp
        next_bars.sort()
        timestamp, symbol = next_bars[0]
        
        # Get data and create bar
        df = dataset[symbol]
        idx = self.bar_indices[symbol]
        
        bar_data = df.iloc[idx].to_dict()
        bar_data['timestamp'] = timestamp
        bar_data['symbol'] = symbol
        
        bar = Bar.from_dict(bar_data)
        
        # Emit bar event
        self.event_bus.publish(Event(EventType.BAR, bar.to_dict()))
        
        # Update indices
        self.bar_indices[symbol] += 1
        self.current_bar_index += 1
        
        # Check if more bars available
        return any(self.bar_indices[s] < len(dataset[s]) for s in self.symbols if s in dataset)
        
    def get_latest_bar(self, symbol):
        """Get the latest bar for a symbol."""
        dataset = self._get_active_dataset()
        
        if not dataset or symbol not in dataset:
            return None
            
        idx = self.bar_indices.get(symbol, 0) - 1
        if idx < 0:
            return None
            
        df = dataset[symbol]
        if idx >= len(df):
            return None
            
        bar_data = df.iloc[idx].to_dict()
        bar_data['timestamp'] = df.index[idx]
        bar_data['symbol'] = symbol
        
        return Bar.from_dict(bar_data)
        
    def get_latest_bars(self, symbol, N=1):
        """Get the last N bars for a symbol."""
        dataset = self._get_active_dataset()
        
        if not dataset or symbol not in dataset:
            return []
            
        end_idx = self.bar_indices.get(symbol, 0)
        start_idx = max(0, end_idx - N)
        
        if start_idx >= end_idx:
            return []
            
        df = dataset[symbol]
        end_idx = min(end_idx, len(df))
        
        bars = []
        for i in range(start_idx, end_idx):
            bar_data = df.iloc[i].to_dict()
            bar_data['timestamp'] = df.index[i]
            bar_data['symbol'] = symbol
            bars.append(Bar.from_dict(bar_data))
            
        return bars
        
    def _get_active_dataset(self):
        """Get the active dataset based on split selection."""
        if self.active_split is None:
            return self.data
        elif self.active_split in self.splits:
            return self.splits[self.active_split]
        else:
            return None
```

Key features:
- Loads and manages time series data from CSV files
- Maintains separate training and testing datasets
- Emits BAR events through the event bus
- Provides access to current and historical data
- Supports multiple symbols with proper synchronization
- Ensures proper isolation between training and testing datasets

### 4. Time Series Splitter

The TimeSeriesSplitter handles the creation of training and testing datasets:

- Supports ratio-based splitting (e.g., 70% train, 30% test)
- Supports date-based splitting (e.g., pre/post specific date)
- Ensures proper isolation between training and testing data
- Verifies no data leakage between splits

### 5. Data Loaders

- **CSV Loader**: Loads and normalizes data from CSV files
- Handles various CSV formats and column mappings
- Standardizes column names and data types
- Adds missing data and performs basic validation

## Data Isolation Mechanisms

The module supports several approaches to data isolation between training and testing datasets to balance memory efficiency with strict separation:

### 1. Deep Copy Isolation

- Creates complete copies of data for train/test splits
- Ensures complete isolation but uses more memory
- Appropriate for small to medium datasets

```python
# Example of deep copy isolation
train_df = df.iloc[:split_idx].copy(deep=True)
test_df = df.iloc[split_idx:].copy(deep=True)

# Verifying isolation
def _verify_isolation(train_df, test_df):
    """Verify complete isolation between datasets."""
    # Check for overlapping indices
    overlap = set(train_df.index) & set(test_df.index)
    if overlap:
        raise ValueError(f"Train and test datasets have overlapping indices: {overlap}")
        
    # Check that dataframes are separate objects
    if id(train_df) == id(test_df):
        raise ValueError("Train and test dataframes are the same object")
        
    # Check that underlying data is not shared
    if id(train_df.values) == id(test_df.values):
        raise ValueError("Train and test dataframes share memory")
```

### 2. View-Based Isolation

- Creates read-only views of the original data
- More memory efficient as it doesn't duplicate data
- Enforces isolation through controlled access

```python
class DataView:
    """Read-only view of market data with controlled access."""
    
    def __init__(self, data, start_idx, end_idx):
        self._data = data
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._current_idx = start_idx
        
    def get_current(self):
        """Get current data point."""
        if self._current_idx >= self._end_idx:
            return None
        return self._data.iloc[self._current_idx]
        
    def advance(self):
        """Advance to next data point."""
        if self._current_idx >= self._end_idx - 1:
            return False
        self._current_idx += 1
        return True
        
    def get_window(self, window_size):
        """Get window of data up to current point."""
        start = max(self._start_idx, self._current_idx - window_size + 1)
        return self._data.iloc[start:self._current_idx + 1]
        
    def reset(self):
        """Reset view to start."""
        self._current_idx = self._start_idx
        
    @property
    def size(self):
        """Get size of view."""
        return self._end_idx - self._start_idx
        
    @property
    def current_index(self):
        """Get current index."""
        return self._current_idx - self._start_idx
```

### 3. Copy-On-Write Isolation

- Shares data until modifications are needed
- Creates copies only for modified data
- Balances memory efficiency and flexibility

```python
class CopyOnWriteDataFrame:
    """DataFrame with copy-on-write semantics."""
    
    def __init__(self, data):
        self._base_data = data
        self._modified = False
        self._modified_data = None
        self._modified_rows = set()
        self._modified_columns = set()
        
    @property
    def data(self):
        """Get data, creating copy if modified."""
        if not self._modified:
            return self._base_data
        else:
            if self._modified_data is None:
                self._modified_data = self._base_data.copy()
            return self._modified_data
            
    def get_row(self, idx):
        """Get row by index."""
        if not self._modified or idx not in self._modified_rows:
            return self._base_data.iloc[idx]
        else:
            return self._modified_data.iloc[idx]
            
    def set_row(self, idx, value):
        """Set row by index."""
        # Mark as modified
        self._modified = True
        self._modified_rows.add(idx)
        
        # Create modified data if necessary
        if self._modified_data is None:
            self._modified_data = self._base_data.copy()
            
        # Update row
        self._modified_data.iloc[idx] = value
            
    def memory_usage(self):
        """Get memory usage information."""
        base_memory = self._base_data.memory_usage(deep=True).sum()
        modified_memory = 0 if self._modified_data is None else self._modified_data.memory_usage(deep=True).sum()
        
        return {
            'base_memory_bytes': base_memory,
            'modified_memory_bytes': modified_memory,
            'total_memory_bytes': base_memory + modified_memory,
            'modified_rows': len(self._modified_rows),
            'modified_columns': len(self._modified_columns),
            'is_modified': self._modified
        }
```

### 4. Shared Memory Isolation

For large datasets, shared memory can be used to avoid duplication:

```python
import multiprocessing as mp
import numpy as np
import pandas as pd

class SharedMemoryDataFrame:
    """
    DataFrame stored in shared memory.
    
    This class enables multiple processes to access the same data
    without copying, while maintaining isolation through views.
    """
    
    def __init__(self, data=None, shared_memory_name=None):
        """
        Initialize shared memory DataFrame.
        
        Args:
            data: DataFrame to store (optional)
            shared_memory_name: Name of existing shared memory block (optional)
        """
        self._shape = None
        self._columns = None
        self._index_name = None
        self._dtype = None
        self._shared_data = None
        self._shared_index = None
        self._shared_memory_name = shared_memory_name
        
        if data is not None:
            self._create_from_dataframe(data)
        elif shared_memory_name is not None:
            self._attach_to_existing(shared_memory_name)
        else:
            raise ValueError("Either data or shared_memory_name must be provided")
            
    def _create_from_dataframe(self, data):
        """Create shared memory from DataFrame."""
        # Save metadata
        self._shape = data.shape
        self._columns = list(data.columns)
        self._index_name = data.index.name
        self._dtype = data.dtypes.to_dict()
        
        # Create shared memory for data
        data_size = data.values.nbytes
        self._shared_data = mp.shared_memory.SharedMemory(
            create=True, size=data_size
        )
        
        # Copy data to shared memory
        data_array = np.ndarray(
            shape=data.shape,
            dtype=data.values.dtype,
            buffer=self._shared_data.buf
        )
        data_array[:] = data.values[:]
        
        # Create shared memory for index
        index_values = data.index.values
        index_size = index_values.nbytes
        self._shared_index = mp.shared_memory.SharedMemory(
            create=True, size=index_size
        )
        
        # Copy index to shared memory
        index_array = np.ndarray(
            shape=index_values.shape,
            dtype=index_values.dtype,
            buffer=self._shared_index.buf
        )
        index_array[:] = index_values[:]
        
        # Generate shared memory name
        self._shared_memory_name = f"dataframe_{id(self)}"
            
    def get_view(self, start_idx=0, end_idx=None):
        """Get DataFrame view."""
        end_idx = end_idx or self._shape[0]
        
        # Create numpy array from shared memory
        data_array = np.ndarray(
            shape=self._shape,
            dtype=np.float64,  # Assuming float data
            buffer=self._shared_data.buf
        )
        
        # Create index array from shared memory
        index_array = np.ndarray(
            shape=(self._shape[0],),
            dtype=np.int64,  # Assuming integer index
            buffer=self._shared_index.buf
        )
        
        # Create DataFrame with view of data
        df = pd.DataFrame(
            data=data_array[start_idx:end_idx],
            index=pd.Index(index_array[start_idx:end_idx], name=self._index_name),
            columns=self._columns
        )
        
        return df
        
    def release(self):
        """Release shared memory."""
        if self._shared_data is not None:
            self._shared_data.close()
            self._shared_data.unlink()
            self._shared_data = None
            
        if self._shared_index is not None:
            self._shared_index.close()
            self._shared_index.unlink()
            self._shared_index = None
```

### 5. Isolation Factory

To automatically select the best isolation method based on dataset characteristics:

```python
class DataIsolationFactory:
    """
    Factory for creating appropriate data isolation.
    
    This class selects and creates the most appropriate data isolation
    mechanism based on dataset characteristics and system constraints.
    """
    
    @staticmethod
    def create_isolation(data, mode=None, memory_threshold_mb=1000):
        """
        Create data isolation.
        
        Args:
            data: DataFrame to create isolation for
            mode: Isolation mode (or None for auto-selection)
            memory_threshold_mb: Memory threshold for auto-selection
            
        Returns:
            Dict with isolation information and data
        """
        # Estimate memory usage
        memory_usage = MemoryTracker.estimate_dataframe_memory(data)
        
        # Auto-select mode if not specified
        if mode is None:
            if memory_usage['total_mb'] < 10:
                # Small dataset - use deep copy for simplicity
                mode = IsolationMode.DEEP_COPY
            elif memory_usage['total_mb'] < memory_threshold_mb:
                # Medium dataset - use copy-on-write
                mode = IsolationMode.COPY_ON_WRITE
            else:
                # Large dataset - use view-based approach
                mode = IsolationMode.VIEW_BASED
                
        # Create isolation based on mode
        result = {
            'mode': mode,
            'original_data_info': {
                'shape': data.shape,
                'memory_mb': memory_usage['total_mb']
            }
        }
        
        if mode == IsolationMode.DEEP_COPY:
            result['data'] = data.copy(deep=True)
        elif mode == IsolationMode.VIEW_BASED:
            result['data'] = DataView(data, 0, len(data))
        elif mode == IsolationMode.COPY_ON_WRITE:
            result['data'] = CopyOnWriteDataFrame(data)
        elif mode == IsolationMode.SHARED_MEMORY:
            shared_df = SharedMemoryDataFrame(data)
            result['data'] = shared_df
            result['shared_memory_name'] = shared_df._shared_memory_name
            
        return result
```

## Memory Management Best Practices

Memory management is critical when dealing with large datasets:

### 1. Memory Tracking and Monitoring

```python
import gc
import psutil
import os
from typing import Dict, Any

class MemoryTracker:
    """
    Memory usage tracking and management.
    
    This class provides tools for tracking and managing memory usage
    during data operations in the ADMF-Trader system.
    """
    
    @staticmethod
    def get_process_memory():
        """
        Get current process memory usage.
        
        Returns:
            Dict with memory usage information
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_bytes': memory_info.rss,  # Resident Set Size
            'vms_bytes': memory_info.vms,  # Virtual Memory Size
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024)
        }
        
    @staticmethod
    def get_system_memory():
        """
        Get system memory usage.
        
        Returns:
            Dict with memory usage information
        """
        memory = psutil.virtual_memory()
        
        return {
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'percent_used': memory.percent
        }
        
    @staticmethod
    def collect_garbage():
        """
        Collect garbage.
        
        Returns:
            Dict with collection counts
        """
        return {
            'collected': gc.collect()
        }
        
    @staticmethod
    def estimate_dataframe_memory(df):
        """
        Estimate memory usage of DataFrame.
        
        Args:
            df: DataFrame to estimate memory for
            
        Returns:
            Dict with memory usage information
        """
        memory_usage = df.memory_usage(deep=True)
        total_bytes = memory_usage.sum()
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'columns': {col: memory_usage[col] for col in df.columns}
        }
```

### 2. Specialized Time Series Data Structure

```python
import numpy as np
from typing import Dict, Optional

class TimeSeriesArray:
    """
    Memory-efficient time series array.
    
    This class stores time series data in a compact format with
    timestamps and values separated for memory efficiency.
    """
    
    def __init__(self, timestamps=None, values=None):
        """
        Initialize time series array.
        
        Args:
            timestamps: Array of timestamps
            values: Array of values
        """
        self._timestamps = timestamps
        self._values = values
        
    @classmethod
    def from_dataframe(cls, df, value_column):
        """
        Create from DataFrame.
        
        Args:
            df: DataFrame with time series data
            value_column: Column containing values
            
        Returns:
            TimeSeriesArray instance
        """
        timestamps = np.array(df.index.astype(np.int64))
        values = np.array(df[value_column].values)
        return cls(timestamps, values)
        
    def get_view(self, start_idx=0, end_idx=None):
        """
        Get view of data.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            TimeSeriesArray with view of data
        """
        end_idx = end_idx if end_idx is not None else len(self._timestamps)
        return TimeSeriesArray(
            self._timestamps[start_idx:end_idx],
            self._values[start_idx:end_idx]
        )
        
    def to_dataframe(self):
        """
        Convert to DataFrame.
        
        Returns:
            DataFrame with time series data
        """
        index = pd.to_datetime(self._timestamps)
        return pd.DataFrame({'value': self._values}, index=index)
        
    def __len__(self):
        """Get length of time series."""
        return len(self._timestamps) if self._timestamps is not None else 0
        
    @property
    def values(self):
        """Get values array."""
        return self._values
        
    @property
    def timestamps(self):
        """Get timestamps array."""
        return self._timestamps
        
    def memory_usage(self):
        """
        Get memory usage information.
        
        Returns:
            Dict with memory usage information
        """
        timestamps_bytes = 0 if self._timestamps is None else self._timestamps.nbytes
        values_bytes = 0 if self._values is None else self._values.nbytes
        
        return {
            'timestamps_bytes': timestamps_bytes,
            'values_bytes': values_bytes,
            'total_bytes': timestamps_bytes + values_bytes
        }
```

### 3. Memory Optimization Techniques

```python
# Optimize numeric columns
for col in df.select_dtypes(include=['int']).columns:
    if df[col].min() >= 0:
        if df[col].max() < 256:
            df[col] = df[col].astype(np.uint8)
        elif df[col].max() < 65536:
            df[col] = df[col].astype(np.uint16)
    else:
        if df[col].min() > -128 and df[col].max() < 128:
            df[col] = df[col].astype(np.int8)
            
# Optimize float columns
for col in df.select_dtypes(include=['float']).columns:
    df[col] = df[col].astype(np.float32)
    
# Optimize object columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() < len(df) * 0.5:
        df[col] = df[col].astype('category')
```

### 4. When to Use Different Isolation Approaches

- **Deep Copying**: Use for small datasets or when complete isolation is critical
- **Data Views**: Use for read-only access to large datasets
- **Copy-On-Write**: Use for medium-sized datasets with occasional modifications
- **Shared Memory**: Use for very large datasets accessed by multiple processes

### 5. Pruning Strategies for Historical Data

```python
# Fixed-size window pruning
def _prune_history(self):
    """Prune historical data to conserve memory."""
    max_history = self.parameters.get('max_history', 10000)
    for symbol in self.symbols:
        if symbol in self.bars and len(self.bars[symbol]) > max_history:
            # Keep only the most recent bars
            self.bars[symbol] = self.bars[symbol][-max_history:]
```

```python
# Time-based pruning
def _prune_history_by_time(self):
    """Prune data older than the specified timeframe."""
    max_age = pd.Timedelta(self.parameters.get('max_history_age', '30D'))
    now = pd.Timestamp.now()
    
    for symbol in self.symbols:
        if symbol in self.bars:
            # Filter by timestamp
            self.bars[symbol] = [
                bar for bar in self.bars[symbol]
                if (now - bar.timestamp) <= max_age
            ]
```

## Implementation Structure

The Data module is organized into the following structure:

```
src/data/
├── __init__.py
├── interfaces/
│   ├── __init__.py
│   ├── data_handler.py        # Abstract data handler
│   └── data_source.py         # Data source interface
├── handlers/
│   ├── __init__.py
│   ├── historical_data_handler.py  # Historical data implementation
│   └── data_view.py           # Memory-efficient data view
├── loaders/
│   ├── __init__.py
│   └── csv_loader.py          # CSV data loading utility
├── models/
│   ├── __init__.py
│   └── bar.py                 # Bar data model
├── splitters/
│   ├── __init__.py
│   └── train_test_splitter.py # Train/test isolation
└── utils/
    ├── __init__.py
    ├── timeframe.py           # Timeframe utilities
    └── data_validator.py      # Data validation
```

## Key Considerations for Data Usage

### 1. Train/Test Isolation

Proper train/test isolation is critical for optimization:

1. **Isolation Verification**: Always verify dataset separation
   ```python
   overlap = set(train_df.index) & set(test_df.index)
   if overlap:
       raise ValueError(f"Train and test datasets have overlapping indices")
   ```

2. **Context Switching**: Reset all state when switching between train/test
   ```python
   def set_active_split(self, split_name):
       self.active_split = split_name
       self.current_bar_index = 0
       self.bar_indices = {symbol: 0 for symbol in self.symbols}
   ```

### 2. Multi-Symbol Synchronization

For multi-symbol backtesting:

1. **Timeline Construction**: Pre-compute a global timeline
   ```python
   def _build_timeline(self):
       self.timeline = []
       for symbol, df in self.data.items():
           for timestamp in df.index:
               self.timeline.append((timestamp, symbol))
       self.timeline.sort()
   ```

2. **Efficient Lookups**: Use optimized data structures like priority queues
   ```python
   # Use a priority queue for timestamp ordering
   import heapq
   
   self.timestamp_queue = []
   for symbol in self.symbols:
       if self.bar_indices[symbol] < len(self.data[symbol]):
           timestamp = self.data[symbol].index[self.bar_indices[symbol]]
           heapq.heappush(self.timestamp_queue, (timestamp, symbol))
   ```

### 3. Data Quality

Implement these data quality checks:

1. **Continuity Validation**: Check for gaps in data
   ```python
   def _validate_continuity(self, df, freq='1D'):
       expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
       missing = expected_index.difference(df.index)
       return len(missing) == 0, missing
   ```

2. **Outlier Detection**: Identify and handle outliers
   ```python
   def _detect_outliers(self, df, column='close', threshold=3):
       mean = df[column].mean()
       std = df[column].std()
       return df[(df[column] - mean).abs() > threshold * std]
   ```

3. **Data Integrity**: Verify OHLC relationships
   ```python
   def _validate_ohlc(self, df):
       valid = (df['high'] >= df['open']) & (df['high'] >= df['close']) & \
               (df['low'] <= df['open']) & (df['low'] <= df['close'])
       return valid.all(), df[~valid]
   ```

## Usage Patterns

### Basic Data Loading

```python
# Initialize data handler
data_handler = HistoricalDataHandler("data_handler", {
    "data_dir": "data",
    "train_test_split": {
        "method": "ratio",
        "train_ratio": 0.7
    }
})

# Load data
data_handler.load_data(["SPY", "AAPL", "MSFT"])

# Set up train/test split
data_handler.setup_train_test_split(method="ratio", train_ratio=0.7)
```

### Emitting Bar Events

```python
# Set active split
data_handler.set_active_split("train")

# Process all bars
while data_handler.update_bars():
    pass  # Events are emitted to the event bus
```

### Accessing Data

```python
# Get latest bar
bar = data_handler.get_latest_bar("SPY")

# Get last 10 bars
bars = data_handler.get_latest_bars("SPY", 10)

# Access bar data
print(f"Latest close: {bar.close}")
```

## Conclusion

The Data module provides robust and efficient market data handling for the ADMF-Trader system. By combining proper train/test isolation, memory-efficient data structures, and comprehensive data quality checks, it enables reliable strategy development and backtesting while handling large datasets effectively.