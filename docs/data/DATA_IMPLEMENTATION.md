# Data Module Implementation Guide

## Overview

The Data module is responsible for loading, processing, and providing market data to the ADMF-Trader system. It manages data isolation between training and testing periods, ensures proper data propagation through the event system, and handles the efficient processing of market data.

## Key Components

1. **Data Models**
   - Bar class for OHLCV data representation
   - Timeframe utilities
   - Standardized data formats

2. **Data Handler Interface**
   - Abstract interface for data management
   - Event-driven data publishing
   - Historical and streaming implementations

3. **Train/Test Splitting**
   - Clean data separation for optimization
   - Memory isolation verification
   - Different splitting methods (ratio, date)

4. **Data Loaders**
   - CSV data loading with flexible formats
   - API and other data source integrations
   - Symbol management

5. **Data Transformation**
   - Resampling to different timeframes
   - Data normalization and cleaning
   - Missing data handling

## Implementation Structure

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

## Component Specifications

### 1. Data Handler Interface

The DataHandler abstract class defines the interface for all data handling components:

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

### 2. Bar Data Model

The Bar class represents OHLCV market data:

```python
class Bar:
    """
    OHLCV bar data model.
    
    Represents a single bar of market data for a specific symbol and timeframe.
    """
    
    def __init__(self, timestamp, symbol, open_price, high, low, close, volume, timeframe="1D"):
        """
        Initialize a bar with OHLCV data.
        
        Args:
            timestamp: Bar timestamp
            symbol: Instrument symbol
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Volume
            timeframe: Bar timeframe
        """
        self.timestamp = self._parse_timestamp(timestamp)
        self.symbol = symbol
        self.open = float(open_price)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)
        self.timeframe = timeframe
        
    def _parse_timestamp(self, timestamp):
        """Convert timestamp to datetime if necessary."""
        if isinstance(timestamp, str):
            return pd.to_datetime(timestamp)
        return timestamp
        
    def to_dict(self):
        """Convert bar to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create bar from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            symbol=data['symbol'],
            open_price=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            timeframe=data.get('timeframe', '1D')
        )
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

### 4. Train/Test Splitter

The TrainTestSplitter ensures complete data isolation for optimization:

```python
class TrainTestSplitter:
    """
    Train/test data splitter with isolation verification.
    
    Ensures proper isolation between training and testing data to
    prevent look-ahead bias during optimization.
    """
    
    def split(self, df, method='ratio', train_ratio=0.7, split_date=None):
        """
        Split dataframe into training and testing sets.
        
        Args:
            df: DataFrame to split
            method: Split method ('ratio', 'date')
            train_ratio: Proportion for training data
            split_date: Date to split on
            
        Returns:
            tuple: (train_df, test_df)
        """
        # Choose split method
        if method == 'ratio':
            train_df, test_df = self._split_by_ratio(df, train_ratio)
        elif method == 'date':
            train_df, test_df = self._split_by_date(df, split_date)
        else:
            raise ValueError(f"Unsupported split method: {method}")
            
        # Verify isolation
        self._verify_isolation(train_df, test_df)
        
        return train_df, test_df
        
    def _split_by_ratio(self, df, ratio):
        """Split by proportion of data points."""
        split_idx = int(len(df) * ratio)
        
        # Create deep copies to ensure isolation
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
        
    def _split_by_date(self, df, date):
        """Split by specific date."""
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex for date splitting")
            
        # Create deep copies to ensure isolation
        train_df = df.loc[:date].copy()
        test_df = df.loc[date:].copy()
        
        # Handle edge case where split date is in the index
        if date in test_df.index and date in train_df.index:
            test_df = test_df.iloc[1:].copy()
            
        return train_df, test_df
        
    def _verify_isolation(self, train_df, test_df):
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

### 5. CSV Loader

The CSVLoader handles loading data from CSV files:

```python
class CSVLoader:
    """
    CSV data loader with format normalization.
    
    Loads data from CSV files and standardizes formats.
    """
    
    def __init__(self):
        """Initialize CSV loader."""
        # Default column mapping
        self.column_map = {
            'date': 'timestamp',
            'time': 'timestamp',
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
    def load_file(self, file_path, symbol=None):
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            symbol: Symbol to assign (default: extract from filename)
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Extract symbol from filename if not provided
        if symbol is None:
            symbol = os.path.basename(file_path).split('.')[0]
            
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
            
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp').sort_index()
            
        return df
        
    def _standardize_columns(self, df):
        """Standardize column names using mapping."""
        # Copy dataframe to avoid modifying original
        df = df.copy()
        
        # Use column mapping to standardize names
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in self.column_map:
                df = df.rename(columns={col: self.column_map[col_lower]})
                
        return df
```

### 6. DataView Pattern

The DataView provides memory-efficient data access:

```python
class DataView:
    """
    Memory-efficient view of market data.
    
    Provides a view into a dataset without copying the underlying data.
    """
    
    def __init__(self, dataset, start_idx=None, end_idx=None):
        """
        Initialize data view.
        
        Args:
            dataset: Underlying dataset
            start_idx: Start index (default: 0)
            end_idx: End index (default: len(dataset))
        """
        self.dataset = dataset
        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx if end_idx is not None else len(dataset)
        
    def __getitem__(self, idx):
        """Get item at specified index."""
        if isinstance(idx, slice):
            # Handle slice
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step
            
            # Adjust indices
            adjusted_start = self.start_idx + start
            adjusted_stop = min(self.start_idx + stop, self.end_idx)
            
            # Return new view
            return DataView(self.dataset, adjusted_start, adjusted_stop)
        else:
            # Handle index
            if idx < 0:
                idx = len(self) + idx
                
            if idx < 0 or idx >= len(self):
                raise IndexError("Index out of range")
                
            return self.dataset[self.start_idx + idx]
            
    def __len__(self):
        """Get length of view."""
        return self.end_idx - self.start_idx
```

## Best Practices

### Train/Test Isolation

Proper train/test isolation is critical for optimization:

1. **Deep Copying**: Always create deep copies of data for train/test splits
   ```python
   train_df = df.iloc[:split_idx].copy()
   test_df = df.iloc[split_idx:].copy()
   ```

2. **Isolation Verification**: Verify that datasets are completely isolated
   ```python
   overlap = set(train_df.index) & set(test_df.index)
   if overlap:
       raise ValueError(f"Train and test datasets have overlapping indices")
   ```

3. **Memory Tracking**: Check that underlying memory is not shared
   ```python
   if id(train_df.values) == id(test_df.values):
       raise ValueError("Train and test dataframes share memory")
   ```

4. **Context Switching**: Properly reset all state when switching between train/test
   ```python
   def set_active_split(self, split_name):
       self.active_split = split_name
       self.current_bar_index = 0
       self.bar_indices = {symbol: 0 for symbol in self.symbols}
   ```

### Data Access Pattern

Follow this pattern for data access:

```python
# Get active dataset
dataset = self._get_active_dataset()

# Check for valid data
if not dataset or symbol not in dataset:
    return None
    
# Get data at specified index
idx = self.bar_indices.get(symbol, 0) - 1
if idx < 0 or idx >= len(dataset[symbol]):
    return None
    
# Create and return bar object
bar_data = dataset[symbol].iloc[idx].to_dict()
bar_data['timestamp'] = dataset[symbol].index[idx]
bar_data['symbol'] = symbol
return Bar.from_dict(bar_data)
```

### Event Emission

Emit events using this pattern:

```python
def update_bars(self):
    # Get next bar
    bar = self._get_next_bar()
    if bar is None:
        return False
        
    # Emit event
    self.event_bus.publish(Event(EventType.BAR, bar.to_dict()))
    
    # Update indices
    self.bar_indices[bar.symbol] += 1
    self.current_bar_index += 1
    
    return True
```

## Memory Management Best Practices

Memory management is critical when dealing with large datasets. The following best practices should be followed to ensure efficient memory usage:

### 1. When to Use Deep Copying vs. Views

**Deep Copying**
- Use when you need to ensure complete data isolation (especially for train/test splits)
- Use when modifying data that shouldn't affect the original source
- Required for proper optimization to prevent data leakage

```python
# Create completely isolated copy for optimization
train_df = full_df.iloc[:split_idx].copy(deep=True)
```

**Data Views**
- Use when you need read-only access to a subset of data
- Use for temporary operations that don't require persistence
- Better performance for large datasets when isolation isn't critical

```python
# Use DataView pattern for efficient windowing
class DataWindow:
    def __init__(self, data, start, end):
        self.data = data
        self.start = start
        self.end = end
        
    def __getitem__(self, idx):
        return self.data[self.start + idx]
        
    def __len__(self):
        return self.end - self.start
```

### 2. Memory Profiling and Monitoring

The system includes built-in memory monitoring to detect issues:

```python
def _track_memory_usage(self):
    """Log memory usage for debugging."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    self.logger.debug(f"Memory usage: {memory_mb:.2f} MB")
    
    # Alert on excessive memory usage
    if memory_mb > self.parameters.get('memory_threshold_mb', 1000):
        self.logger.warning(f"High memory usage detected: {memory_mb:.2f} MB")
```

### 3. Pruning Strategies for Historical Data

Implement these strategies to keep memory usage bounded:

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

### 4. Efficient Data Structures

Choose the right data structure for the use case:

**Pandas DataFrames**
- Use for bulk data operations and vectorized calculations
- Efficient for large datasets with columnar operations
- Good for data with timestamp indices

**NumPy Arrays**
- Use for numerical computations requiring high performance
- More memory efficient than DataFrames for simple numerical data
- Better for mathematical operations on fixed-size data

**Specialized Collections**
- Use ThreadSafeDict for shared state that requires thread safety
- Use collections.deque for fixed-size history with efficient append/pop
- Use SortedDict/SortedList for data that needs to remain ordered

```python
# Example of efficient rolling window calculation
def calculate_moving_average(self, symbol, window):
    """Calculate moving average using numpy for efficiency."""
    prices = np.array(self.bars[symbol]['close'][-window*2:])
    return np.convolve(prices, np.ones(window)/window, mode='valid')
```

## Implementation Considerations

### 2. Multi-Symbol Synchronization

For multi-symbol backtesting, consider:

1. **Timeline Construction**: Pre-compute a global timeline
   ```python
   def _build_timeline(self):
       self.timeline = []
       for symbol, df in self.data.items():
           for timestamp in df.index:
               self.timeline.append((timestamp, symbol))
       self.timeline.sort()
   ```

2. **Efficient Lookups**: Use optimized data structures
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

By following these guidelines, you'll create a robust Data module that provides reliable market data with proper train/test isolation for the ADMF-Trader system.