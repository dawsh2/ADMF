# Data Module Implementation Guide

## Overview

The Data module is responsible for loading, processing, and providing market data to the ADMF-Trader system. It manages data isolation between training and testing periods, ensures proper data propagation through the event system, and handles the efficient processing of market data.

## Directory Structure

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
    ├── data_validator.py      # Data validation
    └── memory_tracker.py      # Memory usage tracking
```

## Core Interfaces

### DataHandler Interface

The foundation of the Data module is the `DataHandler` abstract class, which defines the interface for all data handling components:

```python
class DataHandlerBase(Component):
    """
    Abstract interface for data management components.
    
    Responsible for loading market data, providing access to that data,
    and publishing data events to the system.
    """
    
    @abstractmethod
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols."""
        pass
        
    @abstractmethod
    def update_bars(self) -> bool:
        """Update and emit the next bar."""
        pass
        
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get the latest bar for a symbol."""
        pass
        
    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Bar]:
        """Get the last N bars for a symbol."""
        pass
        
    @abstractmethod
    def setup_train_test_split(self, method: str = 'ratio', 
                              train_ratio: float = 0.7, 
                              split_date: Optional[datetime] = None) -> bool:
        """Set up train/test data split for optimization."""
        pass
        
    @abstractmethod
    def set_active_split(self, split_name: Optional[str]) -> bool:
        """Set the active data split (train/test/None)."""
        pass
```

## Data Models

### Bar Model

The core data model is the `Bar` class representing OHLCV market data:

```python
class Bar:
    """
    OHLCV bar data model.
    
    Represents a single bar of market data for a specific symbol and timeframe.
    """
    
    def __init__(self, timestamp, symbol, open_price, high, low, close, volume, timeframe="1D"):
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

## Handlers

### HistoricalDataHandler

The `HistoricalDataHandler` implements the DataHandler interface for backtesting:

```python
class HistoricalDataHandler(DataHandler):
    """
    Historical data handler implementation.
    
    Manages historical market data with proper train/test isolation.
    """
    
    def __init__(self, name, parameters=None):
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
```

## Memory-Efficient Data Isolation

### Data View Class

The `DataView` class provides memory-efficient data access:

```python
class DataView:
    """
    Memory-efficient view of market data.
    
    Provides a view into a dataset without copying the underlying data.
    """
    
    def __init__(self, data, start_idx=None, end_idx=None):
        """Initialize data view."""
        self.data = data
        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx if end_idx is not None else len(data)
        self.current_idx = start_idx
        
    def get_current(self):
        """Get current data point."""
        if self.current_idx >= self.end_idx:
            return None
            
        return self.data.iloc[self.current_idx]
        
    def advance(self):
        """Advance to next data point."""
        if self.current_idx >= self.end_idx - 1:
            return False
            
        self.current_idx += 1
        return True
        
    def get_window(self, window_size):
        """Get window of data up to current point."""
        start = max(self.start_idx, self.current_idx - window_size + 1)
        return self.data.iloc[start:self.current_idx + 1]
        
    def reset(self):
        """Reset view to start."""
        self.current_idx = self.start_idx
        
    def __len__(self):
        """Get length of view."""
        return self.end_idx - self.start_idx
```

### Enhanced Train/Test Splitter

The `EnhancedTrainTestSplitter` provides memory-efficient data isolation:

```python
class EnhancedTrainTestSplitter:
    """
    Enhanced train/test data splitter with efficient isolation.
    
    Provides multiple isolation methods to balance memory efficiency
    and performance based on dataset characteristics.
    """
    
    def __init__(self, isolation_mode='view'):
        """Initialize with specified isolation mode."""
        self.isolation_mode = isolation_mode
        
    def split(self, df, method='ratio', train_ratio=0.7, split_date=None):
        """Split dataframe into training and testing sets."""
        # Calculate split point
        if method == 'ratio':
            split_idx = int(len(df) * train_ratio)
        elif method == 'date':
            if split_date is None:
                raise ValueError("split_date must be provided for date-based split")
            split_idx = df.index.get_loc(pd.to_datetime(split_date))
        else:
            raise ValueError(f"Unsupported split method: {method}")
            
        # Create train/test splits based on isolation mode
        if self.isolation_mode == 'deep_copy':
            # Traditional deep copy approach
            train_df = df.iloc[:split_idx].copy(deep=True)
            test_df = df.iloc[split_idx:].copy(deep=True)
        elif self.isolation_mode == 'view':
            # View-based isolation
            train_df = DataView(df, 0, split_idx)
            test_df = DataView(df, split_idx, len(df))
        else:
            raise ValueError(f"Unsupported isolation mode: {self.isolation_mode}")
            
        # Verify isolation
        self._verify_isolation(train_df, test_df)
        
        return train_df, test_df
    
    def _verify_isolation(self, train_df, test_df):
        """Verify complete isolation between datasets."""
        # Verification depends on isolation mode
        if self.isolation_mode == 'deep_copy':
            # For deep copy, verify memory isolation
            if id(train_df) == id(test_df):
                raise ValueError("Train and test dataframes are the same object")
                
            # Check for overlapping indices
            if hasattr(train_df, 'index') and hasattr(test_df, 'index'):
                overlap = set(train_df.index) & set(test_df.index)
                if overlap:
                    raise ValueError(f"Train and test datasets have overlapping indices: {overlap}")
        elif self.isolation_mode == 'view':
            # For views, verify non-overlapping ranges
            if train_df.end_idx > test_df.start_idx:
                raise ValueError("Train and test views have overlapping ranges")
```

## Train/Test Splitting

### Train/Test Data Isolation

The system provides several methods for data isolation between training and testing phases:

```python
class IsolationMode(Enum):
    """Data isolation modes."""
    DEEP_COPY = auto()  # Traditional deep copy approach
    VIEW_BASED = auto()  # View-based isolation
    COPY_ON_WRITE = auto()  # Copy-on-write isolation
    SHARED_MEMORY = auto()  # Shared memory isolation

class DataIsolationFactory:
    """Factory for creating appropriate data isolation."""
    
    @staticmethod
    def create_isolation(data, mode=None, memory_threshold_mb=1000):
        """Create data isolation based on dataset characteristics."""
        # Estimate memory usage
        memory_usage = MemoryTracker.estimate_dataframe_memory(data)
        
        # Auto-select mode if not specified
        if mode is None:
            if memory_usage['total_mb'] < 10:
                # Small dataset - use deep copy for simplicity
                mode = IsolationMode.DEEP_COPY
            elif memory_usage['total_mb'] < memory_threshold_mb:
                # Medium dataset - use view-based approach
                mode = IsolationMode.VIEW_BASED
            else:
                # Large dataset - use shared memory approach
                mode = IsolationMode.SHARED_MEMORY
                
        # Create and return appropriate isolation
        if mode == IsolationMode.DEEP_COPY:
            return data.copy(deep=True)
        elif mode == IsolationMode.VIEW_BASED:
            return DataView(data, 0, len(data))
        # Implement other isolation modes as needed
```

## Data Loaders

The Data module includes loaders for different data sources:

```python
class CSVLoader:
    """
    CSV data loader with format normalization.
    
    Loads data from CSV files and standardizes formats.
    """
    
    def __init__(self):
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
        """Load data from CSV file."""
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
            
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp').sort_index()
            
        return df
```

## Memory Management Best Practices

Memory management is critical when dealing with large datasets:

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

### 3. Data Type Optimization

Optimize data types to reduce memory usage:

```python
def optimize_dataframe_memory(df):
    """Optimize memory usage of DataFrame."""
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
            elif df[col].min() > -32768 and df[col].max() < 32768:
                df[col] = df[col].astype(np.int16)
                
    # Optimize float columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype(np.float32)
        
    return df
```

## Data Quality Practices

### 1. Data Integrity Checks

```python
def _validate_ohlc(self, df):
    """Verify OHLC relationships."""
    valid = (df['high'] >= df['open']) & (df['high'] >= df['close']) & \
            (df['low'] <= df['open']) & (df['low'] <= df['close'])
    return valid.all(), df[~valid]
```

### 2. Missing Data Detection

```python
def _validate_continuity(self, df, freq='1D'):
    """Check for gaps in data."""
    expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    missing = expected_index.difference(df.index)
    return len(missing) == 0, missing
```

### 3. Data Isolation Verification

```python
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

## Implementation Timeline

1. **Phase 1: Core Interfaces (Week 1)**
   - Create data handler interface
   - Implement bar data model
   - Define data source interface

2. **Phase 2: Basic Implementation (Week 2)**
   - Implement historical data handler
   - Create CSV data loader
   - Develop train/test splitter

3. **Phase 3: Memory Management (Week 3)**
   - Add memory-efficient data views
   - Implement efficient train/test isolation
   - Add memory monitoring

4. **Phase 4: Data Quality (Week 4)**
   - Implement data validation utilities
   - Add outlier detection
   - Create continuity validation

5. **Phase 5: Integration Testing (Week 5)**
   - Test with strategy module
   - Verify event propagation
   - Benchmark performance