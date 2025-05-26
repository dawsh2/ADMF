## Data Module Documentation

### Overview

The Data module in the ADMF-Trader system is tasked with loading, processing, and supplying market data. Its key functions include managing data isolation between training and testing phases, ensuring correct data propagation via the event system, and efficiently processing market data.

### Key Components

#### 1. Data Models

* **Bar Class**: Represents OHLCV (Open, High, Low, Close, Volume) market data using standardized fields and includes conversion methods.
* **Timeframe Utilities**: Provides tools for managing various time frames and resolutions.
* **Standardized Data Formats**: Ensures consistent market data representation throughout the system.

#### 2. Data Handler Interface

The `DataHandler` is the central abstraction for loading and managing market data. It's a `Component` subclass.

* **`__init__(self, name, parameters=None)`**: Initializes with a name and optional parameters. Sets up `symbols` list and `current_bar_index`.
* **`load_data(self, symbols)`**: Abstract method to load data for specified symbols.
* **`update_bars(self)`**: Abstract method to update and emit the next bar, returning `True` if more bars are available.
* **`get_latest_bar(self, symbol)`**: Abstract method to retrieve the latest bar for a given symbol.
* **`get_latest_bars(self, symbol, N=1)`**: Abstract method to get the last N bars for a symbol.
* **`setup_train_test_split(self, method='ratio', train_ratio=0.7, split_date=None)`**: Abstract method to configure the train/test data split, supporting methods like 'ratio' or 'date'.
* **`set_active_split(self, split_name)`**: Abstract method to set the active data split (e.g., 'train', 'test', or `None` for the full dataset).

#### 3. Historical Data Handler

The `HistoricalDataHandler` implements the `DataHandler` interface for backtesting purposes.

* **`__init__(self, name, parameters=None)`**: Initializes data containers (`data`, `splits` for 'train' and 'test', `bar_indices`), sets `active_split` to `None`, and creates `CSVLoader` and `TrainTestSplitter` instances.
* **`initialize(self, context)`**: Initializes dependencies and can auto-load data if configured in parameters.
* **`load_data(self, symbols)`**: Loads data for specified symbols from CSV files located in `data_dir` (default 'data'). It populates `self.data` and initializes `bar_indices`. If successful, it sets up a default train/test split based on parameters.
* **`setup_train_test_split(self, method='ratio', train_ratio=0.7, split_date=None)`**: Configures train/test data splits for each symbol using the `TrainTestSplitter`.
* **`set_active_split(self, split_name)`**: Sets the currently active data split ('train', 'test', or `None`). Resets `current_bar_index` and `bar_indices`.
* **`update_bars(self)`**: Retrieves the next bar from the active dataset based on the earliest timestamp across all symbols. It creates a `Bar` object from the data and publishes a `BAR` event via the event bus. It then updates the bar indices and returns `True` if more data is available for any symbol.
* **`get_latest_bar(self, symbol)`**: Returns the latest `Bar` object for the specified symbol from the active dataset.
* **`get_latest_bars(self, symbol, N=1)`**: Returns a list of the last N `Bar` objects for the specified symbol from the active dataset.
* **`_get_active_dataset(self)`**: Internal method to retrieve the current dataset (full, train, or test) based on `self.active_split`.

Key features of `HistoricalDataHandler` include loading from CSVs, maintaining train/test separation, emitting BAR events, providing data access, supporting multiple symbols with synchronization, and ensuring split isolation.

#### 4. Time Series Splitter

The `TimeSeriesSplitter` is responsible for creating training and testing datasets. It supports:

* Ratio-based splitting (e.g., 70% train, 30% test).
* Date-based splitting.
* Ensuring proper isolation and verifying no data leakage between splits.

#### 5. Data Loaders

* **CSV Loader**: Loads and normalizes data from CSV files. It handles various formats, standardizes columns and data types, manages missing data, and performs basic validation.

### Data Isolation Mechanisms

The module offers several data isolation strategies for train/test datasets to balance memory use and separation:

1.  **Deep Copy Isolation**: Creates full copies of data for splits. Ensures complete isolation but uses more memory. Suitable for small to medium datasets. Includes verification for overlapping indices and shared memory.
2.  **View-Based Isolation**: Creates read-only views of the original data using a `DataView` class. This is more memory-efficient as it avoids data duplication and enforces isolation via controlled access. `DataView` provides methods to get current data, advance, get a window of data, and reset.
3.  **Copy-On-Write Isolation**: Shares data until modification is needed, creating copies only for modified parts using a `CopyOnWriteDataFrame` class. This balances memory efficiency and flexibility. The class tracks modifications and provides memory usage information.
4.  **Shared Memory Isolation**: For large datasets, uses `multiprocessing.shared_memory` to allow multiple processes to access the same data without copying, with isolation maintained through views. The `SharedMemoryDataFrame` class handles creating shared memory from a DataFrame or attaching to existing shared memory, providing views, and releasing memory.
5.  **Isolation Factory**: The `DataIsolationFactory` automatically selects the best isolation method (Deep Copy, View-Based, Copy-On-Write, Shared Memory) based on dataset size and memory thresholds using `IsolationMode` enum.

### Memory Management Best Practices

Effective memory management is crucial for large datasets:

1.  **Memory Tracking and Monitoring**: The `MemoryTracker` class provides static methods to get process memory usage (`psutil.Process`), system memory usage (`psutil.virtual_memory`), collect garbage (`gc.collect()`), and estimate DataFrame memory usage (`df.memory_usage(deep=True)`).
2.  **Specialized Time Series Data Structure**: The `TimeSeriesArray` class stores timestamps and values separately in NumPy arrays for memory efficiency. It can be created from a DataFrame, provide views, and convert back to a DataFrame.
3.  **Memory Optimization Techniques**: Includes optimizing numeric columns (e.g., `astype(np.uint8)`, `np.int8)`, float columns (`astype(np.float32)`), and object columns (`astype('category')` if appropriate).
4.  **When to Use Different Isolation Approaches**:
    * Deep Copying: Small datasets or critical isolation needs.
    * Data Views: Read-only access to large datasets.
    * Copy-On-Write: Medium datasets with occasional modifications.
    * Shared Memory: Very large datasets accessed by multiple processes.
5.  **Pruning Strategies for Historical Data**:
    * Fixed-size window pruning: Keeps only the most recent `max_history` bars.
    * Time-based pruning: Removes data older than a specified `max_history_age`.

### Implementation Structure

The Data module is organized into subdirectories: `interfaces`, `handlers`, `loaders`, `models`, `splitters`, and `utils`.

### Key Considerations for Data Usage

#### 1. Train/Test Isolation

Crucial for optimization:

1.  **Isolation Verification**: Always check for overlapping indices between train and test sets.
2.  **Context Switching**: Reset all relevant states (e.g., `current_bar_index`, `bar_indices`) when switching between train and test splits in `set_active_split`.

#### 2. Multi-Symbol Synchronization

For multi-symbol backtesting:

1.  **Timeline Construction**: Optionally pre-compute a global timeline of all (timestamp, symbol) events and sort it.
2.  **Efficient Lookups**: Use optimized structures like `heapq` (priority queue) for managing timestamps from multiple symbols to find the next earliest event.

#### 3. Data Quality

Implement checks for:

1.  **Continuity Validation**: Check for gaps in data using expected date ranges (`pd.date_range`).
2.  **Outlier Detection**: Identify and handle outliers based on standard deviations from the mean.
3.  **Data Integrity**: Verify OHLC relationships (e.g., high >= open, low <= open).

### Usage Patterns

#### Basic Data Loading

Initialize `HistoricalDataHandler` with parameters (data directory, train/test split config), load data for symbols, and set up the train/test split.

#### Emitting Bar Events

Set the active split (e.g., "train") and loop through `data_handler.update_bars()` to process all bars, which emits events to the event bus.

#### Accessing Data

Use `get_latest_bar(symbol)` for the most recent bar and `get_latest_bars(symbol, N)` for the last N bars. Bar data (e.g., `bar.close`) can then be accessed.

### Conclusion

The Data module provides robust and efficient handling of market data for the ADMF-Trader system. It enables reliable strategy development and backtesting by incorporating proper train/test isolation, memory-efficient structures, and comprehensive data quality checks, effectively managing large datasets.
