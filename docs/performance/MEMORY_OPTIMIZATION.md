# Memory Optimization Guide

This document provides recommendations for optimizing memory usage in ADMF, especially when working with large datasets during optimization runs.

## Identified Memory Issues

1. **Large Dataset Loading**: 
   - Reading large CSV files fully into memory
   - Creating multiple copies during train/test splits
   - Retaining entire dataset during optimization runs

2. **Verbose Logging**:
   - Accumulating extensive log data
   - Keeping detailed regime detection information
   - Storing all regime performance data for each parameter combination

3. **Optimization Process**:
   - Storing detailed results for every parameter combination
   - Keeping complete regime performance data in memory
   - Not clearing intermediate results after processing

## Memory Optimization Strategies

### 1. Data Loading Optimizations

```python
# In CSVDataHandler
def load_data(self):
    # Instead of loading the entire file at once
    # Use iterative loading with chunks
    chunks = []
    for chunk in pd.read_csv(self._csv_file_path, chunksize=10000):
        # Process only what's needed from each chunk
        processed_chunk = self._preprocess_chunk(chunk)
        chunks.append(processed_chunk)
    
    self._data = pd.concat(chunks)
```

Configuration options to add:
```yaml
data_handler_csv:
  max_rows: 50000  # Limit total rows loaded
  use_chunking: true  # Enable chunked loading
  chunk_size: 10000  # Size of each chunk
```

### 2. Logging Optimizations

As implemented in the logging optimization, we:
- Reduced verbose debug output
- Added statistical summaries instead of per-bar logs
- Made logging configurable via `verbose_logging` flag

Additional recommendations:
- Consider implementing log rotation for long-running optimizations
- Add option to disable file logging during optimization runs
- Implement logging to separate files for different components

### 3. Optimization Process Improvements

Implemented changes:
- Added `clear_regime_performance` flag to free memory during optimization
- Created `top_n_to_test` parameter to control how many parameter sets are tested

Additional recommendations:
```yaml
optimizer:
  # Memory management settings
  store_all_results: false  # Only store the top performers
  max_results_to_store: 10  # Maximum number of results to keep in memory
  gc_after_each_run: true   # Force garbage collection after each parameter test
```

### 4. Code Implementation

```python
# Example implementation in the optimizer
def _perform_single_backtest_run(self, params):
    # Run the backtest
    result = super()._perform_single_backtest_run(params)
    
    # Force garbage collection after each run
    if self._gc_after_each_run:
        import gc
        gc.collect()
    
    return result

def run_grid_search(self):
    # Only keep top N results to save memory
    if not self._store_all_results:
        all_results = []
        top_results = []
        
        for params in param_combinations:
            result = self._perform_single_backtest_run(params)
            
            # Only keep the best N results
            if len(top_results) < self._max_results_to_store:
                top_results.append((params, result))
                top_results.sort(key=lambda x: x[1], reverse=self._higher_is_better)
            elif self._higher_is_better and result > top_results[-1][1]:
                top_results[-1] = (params, result)
                top_results.sort(key=lambda x: x[1], reverse=True)
            elif not self._higher_is_better and result < top_results[-1][1]:
                top_results[-1] = (params, result)
                top_results.sort(key=lambda x: x[1], reverse=False)
```

### 5. System-Level Recommendations

1. **Resource Monitoring**:
   - Add memory usage monitoring to the application
   - Implement adaptive behavior based on available memory
   - Add warning logs when memory usage exceeds thresholds

2. **Process Management**:
   - Consider splitting large optimization runs into smaller batches
   - Implement checkpointing to save progress and free memory
   - Use multiprocessing to isolate memory usage per parameter test

3. **External Data Storage**:
   - Store intermediate results to disk instead of memory
   - Use databases for large datasets instead of in-memory storage
   - Implement streaming data access patterns

## Configuration Reference

```yaml
# Memory optimization settings
system:
  memory:
    monitoring_enabled: true
    high_usage_threshold_percent: 80
    critical_usage_threshold_percent: 90
    enable_emergency_gc: true

# Data handler memory settings
data_handler_csv:
  max_rows: 50000  # Limit maximum rows
  use_chunking: true
  chunk_size: 10000
  
# Optimizer memory settings
optimizer:
  top_n_to_test: 3
  clear_regime_performance: true
  store_all_results: false
  max_results_to_store: 10
  checkpoint_interval: 50  # Save progress every N parameter combinations
```

## Implementation Status

- ✅ Logging optimization implemented
- ✅ Top-N parameter testing implemented
- ✅ Regime performance data clearing implemented
- ⬜ Chunked data loading - Planned
- ⬜ Memory monitoring - Planned
- ⬜ Checkpointing - Planned