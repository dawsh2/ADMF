# ADMF Core: Performance and Resource Optimization

## 1. Overview

Performance is a critical aspect of the ADMF-Trader system, particularly during computationally intensive operations like backtesting, optimization, and live data processing. This document outlines the strategies and frameworks for optimizing CPU usage, memory management, I/O operations, and leveraging caching to enhance system responsiveness and efficiency.

The goal is to provide a robust set of tools and patterns that allow ADMF-Trader to handle large datasets and complex computations effectively while maintaining optimal resource utilization.

## 2. Strategic Caching

Strategic caching aims to optimize performance by storing and reusing the results of expensive or frequently executed computations.

### 2.1. Caching Decorator Framework

A flexible decorator framework provides an easy way to add caching to functions and methods.

* **`@cached` Decorator**:
    * Applies caching logic to a function.
    * **Parameters**:
        * `max_size`: Maximum number of entries in the cache (e.g., 128).
        * `strategy`: Cache eviction strategy (LRU, LFU, TLRU, FIFO, UNBOUNDED).
        * `ttl`: Time-to-live for cache entries in seconds (None for no TTL).
        * `key_maker`: Optional custom function to generate cache keys from function arguments.
* **`Cache` Class**:
    * Implements the caching logic, including storage and eviction.
    * `get(key, default=None)`: Retrieves an item from the cache.
    * `set(key, value, args=None, kwargs=None)`: Stores an item.
    * `invalidate(key=None, pattern=None)`: Removes entries.
    * `clear()`: Clears all entries.
    * `stats()`: Provides cache hit/miss statistics.
* **`CacheKey` Utility**:
    * `generate_key(args, kwargs, include_self=True)`: Creates a hashable key from function arguments, handling various data types and allowing objects to define their own `cache_key()` method.
* **`CacheEntry` Class**:
    * Stores the cached value along with metadata like creation time, last access time, and access count, which are used by eviction strategies.

**Conceptual Usage:**
```python
from core.performance.caching import cached, CacheStrategy # Assuming path

@cached(max_size=100, strategy=CacheStrategy.LRU, ttl=3600)
def expensive_calculation(param1, param2):
    # ... complex computation ...
    return result

# Accessing cache controls:
# expensive_calculation.cache.clear()
# stats = expensive_calculation.cache.stats()
```

The implementation of this framework (Cache, CacheKey, CacheEntry, @cached decorator) would reside in a dedicated caching utility module within src/core/performance/.

### 2.2. Incremental Calculation Framework

For operations where results can be updated based on new data without full recomputation (e.g., moving averages, running totals).

IncrementalCalculator Base Class:
calculate(*args, **kwargs): Main method that decides whether to perform a full calculation or an incremental update.
_can_update_incrementally(*args, **kwargs): Subclass implements logic to check if an incremental update is possible.
_update_incrementally(*args, **kwargs): Subclass implements the incremental update logic.
_calculate_full(*args, **kwargs): Subclass implements the full calculation logic.
reset(): Clears internal state.
Example (MovingAverageCalculator): Demonstrates how to implement an incremental moving average.
2.3. Specific Caching Applications

Strategy and Indicator Caching:
CachedIndicator base class and specialized implementations (e.g., CachedMovingAverage).
Specialized decorators like @cached_indicator and @cached_signal using custom key makers that consider relevant context (e.g., last timestamp, strategy parameters).
Position Tracking with Incremental Updates: IncrementalPositionTracker for efficient updates to position values and total portfolio value based on new trades or price changes.
Risk Metrics Calculation: CachingRiskCalculator for caching computationally intensive risk metrics (Sharpe, drawdown), with logic for incremental updates where possible.
2.4. CacheManager

A centralized singleton (CacheManager.get_instance()) to:

Register and manage all named caches, incremental calculators, and cached functions in the system.
Provide global cache statistics (get_stats()).
Offer global cache operations like clear_all() and invalidate_by_pattern(pattern).
Enable/disable caching globally.
3. Resource Optimization Framework [Idealized Design]
This framework aims to provide tools and patterns for efficient management of memory, CPU, and I/O resources.

3.1. Memory Management (MemoryManager)

A central MemoryManager singleton (MemoryManager.get_instance()) designed to:

Monitor Memory Usage: Track overall process memory (RSS, VMS) and system memory using psutil.
Object Pooling (ObjectPool class):
Provide pools of reusable objects to reduce allocation/deallocation overhead for frequently created/destroyed objects (e.g., Bar objects, Event objects).
acquire(): Get an object from the pool.
release(obj): Return an object to the pool.
_reset_object(obj): Hook for resetting object state before reuse.
Pools register with the MemoryManager.
Memory Allocation Tracking: track_allocation(obj, size) to monitor allocations of significant objects or data structures (e.g., large NumPy arrays).
Memory Pressure Handling:
is_memory_pressure_high(): Detect when memory usage exceeds a configurable threshold.
release_memory_pressure(): Attempt to free memory by clearing caches, object pools, and triggering garbage collection (gc.collect()).
Optimized Data Structures:
MemoryOptimizedArray: A NumPy array wrapper that can track its allocation with MemoryManager and potentially implement dtype optimization (optimize_dtype()).
OptimizedDataFrame: A pandas DataFrame wrapper designed to automatically optimize column dtypes (e.g., downcasting integers/floats, using category type for low-cardinality strings) to reduce memory footprint. It would also track its memory usage via MemoryManager.
TimeSeriesArray: A specialized structure for storing time series data efficiently (timestamps and values separately).
Data Loading Strategies:
Chunked loading for large CSV files instead of reading entirely into memory.
Configuration options to limit rows loaded (max_rows).
Optimization Process Improvements:
Configuration to store only top N optimization results.
Option to force garbage collection after each optimization run.
3.2. CPU Optimization (CPUManager)

A central CPUManager singleton (CPUManager.get_instance()) designed to:

Manage Thread and Process Pools: Provide shared ThreadPoolExecutor and ProcessPoolExecutor instances.
init_thread_pool(max_workers)
init_process_pool(max_workers)
Task Submission: submit_task(fn, *args, use_process=False, task_name=None, **kwargs) for submitting individual tasks.
Map Operations: map_tasks(fn, items, use_process=False, task_name=None, chunksize=1) for parallel processing of item lists.
CPU Usage Monitoring: get_cpu_usage() to get system and process CPU utilization via psutil.
Task Execution Statistics: Track execution count, total time, min/max/avg time for named tasks if monitoring is enabled.
Shutdown: Graceful shutdown of managed pools.
3.3. I/O Optimization (IOManager)

A central IOManager singleton (IOManager.get_instance()) designed to:

Asynchronous I/O: Manage an asynchronous I/O worker thread (_async_io_worker) and task queue (_async_queue) for non-blocking file operations.
read_file_async(path, callback, ...)
write_file_async(path, data, callback, ...)
File Caching: Optionally cache frequently accessed file handles or content (though detailed design for content caching needs care regarding staleness).
Memory-Mapped Files: Support for opening files using mmap for efficient access to large files.
Buffer Pooling: get_buffer(size) and release_buffer(buffer) to reuse byte buffers for I/O operations, reducing allocations.
I/O Statistics Monitoring: Track reads, writes, bytes transferred, and open/close counts per file path if monitoring is enabled.
3.4. Resource-Aware Strategy Execution (ResourceAwareExecutor)

An idealized framework to execute strategies with dynamic resource adjustments:

Monitors system resources (CPU, memory) using the respective managers.
Handles memory/CPU pressure by triggering optimizations (e.g., clearing caches, reducing parallelism).
Can use optimized data structures (e.g., OptimizedDataFrame, MemoryOptimizedArray) for strategy data.
Can dynamically choose between parallel and sequential execution or adjust the degree of parallelism based on load and strategy characteristics.
Provides callbacks for resource pressure events.
4. Implementation Strategy for Resource Optimization
The full implementation of these resource optimization frameworks is a significant undertaking. An incremental approach is recommended:

Basic Monitoring: Implement basic CPU and memory usage reporting using psutil without full-fledged managers.
Caching Framework: Implement the @cached decorator and CacheManager as these can provide immediate benefits.
Memory Optimization in Data Handling: Focus on OptimizedDataFrame or similar techniques in DataHandler and during train/test splits, as this is often a major memory consumer. Implement chunked loading.
Object Pooling: Introduce ObjectPool for frequently used small objects like Event or Bar if profiling shows high allocation overhead.
CPUManager: Implement basic thread/process pool management for optimizers or other clearly parallelizable tasks.
IOManager: Start with asynchronous wrappers for critical file I/O if it becomes a bottleneck.
5. Best Practices for Performance
5.1. Memory Management

Profile memory usage (e.g., using MemoryTracker or external profilers) to identify hotspots.
Use memory-efficient data types (e.g., np.float32 instead of np.float64 if precision allows; category types for strings).
Prefer views over copies of large data arrays (e.g., NumPy/Pandas slicing) where modifications are not needed.
Use del to remove references to large objects that are no longer needed, allowing garbage collection.
Be mindful of Python's object overhead; for very large collections of simple data, NumPy arrays or specialized structures are often better than lists of objects.
Implement pruning strategies for historical data that grows unbounded (e.g., in DataHandler or Portfolio).
5.2. CPU Utilization

Profile CPU usage to identify bottlenecks.
Use vectorized operations (NumPy/Pandas) instead of explicit Python loops for data manipulation.
Offload CPU-bound tasks to process pools to bypass the GIL.
Offload I/O-bound tasks that can be done concurrently to thread pools or use asynchronous programming.
Be cautious with the number of threads/processes; too many can lead to excessive context switching and resource contention.
5.3. I/O Operations

Use buffered I/O.
Read/write data in larger chunks rather than small pieces.
Use asynchronous I/O for operations that don't need to block the main execution flow.
Consider data serialization formats optimized for speed and size (e.g., Parquet, Feather, HDF5) if CSV becomes a bottleneck.
5.4. Caching

Cache results of pure functions (functions that always return the same output for the same input).
Cache selectively; over-caching can consume too much memory or lead to stale data issues.
Choose appropriate cache sizes and eviction strategies based on access patterns and data characteristics.
Implement robust cache invalidation logic when underlying data or parameters change.
Set appropriate Time-To-Live (TTL) for cached data that can become stale.
By systematically applying these performance and resource optimization techniques, ADMF-Trader can achieve high efficiency and scalability, enabling complex analysis and trading operations on substantial datasets.


---
This draft for `6_CORE_PERFORMANCE_AND_OPTIMIZATION.md` combines the three performance-related documents. It's quite extensive, as were the source documents.

Please review this. We can then refine it or move on to document #7, which would be the large `CORE_INFRASTRUCTURE_SERVICES.md`. Given its size, we might want to break that one down further or tackle its sub-sections (Logging, Error Handling, Debugging, Testing, Validation) individually if you prefer
