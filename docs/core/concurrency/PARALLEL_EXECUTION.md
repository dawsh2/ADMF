# Parallel Execution

## Overview

The ADMF-Trader system supports parallel execution for optimization, backtesting, and data processing tasks. This capability enables significant performance improvements by utilizing multiple CPU cores and efficient resource management. This document outlines the parallel execution architecture, patterns, and best practices.

## Parallel Execution Modes

The system supports three primary parallel execution modes:

1. **Process-Based Parallelism**: Multiple separate processes for isolation and performance
2. **Thread-Based Parallelism**: Multiple threads within a process for shared memory optimization
3. **Hybrid Parallelism**: Combination of processes and threads for optimal performance

## Parallel Optimization

Optimization is a natural fit for parallelization as individual parameter combinations can be evaluated independently:

```python
class ParallelOptimizer:
    """Optimizer using parallel execution."""
    
    def __init__(self, workers=None, mode='process'):
        """
        Initialize parallel optimizer.
        
        Args:
            workers: Number of worker processes/threads (defaults to CPU count)
            mode: Parallelism mode ('process', 'thread', or 'hybrid')
        """
        self.workers = workers or multiprocessing.cpu_count()
        self.mode = mode
        
    def optimize(self, parameter_space, objective_function, constraints=None):
        """
        Perform parallel optimization.
        
        Args:
            parameter_space: Parameter space to search
            objective_function: Function to evaluate parameters
            constraints: Optional constraints on parameters
            
        Returns:
            dict: Optimization results
        """
        # Get parameter combinations
        combinations = parameter_space.get_combinations()
        total_combinations = len(combinations)
        
        # Create pool based on mode
        if self.mode == 'process':
            pool = multiprocessing.Pool(processes=self.workers)
            map_func = pool.map
        elif self.mode == 'thread':
            pool = ThreadPool(processes=self.workers)
            map_func = pool.map
        else:  # hybrid
            pool = self._create_hybrid_pool()
            map_func = pool.map
            
        try:
            # Apply constraints if provided
            if constraints:
                combinations = [
                    combo for combo in combinations
                    if all(constraint(combo) for constraint in constraints)
                ]
                
            # Execute optimization in parallel
            results = map_func(objective_function, combinations)
            
            # Combine results with parameter combinations
            combined_results = list(zip(combinations, results))
            
            # Sort by objective function value (assuming higher is better)
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            # Get best result
            best_params, best_score = combined_results[0]
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': combined_results,
                'total_evaluations': len(combined_results),
                'total_combinations': total_combinations
            }
        finally:
            # Clean up
            pool.close()
            pool.join()
```

## Parallel Backtesting

Multiple backtests can be run in parallel:

```python
def run_parallel_backtests(strategy_configs, data_handler, workers=None):
    """
    Run multiple backtests in parallel.
    
    Args:
        strategy_configs: List of strategy configurations
        data_handler: Data handler to use
        workers: Number of worker processes (defaults to CPU count)
        
    Returns:
        list: Backtest results
    """
    # Set default workers
    workers = workers or multiprocessing.cpu_count()
    
    # Create process pool
    pool = multiprocessing.Pool(processes=workers)
    
    try:
        # Map backtest function to strategy configurations
        results = pool.map(
            functools.partial(_run_backtest, data_handler=data_handler),
            strategy_configs
        )
        
        return results
    finally:
        # Clean up
        pool.close()
        pool.join()
        
def _run_backtest(strategy_config, data_handler):
    """
    Run a single backtest.
    
    Args:
        strategy_config: Strategy configuration
        data_handler: Data handler
        
    Returns:
        dict: Backtest result
    """
    # Create isolated components for this backtest
    container = Container()
    
    # Register components
    container.register_instance('data_handler', data_handler.clone())
    container.register('event_bus', EventBus)
    container.register_instance('config', strategy_config)
    
    # Create and register strategy
    strategy_class = get_strategy_class(strategy_config.get('class'))
    strategy = strategy_class(strategy_config.get('parameters', {}))
    container.register_instance('strategy', strategy)
    
    # Create backtest coordinator
    coordinator = BacktestCoordinator(container)
    
    # Run backtest
    return coordinator.run()
```

## Parallel Data Processing

Data processing tasks can be parallelized for improved performance:

```python
class ParallelDataProcessor:
    """Data processor using parallel execution."""
    
    def __init__(self, workers=None):
        """
        Initialize parallel data processor.
        
        Args:
            workers: Number of worker processes (defaults to CPU count)
        """
        self.workers = workers or multiprocessing.cpu_count()
        
    def process_symbols(self, symbols, processor_func, **kwargs):
        """
        Process multiple symbols in parallel.
        
        Args:
            symbols: List of symbols to process
            processor_func: Function to process each symbol
            **kwargs: Additional arguments for processor function
            
        Returns:
            dict: Mapping of symbols to processing results
        """
        # Create process pool
        pool = multiprocessing.Pool(processes=self.workers)
        
        try:
            # Map processor function to symbols
            results = pool.map(
                functools.partial(processor_func, **kwargs),
                symbols
            )
            
            # Combine results with symbols
            return dict(zip(symbols, results))
        finally:
            # Clean up
            pool.close()
            pool.join()
```

## Process Isolation

Processes provide strong isolation, preventing data leakage and memory issues:

```python
def run_isolated_process(func, *args, **kwargs):
    """
    Run a function in an isolated process.
    
    Args:
        func: Function to run
        *args: Arguments for function
        **kwargs: Keyword arguments for function
        
    Returns:
        Any: Function result
    """
    # Create queue for result
    result_queue = multiprocessing.Queue()
    
    # Define wrapper function
    def wrapper(queue, f, a, kw):
        try:
            result = f(*a, **kw)
            queue.put(('result', result))
        except Exception as e:
            import traceback
            queue.put(('exception', (str(e), traceback.format_exc())))
    
    # Create and start process
    process = multiprocessing.Process(
        target=wrapper,
        args=(result_queue, func, args, kwargs)
    )
    process.start()
    
    # Wait for process to complete
    process.join()
    
    # Get result
    result_type, result_value = result_queue.get()
    
    # Handle result
    if result_type == 'exception':
        error_msg, traceback_str = result_value
        raise Exception(f"Error in isolated process: {error_msg}\n{traceback_str}")
    else:
        return result_value
```

## Thread Pools

Thread pools enable efficient parallel execution with shared memory:

```python
class ThreadPool:
    """Pool of worker threads for parallel execution."""
    
    def __init__(self, workers=None):
        """
        Initialize thread pool.
        
        Args:
            workers: Number of worker threads (defaults to CPU count)
        """
        self.workers = workers or multiprocessing.cpu_count()
        self.queue = queue.Queue()
        self.results = {}
        self.exception = None
        self.lock = threading.RLock()
        self.completed = 0
        self.total = 0
        self.threads = []
        self.running = False
        
    def start(self):
        """Start worker threads."""
        self.running = True
        
        # Create and start worker threads
        for i in range(self.workers):
            thread = threading.Thread(
                target=self._worker,
                name=f"worker-{i}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
    def stop(self):
        """Stop worker threads."""
        self.running = False
        
        # Signal all threads to stop
        for _ in range(len(self.threads)):
            self.queue.put(None)
            
        # Wait for threads to terminate
        for thread in self.threads:
            thread.join()
            
        self.threads = []
        
    def map(self, func, items):
        """
        Apply function to items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            
        Returns:
            list: Results in the same order as items
        """
        # Reset state
        self.results = {}
        self.exception = None
        self.completed = 0
        self.total = len(items)
        
        # Start pool if not already running
        was_running = self.running
        if not was_running:
            self.start()
            
        try:
            # Add items to queue
            for i, item in enumerate(items):
                self.queue.put((i, func, item))
                
            # Wait for completion or exception
            while self.completed < self.total:
                if self.exception:
                    raise self.exception
                time.sleep(0.01)
                
            # Check for exception
            if self.exception:
                raise self.exception
                
            # Return results in order
            return [self.results[i] for i in range(self.total)]
        finally:
            # Stop pool if we started it
            if not was_running:
                self.stop()
                
    def _worker(self):
        """Worker thread function."""
        while self.running:
            try:
                # Get task from queue
                task = self.queue.get(timeout=0.1)
                
                # Check for termination signal
                if task is None:
                    break
                    
                # Unpack task
                index, func, item = task
                
                try:
                    # Process item
                    result = func(item)
                    
                    # Store result
                    with self.lock:
                        self.results[index] = result
                        self.completed += 1
                except Exception as e:
                    # Store exception
                    with self.lock:
                        self.exception = e
                        
                # Mark task as done
                self.queue.task_done()
            except queue.Empty:
                # No tasks available
                pass
```

## Resource Management

Proper resource management is critical for parallel execution:

```python
class ResourceManager:
    """Manages resources for parallel execution."""
    
    def __init__(self, max_memory_percent=80, max_cpu_percent=90):
        """
        Initialize resource manager.
        
        Args:
            max_memory_percent: Maximum memory usage percentage
            max_cpu_percent: Maximum CPU usage percentage
        """
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self._lock = threading.RLock()
        
    def get_optimal_workers(self):
        """
        Get optimal number of worker processes based on system resources.
        
        Returns:
            int: Optimal number of workers
        """
        with self._lock:
            # Get CPU count
            cpu_count = multiprocessing.cpu_count()
            
            # Get memory information
            memory_info = self._get_memory_info()
            memory_available = memory_info.get('available', 0)
            memory_total = memory_info.get('total', 0)
            
            if memory_total > 0:
                # Calculate memory-based worker count
                memory_percent = (memory_total - memory_available) / memory_total * 100
                memory_headroom = self.max_memory_percent - memory_percent
                
                if memory_headroom <= 0:
                    # No memory headroom, use minimum
                    return 1
                    
                # Scale worker count based on memory headroom
                memory_workers = max(1, int(cpu_count * memory_headroom / self.max_memory_percent))
            else:
                # Memory info not available, use CPU count
                memory_workers = cpu_count
                
            # Get CPU usage
            cpu_percent = self._get_cpu_percent()
            
            if cpu_percent > 0:
                # Calculate CPU-based worker count
                cpu_headroom = self.max_cpu_percent - cpu_percent
                
                if cpu_headroom <= 0:
                    # No CPU headroom, use minimum
                    return 1
                    
                # Scale worker count based on CPU headroom
                cpu_workers = max(1, int(cpu_count * cpu_headroom / self.max_cpu_percent))
            else:
                # CPU info not available, use CPU count
                cpu_workers = cpu_count
                
            # Use minimum of memory and CPU worker counts
            return min(memory_workers, cpu_workers)
            
    def _get_memory_info(self):
        """
        Get memory information.
        
        Returns:
            dict: Memory information
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            }
        except:
            return {}
            
    def _get_cpu_percent(self):
        """
        Get CPU usage percentage.
        
        Returns:
            float: CPU usage percentage
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0
```

## Thread-Safe Data Structures

Specialized data structures for thread-safe operation:

```python
class ThreadSafeDict:
    """Thread-safe dictionary."""
    
    def __init__(self):
        """Initialize thread-safe dictionary."""
        self._dict = {}
        self._lock = threading.RLock()
        
    def __getitem__(self, key):
        """Get item by key."""
        with self._lock:
            return self._dict[key]
            
    def __setitem__(self, key, value):
        """Set item by key."""
        with self._lock:
            self._dict[key] = value
            
    def __delitem__(self, key):
        """Delete item by key."""
        with self._lock:
            del self._dict[key]
            
    def __contains__(self, key):
        """Check if key exists."""
        with self._lock:
            return key in self._dict
            
    def get(self, key, default=None):
        """Get item with default."""
        with self._lock:
            return self._dict.get(key, default)
            
    def update(self, other):
        """Update with items from other dictionary."""
        with self._lock:
            self._dict.update(other)
            
    def items(self):
        """Get items snapshot."""
        with self._lock:
            return list(self._dict.items())
            
    def keys(self):
        """Get keys snapshot."""
        with self._lock:
            return list(self._dict.keys())
            
    def values(self):
        """Get values snapshot."""
        with self._lock:
            return list(self._dict.values())
```

## Task Queues

Task queues for efficient workload distribution:

```python
class TaskQueue:
    """Queue for distributing tasks to workers."""
    
    def __init__(self, max_size=0):
        """
        Initialize task queue.
        
        Args:
            max_size: Maximum queue size (0 for unlimited)
        """
        self.queue = queue.Queue(maxsize=max_size)
        self.results = {}
        self._lock = threading.RLock()
        self.task_count = 0
        
    def add_task(self, task_id, task_func, *args, **kwargs):
        """
        Add a task to the queue.
        
        Args:
            task_id: Unique task identifier
            task_func: Function to execute
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            str: Task ID
        """
        with self._lock:
            self.task_count += 1
            
        self.queue.put((task_id, task_func, args, kwargs))
        return task_id
        
    def get_result(self, task_id, timeout=None):
        """
        Get task result.
        
        Args:
            task_id: Task identifier
            timeout: Optional timeout in seconds
            
        Returns:
            Any: Task result
            
        Raises:
            KeyError: If task not found
            TimeoutError: If timeout exceeded
        """
        start_time = time.time()
        
        while timeout is None or time.time() - start_time < timeout:
            with self._lock:
                if task_id in self.results:
                    return self.results[task_id]
                    
            time.sleep(0.01)
            
        raise TimeoutError(f"Timeout waiting for task {task_id}")
        
    def process_tasks(self, worker_count=1):
        """
        Process tasks in the queue.
        
        Args:
            worker_count: Number of worker threads
            
        Returns:
            int: Number of tasks processed
        """
        workers = []
        task_counter = [0]  # Mutable object for counting
        
        # Create and start worker threads
        for i in range(worker_count):
            thread = threading.Thread(
                target=self._worker,
                args=(task_counter,),
                name=f"worker-{i}"
            )
            thread.daemon = True
            thread.start()
            workers.append(thread)
            
        # Wait for all tasks to be processed
        self.queue.join()
        
        # Signal all threads to stop
        for _ in range(worker_count):
            self.queue.put((None, None, None, None))
            
        # Wait for threads to terminate
        for thread in workers:
            thread.join()
            
        return task_counter[0]
        
    def _worker(self, task_counter):
        """
        Worker thread function.
        
        Args:
            task_counter: Mutable counter for processed tasks
        """
        while True:
            # Get task from queue
            task_id, task_func, args, kwargs = self.queue.get()
            
            # Check for termination signal
            if task_id is None:
                self.queue.task_done()
                break
                
            try:
                # Execute task
                result = task_func(*args, **kwargs)
                
                # Store result
                with self._lock:
                    self.results[task_id] = result
                    
                # Increment counter
                task_counter[0] += 1
            except Exception as e:
                # Store exception
                with self._lock:
                    self.results[task_id] = e
            finally:
                # Mark task as done
                self.queue.task_done()
```

## Progress Reporting

Progress reporting for long-running parallel tasks:

```python
class ParallelProgress:
    """Progress reporting for parallel operations."""
    
    def __init__(self, total, description=None):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Optional description
        """
        self.total = total
        self.description = description or "Processing"
        self.completed = multiprocessing.Value('i', 0)
        self.start_time = time.time()
        self.last_update = time.time()
        self.update_interval = 0.5  # seconds
        
    def update(self, increment=1):
        """
        Update progress.
        
        Args:
            increment: Number of items completed
            
        Returns:
            bool: Whether a progress update was displayed
        """
        with self.completed.get_lock():
            self.completed.value += increment
            completed = self.completed.value
            
        # Check if update interval has passed
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            self.display()
            return True
            
        return False
        
    def display(self):
        """Display progress information."""
        completed = self.completed.value
        percent = completed / self.total * 100 if self.total > 0 else 0
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Calculate estimated time remaining
        if completed > 0:
            items_per_second = completed / elapsed
            remaining = (self.total - completed) / items_per_second if items_per_second > 0 else 0
        else:
            remaining = 0
            
        # Format times
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)
        
        # Display progress
        print(f"\r{self.description}: {completed}/{self.total} ({percent:.1f}%) - Elapsed: {elapsed_str}, Remaining: {remaining_str}", end="")
        
        # Add newline if complete
        if completed >= self.total:
            print()
            
    def _format_time(self, seconds):
        """
        Format time in seconds.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
```

## Best Practices

1. **Resource Management**: Be mindful of system resources (CPU, memory)
2. **Process vs Thread**: Use processes for isolation, threads for shared memory
3. **Data Partitioning**: Efficiently partition data for parallel processing
4. **Thread Safety**: Ensure thread-safe access to shared resources
5. **Error Handling**: Properly handle and propagate errors from worker processes/threads
6. **Graceful Shutdown**: Implement clean shutdown of parallel tasks
7. **State Isolation**: Ensure proper state isolation between parallel tasks
8. **Progress Reporting**: Provide visibility into long-running parallel operations
9. **Resource Limits**: Set appropriate limits on resource usage
10. **Parallelism Level**: Adapt parallelism level to available resources

## Configuration

Parallel execution can be configured through the system configuration:

```yaml
parallel:
  # Enable or disable parallel execution
  enabled: true
  
  # Parallelism mode (process, thread, or hybrid)
  mode: process
  
  # Maximum number of workers (0 for automatic)
  max_workers: 0
  
  # Resource limits
  resources:
    max_memory_percent: 80
    max_cpu_percent: 90
```

By implementing these parallel execution patterns, the ADMF-Trader system can efficiently utilize available resources to significantly improve performance for compute-intensive tasks.