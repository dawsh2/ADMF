# Context-Aware Thread Safety

## Overview

This document outlines the design for a context-aware thread safety mechanism in the ADMF-Trader system. This mechanism enables components to adjust their thread safety approach based on execution context, optimizing for both performance and safety.

## Problem Statement

The ADMF-Trader system operates in multiple execution contexts:

1. **Backtesting (Single-threaded)**: Historical simulations in a controlled, single-threaded environment
2. **Optimization (Multi-threaded)**: Multiple backtest instances running in parallel
3. **Live Trading (Multi-threaded)**: Real-time market data processing and trade execution

Applying full thread safety measures universally introduces significant overhead in single-threaded contexts, where such protections are unnecessary. Conversely, failing to apply proper thread safety in multi-threaded contexts risks race conditions and data corruption.

We need a solution that:
- Provides maximum thread safety when needed
- Minimizes overhead when thread safety is unnecessary
- Makes the transition between contexts seamless and transparent to components
- Prevents accidental thread safety violations

## Design Solution

### 1. Execution Context

The foundation of our context-aware thread safety is the `ExecutionContext` class:

```python
from enum import Enum, auto
from typing import Dict, Any, Optional
import threading

class ThreadingMode(Enum):
    """Threading mode options for execution contexts."""
    SINGLE_THREADED = auto()  # No thread safety needed
    MULTI_THREADED = auto()   # Full thread safety required
    AUTO_DETECT = auto()      # Automatically detect based on thread count

class ExecutionContext:
    """
    Execution context with thread safety configuration.
    
    This class manages the execution context for components, including
    thread safety configuration and runtime environment information.
    """
    
    _current_context = threading.local()  # Thread-local storage for current context
    
    @classmethod
    def get_current(cls) -> Optional['ExecutionContext']:
        """Get current execution context for this thread."""
        return getattr(cls._current_context, 'context', None)
        
    @classmethod
    def set_current(cls, context: Optional['ExecutionContext']) -> None:
        """Set current execution context for this thread."""
        cls._current_context.context = context
    
    def __init__(self, name: str, threading_mode: ThreadingMode = ThreadingMode.AUTO_DETECT):
        """
        Initialize execution context.
        
        Args:
            name: Context name
            threading_mode: Threading mode for this context
        """
        self.name = name
        self.threading_mode = threading_mode
        self._metadata = {}
        self._active_threads = set()
        self._creation_thread = threading.current_thread()
        
    def __enter__(self) -> 'ExecutionContext':
        """Enter context scope."""
        # Store previous context
        self._previous_context = ExecutionContext.get_current()
        
        # Set current context
        ExecutionContext.set_current(self)
        
        # Track active thread
        self._active_threads.add(threading.current_thread())
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context scope."""
        # Restore previous context
        ExecutionContext.set_current(self._previous_context)
        
        # Remove from active threads
        current_thread = threading.current_thread()
        if current_thread in self._active_threads:
            self._active_threads.remove(current_thread)
            
    @property
    def is_multi_threaded(self) -> bool:
        """
        Determine if context is multi-threaded.
        
        Returns:
            bool: Whether context is multi-threaded
        """
        if self.threading_mode == ThreadingMode.SINGLE_THREADED:
            return False
        elif self.threading_mode == ThreadingMode.MULTI_THREADED:
            return True
        else:  # AUTO_DETECT
            # Consider multi-threaded if multiple threads ever accessed this context
            return len(self._active_threads) > 1
            
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)
```

### 2. Thread Safety Factory

The `ThreadSafetyFactory` creates appropriate collections based on context:

```python
import threading
from typing import Dict, List, Set, TypeVar, Generic, Iterator, Any, Optional
from collections import deque

K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')

class ThreadSafetyFactory:
    """Factory for creating context-appropriate collections."""
    
    @staticmethod
    def get_context() -> Optional['ExecutionContext']:
        """Get current execution context."""
        return ExecutionContext.get_current()
        
    @staticmethod
    def is_multi_threaded() -> bool:
        """
        Determine if current context is multi-threaded.
        
        Returns:
            bool: Whether context is multi-threaded
        """
        context = ThreadSafetyFactory.get_context()
        if context:
            return context.is_multi_threaded
        else:
            # Default to safe behavior if no context
            return True
            
    @staticmethod
    def dict() -> Dict[K, V]:
        """
        Create context-appropriate dictionary.
        
        Returns:
            Thread-safe dict if multi-threaded, regular dict otherwise
        """
        if ThreadSafetyFactory.is_multi_threaded():
            return ThreadSafeDict()
        else:
            return {}
            
    @staticmethod
    def list() -> List[T]:
        """
        Create context-appropriate list.
        
        Returns:
            Thread-safe list if multi-threaded, regular list otherwise
        """
        if ThreadSafetyFactory.is_multi_threaded():
            return ThreadSafeList()
        else:
            return []
            
    @staticmethod
    def set() -> Set[T]:
        """
        Create context-appropriate set.
        
        Returns:
            Thread-safe set if multi-threaded, regular set otherwise
        """
        if ThreadSafetyFactory.is_multi_threaded():
            return ThreadSafeSet()
        else:
            return set()
            
    @staticmethod
    def deque(maxlen: Optional[int] = None) -> deque:
        """
        Create context-appropriate deque.
        
        Args:
            maxlen: Maximum length of deque
            
        Returns:
            Thread-safe deque if multi-threaded, regular deque otherwise
        """
        if ThreadSafetyFactory.is_multi_threaded():
            return ThreadSafeDeque(maxlen=maxlen)
        else:
            return deque(maxlen=maxlen)
            
    @staticmethod
    def counter(initial_value: int = 0) -> Any:
        """
        Create context-appropriate counter.
        
        Args:
            initial_value: Initial counter value
            
        Returns:
            Thread-safe counter if multi-threaded, regular integer otherwise
        """
        if ThreadSafetyFactory.is_multi_threaded():
            return AtomicCounter(initial_value)
        else:
            return SimpleCounter(initial_value)
            
    @staticmethod
    def lock() -> Any:
        """
        Create context-appropriate lock.
        
        Returns:
            Real lock if multi-threaded, dummy lock otherwise
        """
        if ThreadSafetyFactory.is_multi_threaded():
            return threading.RLock()
        else:
            return DummyLock()
```

### 3. Context-Aware Collections

#### Thread-Safe and Non-Thread-Safe Implementations

Each collection type has two implementations: thread-safe and non-thread-safe.

#### Dummy Lock for Single-Threaded Contexts

```python
class DummyLock:
    """Dummy lock for single-threaded contexts."""
    
    def __enter__(self):
        """Enter lock context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit lock context."""
        pass
        
    def acquire(self, blocking=True, timeout=-1):
        """Acquire lock (no-op)."""
        return True
        
    def release(self):
        """Release lock (no-op)."""
        pass
        
    def locked(self):
        """Check if lock is held (always False)."""
        return False
```

#### Counter Implementations

```python
class SimpleCounter:
    """Non-thread-safe counter for single-threaded contexts."""
    
    def __init__(self, initial_value=0):
        """Initialize counter."""
        self._value = initial_value
        
    @property
    def value(self):
        """Get current value."""
        return self._value
        
    def increment(self, amount=1):
        """Increment counter."""
        self._value += amount
        return self._value
        
    def decrement(self, amount=1):
        """Decrement counter."""
        self._value -= amount
        return self._value
        
    def reset(self, value=0):
        """Reset counter."""
        self._value = value

class AtomicCounter:
    """Thread-safe counter for multi-threaded contexts."""
    
    def __init__(self, initial_value=0):
        """Initialize counter."""
        self._value = initial_value
        self._lock = threading.RLock()
        
    @property
    def value(self):
        """Get current value with thread safety."""
        with self._lock:
            return self._value
            
    def increment(self, amount=1):
        """Increment counter with thread safety."""
        with self._lock:
            self._value += amount
            return self._value
            
    def decrement(self, amount=1):
        """Decrement counter with thread safety."""
        with self._lock:
            self._value -= amount
            return self._value
            
    def reset(self, value=0):
        """Reset counter with thread safety."""
        with self._lock:
            self._value = value
```

### 4. Context-Aware Component Base Class

The `Component` base class uses context-aware thread safety:

```python
from typing import Dict, Any, Optional

class Component:
    """Base class for all system components with context-aware thread safety."""
    
    def __init__(self, name, parameters=None):
        """Initialize component with name and parameters."""
        self.name = name
        self.parameters = parameters or {}
        self._initialized = False
        self._running = False
        self._lock = ThreadSafetyFactory.lock()  # Context-aware lock
        
        # Initialize context-aware collections
        self._event_subscriptions = ThreadSafetyFactory.dict()
        self._state = ThreadSafetyFactory.dict()
        
    def get_context(self) -> Optional['ExecutionContext']:
        """Get current execution context."""
        return ExecutionContext.get_current()
        
    @property
    def is_multi_threaded(self) -> bool:
        """Determine if component is operating in multi-threaded context."""
        context = self.get_context()
        if context:
            return context.is_multi_threaded
        else:
            # Default to safe behavior if no context
            return True
            
    def thread_safe_operation(self, operation, *args, **kwargs):
        """
        Execute operation with context-aware thread safety.
        
        Args:
            operation: Function to execute
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
        """
        if self.is_multi_threaded:
            with self._lock:
                return operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)
            
    @property
    def initialized(self) -> bool:
        """Thread-safe access to initialized flag."""
        if self.is_multi_threaded:
            with self._lock:
                return self._initialized
        else:
            return self._initialized
            
    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Thread-safe setting of initialized flag."""
        if self.is_multi_threaded:
            with self._lock:
                self._initialized = value
        else:
            self._initialized = value
            
    @property
    def running(self) -> bool:
        """Thread-safe access to running flag."""
        if self.is_multi_threaded:
            with self._lock:
                return self._running
        else:
            return self._running
            
    @running.setter
    def running(self, value: bool) -> None:
        """Thread-safe setting of running flag."""
        if self.is_multi_threaded:
            with self._lock:
                self._running = value
        else:
            self._running = value
```

### 5. Thread Detection Utility

```python
import threading
import time
from typing import Set, Dict, Any, List, Optional

class ThreadDetector:
    """Utility for runtime thread detection and monitoring."""
    
    def __init__(self):
        """Initialize thread detector."""
        self._active_threads = set()
        self._thread_activity = {}  # thread_id -> last_active_time
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread = None
        
    def register_thread_activity(self, thread=None) -> None:
        """
        Register thread activity.
        
        Args:
            thread: Thread to register (default: current thread)
        """
        thread = thread or threading.current_thread()
        with self._lock:
            self._active_threads.add(thread)
            self._thread_activity[thread.ident] = time.time()
            
    def get_active_threads(self) -> Set[threading.Thread]:
        """
        Get active threads.
        
        Returns:
            Set of active threads
        """
        with self._lock:
            return set(self._active_threads)
            
    def get_thread_count(self) -> int:
        """
        Get active thread count.
        
        Returns:
            Number of active threads
        """
        with self._lock:
            return len(self._active_threads)
            
    def is_multi_threaded(self) -> bool:
        """
        Determine if system is multi-threaded.
        
        Returns:
            bool: Whether multiple active threads exist
        """
        return self.get_thread_count() > 1
        
    def get_thread_activity(self) -> Dict[int, float]:
        """
        Get thread activity timestamps.
        
        Returns:
            Dict mapping thread IDs to last activity times
        """
        with self._lock:
            return dict(self._thread_activity)
            
    def clean_inactive_threads(self, max_inactive_time: float = 60.0) -> int:
        """
        Clean inactive threads.
        
        Args:
            max_inactive_time: Maximum inactive time in seconds
            
        Returns:
            Number of threads cleaned
        """
        with self._lock:
            current_time = time.time()
            inactive_threads = [
                thread for thread in self._active_threads
                if current_time - self._thread_activity.get(thread.ident, 0) > max_inactive_time
            ]
            
            for thread in inactive_threads:
                self._active_threads.remove(thread)
                if thread.ident in self._thread_activity:
                    del self._thread_activity[thread.ident]
                    
            return len(inactive_threads)
            
    def start_monitoring(self, interval: float = 5.0) -> None:
        """
        Start thread monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
            
        self._monitoring = True
        
        def monitor_func():
            while self._monitoring:
                self.clean_inactive_threads()
                time.sleep(interval)
                
        self._monitor_thread = threading.Thread(
            target=monitor_func,
            name="ThreadDetectorMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop thread monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
```

### 6. Context-Aware Thread-Safe Collections

These collections adapt their behavior based on threading mode:

```python
class ContextAwareDict(Dict[K, V]):
    """Context-aware dictionary implementation."""
    
    def __init__(self, *args, **kwargs):
        """Initialize dictionary."""
        super().__init__(*args, **kwargs)
        self._lock = ThreadSafetyFactory.lock()
        
    def __getitem__(self, key: K) -> V:
        """Get item with context-aware thread safety."""
        if ThreadSafetyFactory.is_multi_threaded():
            with self._lock:
                return super().__getitem__(key)
        else:
            return super().__getitem__(key)
            
    def __setitem__(self, key: K, value: V) -> None:
        """Set item with context-aware thread safety."""
        if ThreadSafetyFactory.is_multi_threaded():
            with self._lock:
                super().__setitem__(key, value)
        else:
            super().__setitem__(key, value)
            
    def __delitem__(self, key: K) -> None:
        """Delete item with context-aware thread safety."""
        if ThreadSafetyFactory.is_multi_threaded():
            with self._lock:
                super().__delitem__(key)
        else:
            super().__delitem__(key)
            
    # Additional methods follow similar pattern
```

## System Integration

### 1. Execution Context Configuration

The execution context is configured in system bootstrap:

```python
def bootstrap(config):
    """Bootstrap system with appropriate execution context."""
    # Get threading mode from configuration
    threading_mode_str = config.get('system.threading_mode', 'auto_detect')
    
    # Convert to enum
    if threading_mode_str.lower() == 'single_threaded':
        threading_mode = ThreadingMode.SINGLE_THREADED
    elif threading_mode_str.lower() == 'multi_threaded':
        threading_mode = ThreadingMode.MULTI_THREADED
    else:
        threading_mode = ThreadingMode.AUTO_DETECT
        
    # Create execution context
    context = ExecutionContext('main', threading_mode)
    
    # Enter context
    with context:
        # Initialize components
        # ...
```

### 2. Optimization Context

For optimization, create separate contexts for each run:

```python
def run_optimization(parameter_sets, data_handler):
    """Run optimization with separate contexts for each run."""
    results = []
    
    # Process each parameter set in its own context
    for params in parameter_sets:
        # Create context for this optimization run
        with ExecutionContext(f'optimization_{params["id"]}', ThreadingMode.MULTI_THREADED):
            # Run backtest
            result = run_backtest(params, data_handler)
            results.append(result)
            
    return results
```

### 3. Live Trading Context

For live trading, use a multi-threaded context:

```python
def start_live_trading(config):
    """Start live trading with multi-threaded context."""
    # Create live trading context
    with ExecutionContext('live_trading', ThreadingMode.MULTI_THREADED):
        # Initialize system
        system = initialize_system(config)
        
        # Start system
        system.start()
        
        # Run until stopped
        try:
            while system.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop()
```

## Thread Safety Benchmarking

To measure the performance impact of context-aware thread safety:

```python
import time
from contextlib import contextmanager

@contextmanager
def timing(name):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f}ms")

def benchmark_thread_safety():
    """Benchmark thread safety overhead."""
    # Single-threaded context benchmark
    with ExecutionContext('single_threaded', ThreadingMode.SINGLE_THREADED):
        # Standard dict operations
        with timing("Standard dict (single-threaded)"):
            d = ThreadSafetyFactory.dict()
            for i in range(100000):
                d[i] = i
            for i in range(100000):
                x = d[i]
                
    # Multi-threaded context benchmark
    with ExecutionContext('multi_threaded', ThreadingMode.MULTI_THREADED):
        # Thread-safe dict operations
        with timing("Thread-safe dict (multi-threaded)"):
            d = ThreadSafetyFactory.dict()
            for i in range(100000):
                d[i] = i
            for i in range(100000):
                x = d[i]
```

## Implementation Strategy

### 1. Core Implementation

1. Implement `ExecutionContext` class
2. Implement `ThreadSafetyFactory` class
3. Implement context-aware collections
4. Update `Component` base class

### 2. Integration

1. Update system bootstrap to create appropriate execution context
2. Update component initialization to use context-aware collections
3. Update optimization framework to use separate contexts
4. Update live trading system to use multi-threaded context

### 3. Testing

1. Create benchmarks to measure performance differences
2. Test context-aware components in both single and multi-threaded contexts
3. Verify thread safety in multi-threaded contexts
4. Ensure performance optimization in single-threaded contexts

## Best Practices

### 1. Threading Mode Guidelines

- Use **SINGLE_THREADED** for:
  - Simple backtesting without optimization
  - When performance is critical and no parallelism is needed

- Use **MULTI_THREADED** for:
  - Live trading environments
  - When multiple components run in parallel
  - When external dependencies may create threads

- Use **AUTO_DETECT** for:
  - General-purpose code that might run in different contexts
  - When thread usage patterns are not known in advance

### 2. Collection Usage Guidelines

- Use `ThreadSafetyFactory` to create collections:
  ```python
  self.positions = ThreadSafetyFactory.dict()
  self.trades = ThreadSafetyFactory.list()
  ```

- Avoid mixing standard and thread-safe collections:
  ```python
  # Incorrect - mixing types
  if is_multi_threaded:
      self.positions = ThreadSafeDict()
  else:
      self.positions = {}
  ```

- Use factory methods consistently:
  ```python
  # Correct
  self.positions = ThreadSafetyFactory.dict()
  ```

### 3. Thread Safety Documentation

Document threading behavior for all components:

```python
class Portfolio(Component):
    """
    Portfolio component for tracking positions and equity.
    
    Thread Safety:
    - Context-aware thread safety with ThreadSafetyFactory
    - Thread-safe in multi-threaded contexts
    - Optimized for single-threaded contexts
    """
```

## Conclusion

The context-aware thread safety mechanism provides an optimal balance between performance and safety. By adapting thread safety measures based on execution context, the system can maximize performance in single-threaded environments while ensuring robustness in multi-threaded contexts.

This approach eliminates the need for manual thread safety configuration, providing seamless transitions between different execution contexts without requiring code changes. Components can focus on their core functionality without worrying about the threading model, as the thread safety infrastructure adapts automatically.