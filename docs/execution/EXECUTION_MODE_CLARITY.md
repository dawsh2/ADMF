# Execution Mode Clarity

## Overview

This document defines explicit execution modes and thread management guidelines for the ADMF-Trader system. The execution mode framework provides clear distinctions between backtesting and live trading environments, ensuring consistent behavior and appropriate thread management across different operating contexts.

## Problem Statement

The ADMF-Trader system operates in multiple execution contexts with different threading requirements:

1. **Backtesting**: Historical simulation with varying thread requirements
2. **Optimization**: Multiple backtest instances running in parallel
3. **Live Trading**: Real-time market data and order processing

Without clearly defined execution modes, several issues can arise:

- Inconsistent thread safety applications leading to either race conditions or performance overhead
- Unclear concurrency assumptions across components
- Thread management inconsistencies across execution contexts
- Performance bottlenecks from unnecessary synchronization overhead
- Difficulty reasoning about system behavior in different execution modes

## Execution Mode Definitions

### 1. Execution Mode Enumeration

The system defines the following execution modes:

```python
from enum import Enum, auto

class ExecutionMode(Enum):
    """Execution mode options for the ADMF-Trader system."""
    BACKTEST_SINGLE = auto()  # Single-threaded backtesting (fast, no thread safety needed)
    BACKTEST_PARALLEL = auto()  # Multi-threaded backtest components (thread safety required)
    OPTIMIZATION = auto()      # Parallel optimization (multiple backtest instances, thread safety required)
    LIVE_TRADING = auto()      # Live market trading (multi-threaded, thread safety required)
    PAPER_TRADING = auto()     # Simulated live trading (multi-threaded, thread safety required)
    REPLAY = auto()            # Event replay mode (configurable threading model)
```

### 2. Thread Model Enumeration

Each execution mode has an associated thread model:

```python
from enum import Enum, auto

class ThreadModel(Enum):
    """Thread model options for execution contexts."""
    SINGLE_THREADED = auto()    # All operations in a single thread
    MULTI_THREADED = auto()     # Operations can occur across multiple threads
    PROCESS_PARALLEL = auto()   # Parallel processes with internal thread management
    ASYNC_SINGLE = auto()       # Single event loop, asynchronous processing
    ASYNC_MULTI = auto()        # Multiple event loops, asynchronous processing
    MIXED = auto()              # Mixed model with custom thread management
```

### 3. Mode-Model Mapping

The default mapping between execution modes and thread models:

```python
DEFAULT_THREAD_MODELS = {
    ExecutionMode.BACKTEST_SINGLE: ThreadModel.SINGLE_THREADED,
    ExecutionMode.BACKTEST_PARALLEL: ThreadModel.MULTI_THREADED,
    ExecutionMode.OPTIMIZATION: ThreadModel.PROCESS_PARALLEL,
    ExecutionMode.LIVE_TRADING: ThreadModel.ASYNC_MULTI,  # Live trading uses async by default
    ExecutionMode.PAPER_TRADING: ThreadModel.ASYNC_MULTI, # Paper trading uses async by default
    ExecutionMode.REPLAY: ThreadModel.SINGLE_THREADED
}
```

## Execution Context Implementation

### 1. Execution Context Class

The `ExecutionContext` class encapsulates the execution environment:

```python
import threading
from typing import Dict, Any, Optional

class ExecutionContext:
    """
    Execution context for the ADMF-Trader system.
    
    This class encapsulates the execution mode, thread model, and related
    configuration for a particular execution run.
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
    
    def __init__(self, 
                 name: str,
                 execution_mode: ExecutionMode,
                 thread_model: Optional[ThreadModel] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize execution context.
        
        Args:
            name: Context name
            execution_mode: Execution mode for this context
            thread_model: Thread model for this context (defaults based on execution mode)
            config: Additional configuration
        """
        self.name = name
        self.execution_mode = execution_mode
        
        # Use default thread model if not specified
        self.thread_model = thread_model or DEFAULT_THREAD_MODELS.get(
            execution_mode, ThreadModel.SINGLE_THREADED)
            
        self.config = config or {}
        self._metadata = {}
        self._active_threads = set()
        self._creation_thread = threading.current_thread()
        self._previous_context = None
        
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
        return self.thread_model in (ThreadModel.MULTI_THREADED, ThreadModel.PROCESS_PARALLEL, ThreadModel.MIXED)
        
    @property
    def is_async(self) -> bool:
        """
        Determine if context is asynchronous.
        
        Returns:
            bool: Whether context uses async/await pattern
        """
        return self.thread_model in (ThreadModel.ASYNC_SINGLE, ThreadModel.ASYNC_MULTI)
        
    @property
    def is_backtest(self) -> bool:
        """
        Determine if context is a backtest.
        
        Returns:
            bool: Whether context is any type of backtest
        """
        return self.execution_mode in (
            ExecutionMode.BACKTEST_SINGLE, 
            ExecutionMode.BACKTEST_PARALLEL,
            ExecutionMode.OPTIMIZATION
        )
        
    @property
    def is_live(self) -> bool:
        """
        Determine if context is live trading.
        
        Returns:
            bool: Whether context is live trading
        """
        return self.execution_mode in (
            ExecutionMode.LIVE_TRADING, 
            ExecutionMode.PAPER_TRADING
        )
        
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

### 2. System Bootstrap

Setting up the execution context during system bootstrap:

```python
def bootstrap_system(config: Dict[str, Any]) -> Any:
    """
    Bootstrap system with execution context based on configuration.
    
    Args:
        config: System configuration
        
    Returns:
        System instance
    """
    # Extract execution mode from config
    mode_name = config.get('execution.mode', 'BACKTEST_SINGLE')
    try:
        execution_mode = ExecutionMode[mode_name]
    except KeyError:
        execution_mode = ExecutionMode.BACKTEST_SINGLE
        
    # Extract thread model from config (optional)
    thread_model_name = config.get('execution.thread_model')
    thread_model = None
    if thread_model_name:
        try:
            thread_model = ThreadModel[thread_model_name]
        except KeyError:
            pass  # Use default based on execution mode
            
    # Create execution context
    context = ExecutionContext(
        name=config.get('system.name', 'ADMF-Trader'),
        execution_mode=execution_mode,
        thread_model=thread_model,
        config=config
    )
    
    # Set up system within context
    with context:
        # Initialize component container
        container = create_container(config, context)
        
        # Create and initialize system
        system = create_system(container, config)
        
        return system
```

## Mode-Specific Behavior

### 1. Backtest Modes

#### 1.1 Single-Threaded Backtest

Single-threaded backtest mode prioritizes performance over thread safety:

```python
def run_single_threaded_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run single-threaded backtest.
    
    Args:
        config: Backtest configuration
        
    Returns:
        dict: Backtest results
    """
    # Create execution context for single-threaded backtest
    context = ExecutionContext(
        name=f"backtest_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        execution_mode=ExecutionMode.BACKTEST_SINGLE,
        config=config
    )
    
    with context:
        # Initialize components
        container = create_container(config, context)
        
        # Run backtest
        backtest_engine = container.resolve('backtest_engine')
        results = backtest_engine.run()
        
        return results
```

#### 1.2 Parallel Backtest

Multi-threaded backtest mode with thread safety for all components:

```python
def run_parallel_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run parallel backtest with multi-threaded components.
    
    Args:
        config: Backtest configuration
        
    Returns:
        dict: Backtest results
    """
    # Create execution context for parallel backtest
    context = ExecutionContext(
        name=f"parallel_backtest_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        execution_mode=ExecutionMode.BACKTEST_PARALLEL,
        config=config
    )
    
    with context:
        # Initialize components
        container = create_container(config, context)
        
        # Run backtest with parallelized components
        backtest_engine = container.resolve('backtest_engine')
        results = backtest_engine.run()
        
        return results
```

#### 1.3 Optimization

Multi-process optimization mode with parallel backtest instances:

```python
def run_optimization(config: Dict[str, Any], parameter_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run optimization with multiple backtest instances.
    
    Args:
        config: Base configuration
        parameter_sets: List of parameter sets to evaluate
        
    Returns:
        list: List of backtest results for each parameter set
    """
    # Create execution context for optimization
    context = ExecutionContext(
        name=f"optimization_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        execution_mode=ExecutionMode.OPTIMIZATION,
        config=config
    )
    
    with context:
        # Initialize optimizer
        optimizer = OptimizationEngine(config)
        
        # Run optimization
        results = optimizer.run_parameter_sweep(parameter_sets)
        
        return results
```

### 2. Live Trading Modes

#### 2.1 Live Trading

Multi-threaded live trading mode with full thread safety:

```python
def run_live_trading(config: Dict[str, Any]) -> None:
    """
    Run live trading system.
    
    Args:
        config: Live trading configuration
    """
    # Create execution context for live trading
    context = ExecutionContext(
        name=f"live_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        execution_mode=ExecutionMode.LIVE_TRADING,
        config=config
    )
    
    with context:
        # Initialize components
        container = create_container(config, context)
        
        # Start live trading system
        trading_system = container.resolve('trading_system')
        trading_system.start()
        
        try:
            # Run until stopped
            while trading_system.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            # Handle graceful shutdown
            trading_system.stop()
```

#### 2.2 Paper Trading

Multi-threaded paper trading mode with simulated execution:

```python
def run_paper_trading(config: Dict[str, Any]) -> None:
    """
    Run paper trading system (simulated execution).
    
    Args:
        config: Paper trading configuration
    """
    # Configure for paper trading
    paper_config = copy.deepcopy(config)
    paper_config['broker.type'] = 'simulated'
    paper_config['broker.simulation.enabled'] = True
    
    # Create execution context for paper trading
    context = ExecutionContext(
        name=f"paper_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        execution_mode=ExecutionMode.PAPER_TRADING,
        config=paper_config
    )
    
    with context:
        # Initialize components
        container = create_container(paper_config, context)
        
        # Start paper trading system
        trading_system = container.resolve('trading_system')
        trading_system.start()
        
        try:
            # Run until stopped
            while trading_system.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            # Handle graceful shutdown
            trading_system.stop()
```

## Thread Management Guidelines

### 1. Thread Safety Requirements

Each thread model has specific thread safety requirements:

| Thread Model | Thread Safety Requirements |
|--------------|----------------------------|
| SINGLE_THREADED | No thread safety required; all operations occur in the same thread |
| MULTI_THREADED | Full thread safety required; any operation may occur from any thread |
| PROCESS_PARALLEL | Thread safety within processes; inter-process communication through safe channels |
| MIXED | Component-specific thread safety based on documented requirements |

### 2. Thread Pool Management

Guidelines for thread pool management across execution modes:

```python
class ThreadPoolManager:
    """
    Thread pool manager for execution modes.
    
    Creates and manages appropriate thread pools based on execution mode.
    """
    
    def __init__(self, context: ExecutionContext):
        """
        Initialize thread pool manager.
        
        Args:
            context: Execution context
        """
        self.context = context
        self._thread_pools = {}
        
        # Configure based on execution mode and thread model
        self._configure()
        
    def _configure(self):
        """Configure thread pools based on context."""
        if self.context.thread_model == ThreadModel.SINGLE_THREADED:
            # No thread pools needed
            pass
        elif self.context.thread_model == ThreadModel.MULTI_THREADED:
            # Create thread pools based on execution mode
            if self.context.execution_mode == ExecutionMode.BACKTEST_PARALLEL:
                # Limited threads for parallel backtest
                max_workers = min(os.cpu_count(), 4)
                self._thread_pools['data'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, 
                    thread_name_prefix='data_worker'
                )
                self._thread_pools['compute'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, 
                    thread_name_prefix='compute_worker'
                )
            elif self.context.is_live:
                # More threads for live trading
                self._thread_pools['market_data'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, 
                    thread_name_prefix='market_data_worker'
                )
                self._thread_pools['order_processing'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, 
                    thread_name_prefix='order_worker'
                )
                self._thread_pools['strategy'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, 
                    thread_name_prefix='strategy_worker'
                )
        elif self.context.thread_model == ThreadModel.PROCESS_PARALLEL:
            # Process pool for optimization
            max_workers = max(1, os.cpu_count() - 1)
            self._thread_pools['optimization'] = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            )
            
    def get_executor(self, pool_name: str) -> concurrent.futures.Executor:
        """
        Get thread pool executor.
        
        Args:
            pool_name: Name of thread pool
            
        Returns:
            Executor for the requested pool
            
        Raises:
            ValueError: If pool doesn't exist
        """
        if pool_name not in self._thread_pools:
            raise ValueError(f"Thread pool '{pool_name}' does not exist")
            
        return self._thread_pools[pool_name]
        
    def shutdown(self):
        """Shutdown all thread pools."""
        for pool in self._thread_pools.values():
            pool.shutdown()
```

### 3. Thread Isolation Guidelines

Best practices for thread isolation across components:

```python
class ThreadIsolationGuidelines:
    """
    Thread isolation guidelines for different execution modes.
    
    This is a static class providing guidance rather than implementation.
    """
    
    @staticmethod
    def get_isolation_level(context: ExecutionContext, component_type: str) -> str:
        """
        Get recommended isolation level for component type in context.
        
        Args:
            context: Execution context
            component_type: Type of component
            
        Returns:
            str: Recommended isolation level
        """
        # Default isolation levels by component type
        default_levels = {
            'data_handler': 'shared',
            'strategy': 'isolated',
            'risk_manager': 'shared',
            'portfolio': 'shared',
            'broker': 'shared',
            'order_manager': 'shared',
        }
        
        # Override based on execution mode
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            # All components shared in single thread
            return 'shared'
        elif context.execution_mode == ExecutionMode.OPTIMIZATION:
            # All components isolated by process
            return 'process_isolated'
        elif context.is_live:
            # Component-specific isolation in live trading
            live_overrides = {
                'data_handler': 'thread_isolated',
                'strategy': 'thread_isolated',
            }
            return live_overrides.get(component_type, default_levels.get(component_type, 'shared'))
            
        # Default to component-specific setting
        return default_levels.get(component_type, 'shared')
        
    @staticmethod
    def should_use_locks(context: ExecutionContext, component_type: str) -> bool:
        """
        Determine if component should use locks.
        
        Args:
            context: Execution context
            component_type: Type of component
            
        Returns:
            bool: Whether locks should be used
        """
        # Single-threaded backtest doesn't need locks
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            return False
            
        # Live trading always needs locks
        if context.is_live:
            return True
            
        # Optimization depends on component sharing
        if context.execution_mode == ExecutionMode.OPTIMIZATION:
            shared_components = ['data_handler']
            return component_type in shared_components
            
        # Default to using locks in multi-threaded contexts
        return context.is_multi_threaded
```

## Mode-Specific Concurrency Models

### 1. Backtesting Concurrency

Explicit concurrency models for different backtest modes:

```python
class BacktestConcurrencyModel:
    """
    Concurrency models for backtesting modes.
    
    This class defines how components interact in different backtest modes.
    """
    
    @staticmethod
    def apply_concurrency_model(context: ExecutionContext, container) -> None:
        """
        Apply appropriate concurrency model to components.
        
        Args:
            context: Execution context
            container: Component container
        """
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            # Configure for single-threaded operation
            BacktestConcurrencyModel._configure_single_threaded(container)
        elif context.execution_mode == ExecutionMode.BACKTEST_PARALLEL:
            # Configure for multi-threaded operation
            BacktestConcurrencyModel._configure_multi_threaded(container)
        elif context.execution_mode == ExecutionMode.OPTIMIZATION:
            # Configure for optimization
            BacktestConcurrencyModel._configure_optimization(container)
            
    @staticmethod
    def _configure_single_threaded(container) -> None:
        """
        Configure components for single-threaded operation.
        
        Args:
            container: Component container
        """
        # Get components
        data_handler = container.resolve('data_handler')
        strategy = container.resolve('strategy')
        event_bus = container.resolve('event_bus')
        
        # Set single-threaded mode
        if hasattr(data_handler, 'set_thread_mode'):
            data_handler.set_thread_mode(ThreadModel.SINGLE_THREADED)
            
        if hasattr(strategy, 'set_thread_mode'):
            strategy.set_thread_mode(ThreadModel.SINGLE_THREADED)
            
        if hasattr(event_bus, 'set_thread_mode'):
            event_bus.set_thread_mode(ThreadModel.SINGLE_THREADED)
            
    @staticmethod
    def _configure_multi_threaded(container) -> None:
        """
        Configure components for multi-threaded operation.
        
        Args:
            container: Component container
        """
        # Get components
        data_handler = container.resolve('data_handler')
        strategy = container.resolve('strategy')
        event_bus = container.resolve('event_bus')
        
        # Create thread pools
        thread_pool_manager = container.resolve('thread_pool_manager')
        
        # Configure data handler for parallel operation
        if hasattr(data_handler, 'set_thread_mode'):
            data_handler.set_thread_mode(ThreadModel.MULTI_THREADED)
            
        if hasattr(data_handler, 'set_executor'):
            data_handler.set_executor(
                thread_pool_manager.get_executor('data')
            )
            
        # Configure strategy for parallel operation
        if hasattr(strategy, 'set_thread_mode'):
            strategy.set_thread_mode(ThreadModel.MULTI_THREADED)
            
        if hasattr(strategy, 'set_executor'):
            strategy.set_executor(
                thread_pool_manager.get_executor('compute')
            )
            
        # Configure event bus for parallel operation
        if hasattr(event_bus, 'set_thread_mode'):
            event_bus.set_thread_mode(ThreadModel.MULTI_THREADED)
            
    @staticmethod
    def _configure_optimization(container) -> None:
        """
        Configure components for optimization.
        
        Args:
            container: Component container
        """
        # Get optimization engine
        optimization_engine = container.resolve('optimization_engine')
        
        # Create process pool
        thread_pool_manager = container.resolve('thread_pool_manager')
        
        # Configure optimization engine
        if hasattr(optimization_engine, 'set_executor'):
            optimization_engine.set_executor(
                thread_pool_manager.get_executor('optimization')
            )
```

### 2. Live Trading Concurrency

Explicit concurrency model for live trading:

```python
class LiveTradingConcurrencyModel:
    """
    Concurrency model for live trading modes.
    
    This class defines how components interact in live trading modes.
    """
    
    @staticmethod
    def apply_concurrency_model(context: ExecutionContext, container) -> None:
        """
        Apply live trading concurrency model to components.
        
        Args:
            context: Execution context
            container: Component container
        """
        # Get components
        data_handler = container.resolve('data_handler')
        strategy = container.resolve('strategy')
        broker = container.resolve('broker')
        event_bus = container.resolve('event_bus')
        
        # Create thread pools
        thread_pool_manager = container.resolve('thread_pool_manager')
        
        # Configure data handler
        if hasattr(data_handler, 'set_thread_mode'):
            data_handler.set_thread_mode(ThreadModel.MULTI_THREADED)
            
        if hasattr(data_handler, 'set_executor'):
            data_handler.set_executor(
                thread_pool_manager.get_executor('market_data')
            )
            
        # Configure strategy
        if hasattr(strategy, 'set_thread_mode'):
            strategy.set_thread_mode(ThreadModel.MULTI_THREADED)
            
        if hasattr(strategy, 'set_executor'):
            strategy.set_executor(
                thread_pool_manager.get_executor('strategy')
            )
            
        # Configure broker
        if hasattr(broker, 'set_thread_mode'):
            broker.set_thread_mode(ThreadModel.MULTI_THREADED)
            
        if hasattr(broker, 'set_executor'):
            broker.set_executor(
                thread_pool_manager.get_executor('order_processing')
            )
            
        # Configure event bus
        if hasattr(event_bus, 'set_thread_mode'):
            event_bus.set_thread_mode(ThreadModel.MULTI_THREADED)
```

## Thread Management Best Practices

### 1. Thread Creation Guidelines

Best practices for thread creation across execution modes:

| Execution Mode | Thread Creation Guidelines |
|----------------|----------------------------|
| BACKTEST_SINGLE | No thread creation permitted - all operations must be single-threaded |
| BACKTEST_PARALLEL | Use only thread pools provided by ThreadPoolManager - no direct thread creation |
| OPTIMIZATION | Each process handles a single backtest; shared data uses locking where necessary |
| LIVE_TRADING | Use dedicated threads for I/O operations; use thread pools for computation |
| PAPER_TRADING | Same as LIVE_TRADING but with simulated execution |

### 2. Thread Affinity

Guidelines for managing thread affinity:

```python
def set_thread_affinity(thread_type, cpu_ids):
    """
    Set thread affinity for specific thread types.
    
    Args:
        thread_type: Type of thread
        cpu_ids: List of CPU IDs to bind to
    """
    # Example implementation using Python's multiprocessing module
    # Actual implementation would depend on platform
    import multiprocessing
    
    # Get process ID
    pid = os.getpid()
    
    # Set affinity
    os.sched_setaffinity(pid, cpu_ids)
```

Recommended affinity settings:

| Thread Type | Recommended Affinity |
|-------------|----------------------|
| Market Data Processing | Dedicated CPU core for low-latency response |
| Strategy Computation | Multiple CPU cores for parallel computation |
| Order Processing | Dedicated CPU core for consistent latency |
| Background Tasks | Shared CPU cores for lower-priority tasks |

### 3. Thread Synchronization

Guidelines for thread synchronization across components:

```python
class ThreadSynchronizationGuidelines:
    """
    Thread synchronization guidelines for different execution modes.
    
    This is a static class providing guidance rather than implementation.
    """
    
    @staticmethod
    def get_recommended_sync_primitives(context: ExecutionContext, component_type: str) -> List[str]:
        """
        Get recommended synchronization primitives for component in context.
        
        Args:
            context: Execution context
            component_type: Type of component
            
        Returns:
            list: Recommended synchronization primitives
        """
        # Single-threaded mode needs no synchronization
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            return ['none']
            
        # Default recommendations by component type
        default_primitives = {
            'data_handler': ['lock', 'thread_local'],
            'strategy': ['lock'],
            'risk_manager': ['lock'],
            'portfolio': ['lock'],
            'broker': ['lock', 'event'],
            'order_manager': ['lock', 'queue'],
        }
        
        # Live trading may need additional synchronization
        if context.is_live:
            live_overrides = {
                'data_handler': ['lock', 'thread_local', 'queue'],
                'broker': ['lock', 'event', 'queue'],
            }
            return live_overrides.get(component_type, default_primitives.get(component_type, ['lock']))
            
        # Return default recommendations
        return default_primitives.get(component_type, ['lock'])
        
    @staticmethod
    def get_lock_strategy(context: ExecutionContext, component_type: str) -> str:
        """
        Get recommended locking strategy for component in context.
        
        Args:
            context: Execution context
            component_type: Type of component
            
        Returns:
            str: Recommended locking strategy
        """
        # Component-specific recommendations
        if component_type == 'data_handler':
            return 'reader_writer_lock' if context.is_multi_threaded else 'none'
        elif component_type == 'portfolio':
            return 'fine_grained_lock' if context.is_multi_threaded else 'none'
        elif component_type == 'order_manager':
            return 'fine_grained_lock' if context.is_multi_threaded else 'none'
            
        # Default recommendations
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            return 'none'
        elif context.is_multi_threaded:
            return 'reentrant_lock'
        else:
            return 'none'
```

## Configuration Examples

### 1. Single-Threaded Backtest Configuration

```yaml
system:
  name: ADMF-Trader
  
execution:
  mode: BACKTEST_SINGLE
  thread_model: SINGLE_THREADED
  
backtest:
  start_date: 2022-01-01
  end_date: 2022-12-31
  symbols: [SPY, AAPL, MSFT]
  
components:
  data_handler:
    class: HistoricalDataHandler
    thread_safe: false  # Disable thread safety for performance
    
  strategy:
    class: MovingAverageCrossover
    thread_safe: false  # Disable thread safety for performance
```

### 2. Multi-Threaded Backtest Configuration

```yaml
system:
  name: ADMF-Trader
  
execution:
  mode: BACKTEST_PARALLEL
  thread_model: MULTI_THREADED
  thread_pools:
    data:
      max_workers: 4
    compute:
      max_workers: 4
  
backtest:
  start_date: 2022-01-01
  end_date: 2022-12-31
  symbols: [SPY, AAPL, MSFT, GOOG, AMZN, FB, TSLA, NVDA, JPM, JNJ]
  
components:
  data_handler:
    class: ParallelDataHandler
    thread_safe: true
    
  strategy:
    class: MovingAverageCrossover
    thread_safe: true
```

### 3. Optimization Configuration

```yaml
system:
  name: ADMF-Trader
  
execution:
  mode: OPTIMIZATION
  thread_model: PROCESS_PARALLEL
  process_pools:
    optimization:
      max_workers: 7  # One less than CPU count for system responsiveness
  
optimization:
  parameter_space:
    fast_ma: [5, 10, 15, 20]
    slow_ma: [30, 50, 100, 200]
    stop_loss: [0.01, 0.02, 0.03]
  
backtest:
  start_date: 2022-01-01
  end_date: 2022-12-31
  symbols: [SPY]
  
components:
  data_handler:
    class: HistoricalDataHandler
    thread_safe: true
    
  strategy:
    class: MovingAverageCrossover
    thread_safe: true
```

### 4. Live Trading Configuration

```yaml
system:
  name: ADMF-Trader
  
execution:
  mode: LIVE_TRADING
  thread_model: MULTI_THREADED
  thread_pools:
    market_data:
      max_workers: 2
    order_processing:
      max_workers: 2
    strategy:
      max_workers: 2
  
live_trading:
  symbols: [SPY, AAPL, MSFT]
  
components:
  data_handler:
    class: LiveMarketDataHandler
    thread_safe: true
    
  strategy:
    class: MovingAverageCrossover
    thread_safe: true
    
  broker:
    class: LiveBrokerHandler
    thread_safe: true
```

## Implementation Strategy

### 1. Core Implementation

1. Implement `ExecutionMode` and `ThreadModel` enumerations
2. Implement `ExecutionContext` class
3. Update system bootstrap to use execution context

### 2. Thread Management

1. Implement `ThreadPoolManager` class
2. Create thread isolation guidelines
3. Implement thread affinity management

### 3. Mode-Specific Components

1. Implement single-threaded backtest coordinator
2. Implement multi-threaded backtest coordinator
3. Implement optimization engine with process parallelism
4. Implement live trading system with thread pools

### 4. Testing

1. Create mode-specific test suites
2. Implement thread safety validation tests
3. Create performance benchmark tests for different modes

## Best Practices

### 1. Component Design

- Make all components mode-aware through the `ExecutionContext`
- Design components to adapt their behavior based on execution mode
- Document thread safety requirements for each component
- Use factory methods to create mode-appropriate instances

### 2. Thread Safety

- Use thread-safe collections from `ThreadSafetyFactory` based on execution mode
- Document thread safety guarantees for all public APIs
- Implement fine-grained locking where appropriate
- Use atomic operations for simple state changes

### 3. Performance Optimization

- Disable thread safety in single-threaded contexts
- Use thread pools instead of creating individual threads
- Implement batch processing where appropriate
- Minimize lock contention through careful design

### 4. Error Handling

- Handle thread interruption appropriately
- Implement thread-safe error reporting
- Use timeouts to prevent deadlocks
- Implement graceful shutdown for all thread pools

## Asynchronous Architecture Support

For more detailed information on the asynchronous implementation, please see the [ASYNCHRONOUS_ARCHITECTURE.md](/Users/daws/ADMF/docs/core/ASYNCHRONOUS_ARCHITECTURE.md) document. This document provides:

1. Comprehensive async component interfaces
2. Event loop management strategies
3. Async-specific thread safety guidelines
4. Implementation patterns for async components
5. Examples of async implementations for key system components

### Hybrid Execution Model

The ADMF-Trader system supports a hybrid execution model that combines the strengths of both synchronous and asynchronous paradigms:

| Execution Context | Primary Paradigm | Benefits |
|-------------------|------------------|----------|
| Backtest Single   | Synchronous      | Simplicity, performance for single-threaded operation |
| Backtest Parallel | Multi-threaded   | Parallel processing with thread pool |
| Optimization      | Process-parallel | Maximum CPU utilization across cores |
| Live Trading      | Asynchronous     | Non-blocking I/O, efficient resource utilization |
| Paper Trading     | Asynchronous     | Same model as live trading for accurate simulation |

The system automatically selects the appropriate execution paradigm based on the execution mode but allows for explicit configuration when needed.

## Conclusion

This execution mode clarity framework provides explicit models for backtesting and live trading, with clear concurrency assumptions for each execution mode. By following these guidelines, developers can create components that adapt their behavior based on execution context, ensuring consistent thread management and optimal performance across all operating modes.

The framework enables:

1. Clear distinction between different execution modes
2. Explicit thread management tailored to each mode
3. Performance optimization through context-aware thread safety
4. Consistent concurrency assumptions across components
5. Guidelines for thread creation and synchronization
6. Seamless integration of asynchronous programming model

By implementing this framework, the ADMF-Trader system will achieve improved reliability, performance, and maintainability across all execution scenarios while providing robust support for both synchronous and asynchronous execution models.