# Asynchronous Architecture

## Overview

This document outlines the asynchronous architecture for the ADMF-Trader system, focusing on handling non-blocking operations, event-driven processing, and efficient resource utilization in a live trading environment. The system is designed to support both synchronous operation (for backtesting) and asynchronous operation (for live trading) through a unified programming model.

## Motivation

Live trading systems require efficient handling of concurrent operations:

1. **Multiple Data Sources**: Processing market data from different sources simultaneously
2. **Non-blocking I/O**: Handling network requests to exchanges/brokers without blocking
3. **Responsive GUI**: Maintaining UI responsiveness while processing data and orders
4. **Resource Efficiency**: Optimizing CPU and memory usage during idle periods
5. **Handling Timeouts**: Managing operations with time constraints effectively

An asynchronous architecture addresses these requirements by allowing concurrent operations without the overhead of thread management.

## Core Async Components

### 1. Async Component Base

```python
from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Any, Optional, Coroutine

class AsyncComponentBase(ABC):
    """Base class for all async components in the system."""
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self._initialized = False
        self._running = False
        self._lock = asyncio.Lock()  # Async lock for state changes
    
    @abstractmethod
    async def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize component asynchronously.
        
        Args:
            context: Dependency context containing required components
        """
        async with self._lock:
            self._initialized = True
    
    @abstractmethod
    async def start(self) -> None:
        """Begin component operation."""
        if not self._initialized:
            raise RuntimeError(f"Component {self.name} must be initialized before starting")
        
        async with self._lock:
            self._running = True
    
    @abstractmethod
    async def stop(self) -> None:
        """End component operation."""
        async with self._lock:
            self._running = False
    
    @abstractmethod
    async def reset(self) -> None:
        """Clear component state for a new run."""
        pass
    
    @abstractmethod
    async def teardown(self) -> None:
        """Release resources."""
        async with self._lock:
            self._initialized = False
            self._running = False
    
    @property
    def name(self) -> str:
        """Get component name."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set component name."""
        self._name = value
    
    @property
    def initialized(self) -> bool:
        """Whether component is initialized."""
        return self._initialized
    
    @property
    def running(self) -> bool:
        """Whether component is running."""
        return self._running
```

### 2. Async Event Bus

```python
class AsyncEventBus:
    """
    Asynchronous event bus for publishing and subscribing to events.
    
    Supports both sync and async event handlers.
    """
    
    def __init__(self, context_id: str = "default"):
        self.context_id = context_id
        self.subscribers = {}  # event_type -> list of handlers
        self._lock = asyncio.Lock()
    
    async def publish(self, event: Dict[str, Any]) -> bool:
        """
        Publish an event asynchronously.
        
        Args:
            event: Event dictionary with at least 'type' field
            
        Returns:
            bool: Whether the event was successfully published
        """
        event_type = event.get('type')
        if not event_type:
            return False
            
        # Add context ID and timestamp if not present
        if 'context_id' not in event:
            event['context_id'] = self.context_id
            
        if 'timestamp' not in event:
            event['timestamp'] = datetime.now()
            
        # Get subscribers for this event type
        handlers = []
        async with self._lock:
            handlers = self.subscribers.get(event_type, []).copy()
            
        # Call each handler
        results = []
        for handler_info in handlers:
            handler = handler_info['handler']
            is_async = handler_info['is_async']
            
            try:
                if is_async:
                    # Async handler
                    result = await handler(event)
                else:
                    # Sync handler - run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: handler(event))
                    
                results.append(result)
            except Exception as e:
                # Log but don't crash on handler exception
                print(f"Error in event handler: {str(e)}")
                
        return len(results) > 0
    
    async def subscribe(self, event_type: str, handler, is_async: bool = False) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when events occur
            is_async: Whether the handler is an async function
        """
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
                
            self.subscribers[event_type].append({
                'handler': handler,
                'is_async': is_async
            })
    
    async def unsubscribe(self, event_type: str, handler) -> bool:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to unsubscribe
            
        Returns:
            bool: Whether handler was successfully unsubscribed
        """
        async with self._lock:
            if event_type not in self.subscribers:
                return False
                
            # Find and remove handler
            for i, handler_info in enumerate(self.subscribers[event_type]):
                if handler_info['handler'] == handler:
                    self.subscribers[event_type].pop(i)
                    return True
                    
        return False
    
    async def unsubscribe_all(self, handler) -> None:
        """
        Unsubscribe a handler from all event types.
        
        Args:
            handler: Handler to unsubscribe
        """
        async with self._lock:
            for event_type in self.subscribers:
                self.subscribers[event_type] = [
                    h for h in self.subscribers[event_type] 
                    if h['handler'] != handler
                ]
                
    def sync_publish(self, event: Dict[str, Any]) -> bool:
        """
        Synchronous version of publish for compatibility.
        
        Args:
            event: Event dictionary
            
        Returns:
            bool: Whether the event was successfully published
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create task if loop is already running
            task = loop.create_task(self.publish(event))
            # We can't wait for result here, so return True
            return True
        else:
            # Run the async method in the loop
            return loop.run_until_complete(self.publish(event))
    
    def sync_subscribe(self, event_type: str, handler, is_async: bool = False) -> None:
        """Synchronous version of subscribe for compatibility."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.subscribe(event_type, handler, is_async))
        else:
            loop.run_until_complete(self.subscribe(event_type, handler, is_async))
```

## Execution Model

### 1. Event Loop Management

The system uses different event loop strategies based on the execution mode:

#### Backtesting Mode
```python
def run_backtest(strategy, data, **kwargs):
    """Run backtest in synchronous mode."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(_run_backtest_async(strategy, data, **kwargs))
    finally:
        loop.close()
        
async def _run_backtest_async(strategy, data, **kwargs):
    """Internal async implementation of backtest."""
    # Initialize components
    event_bus = AsyncEventBus(context_id="backtest")
    await strategy.initialize({"event_bus": event_bus})
    await strategy.start()
    
    # Process each bar synchronously
    try:
        for bar in data:
            # Create and publish bar event
            bar_event = {"type": "BAR", "data": bar}
            await event_bus.publish(bar_event)
            
        # Wait for any pending tasks
        await asyncio.sleep(0)
    finally:
        await strategy.stop()
        await strategy.teardown()
```

#### Live Trading Mode
```python
def run_live_trading(strategy, data_feed, **kwargs):
    """Run live trading in fully async mode."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Set up signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(loop)))
    
    try:
        # Start components and run forever
        loop.run_until_complete(_setup_live_trading(strategy, data_feed, **kwargs))
        loop.run_forever()
    finally:
        loop.close()
        
async def _setup_live_trading(strategy, data_feed, **kwargs):
    """Internal async setup for live trading."""
    # Initialize components
    event_bus = AsyncEventBus(context_id="live")
    
    global components  # For shutdown
    components = {
        "strategy": strategy,
        "data_feed": data_feed,
        "event_bus": event_bus
    }
    
    # Initialize and start
    await strategy.initialize({"event_bus": event_bus})
    await data_feed.initialize({"event_bus": event_bus})
    
    await strategy.start()
    await data_feed.start()
    
async def shutdown(loop):
    """Graceful shutdown of all components."""
    for name, component in components.items():
        try:
            await component.stop()
            await component.teardown()
        except Exception as e:
            print(f"Error shutting down {name}: {str(e)}")
            
    loop.stop()
```

### 2. Dual-Mode Components

Components can implement both sync and async interfaces for maximum compatibility:

```python
class DualModeComponent:
    """Component that supports both sync and async operation."""
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self._initialized = False
        self._running = False
        
    # Sync interface
    def initialize(self, context):
        """Synchronous initialize."""
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.initialize_async(context))
        except RuntimeError:
            # New loop if none exists
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.initialize_async(context))
            loop.close()
            
    # Async interface
    async def initialize_async(self, context):
        """Asynchronous initialize implementation."""
        self._initialized = True
        
    # Similar dual implementations for other methods...
```

## Thread Safety for Async Code

### 1. Async-Safe Collections

```python
class AsyncSafeDict:
    """Dictionary that's safe for async access."""
    
    def __init__(self):
        self._data = {}
        self._lock = asyncio.Lock()
        
    async def get(self, key, default=None):
        """Get a value asynchronously."""
        async with self._lock:
            return self._data.get(key, default)
            
    async def set(self, key, value):
        """Set a value asynchronously."""
        async with self._lock:
            self._data[key] = value
            
    # Sync compatibility methods
    def sync_get(self, key, default=None):
        """Sync version of get."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't wait for result in running loop
            # Return current value without lock (may be stale)
            return self._data.get(key, default)
        else:
            return loop.run_until_complete(self.get(key, default))
```

### 2. Thread/Task Coordination

```python
class AsyncEventBarrier:
    """Barrier that multiple tasks can wait on."""
    
    def __init__(self, parties):
        self.parties = parties
        self.waiting = 0
        self.lock = asyncio.Lock()
        self.barrier = asyncio.Condition(self.lock)
        
    async def wait(self):
        """Wait for all parties to reach the barrier."""
        async with self.lock:
            self.waiting += 1
            if self.waiting == self.parties:
                # Last task to arrive
                self.waiting = 0
                # Notify all waiting tasks
                await self.barrier.notify_all()
                return True
            else:
                # Wait for the last task
                await self.barrier.wait()
                return False
```

## Module-Specific Async Patterns

### 1. Data Handler

```python
class AsyncDataHandler(AsyncComponentBase):
    """Asynchronous data handler implementation."""
    
    def __init__(self, name="async_data_handler", parameters=None):
        super().__init__(name, parameters)
        self.data_sources = []
        self._data_tasks = []
        
    async def initialize(self, context):
        """Initialize async data handler."""
        await super().initialize(context)
        
        # Initialize data sources
        for source in self.data_sources:
            await source.initialize(context)
            
    async def start(self):
        """Start data processing."""
        await super().start()
        
        # Start data source tasks
        for source in self.data_sources:
            task = asyncio.create_task(self._process_source(source))
            self._data_tasks.append(task)
            
    async def _process_source(self, source):
        """Process data from a source continuously."""
        await source.start()
        
        try:
            while self.running:
                data = await source.get_next_data()
                if data:
                    # Create and emit bar event
                    event = {
                        "type": "BAR",
                        "data": data,
                        "source": source.name
                    }
                    await self.event_bus.publish(event)
                else:
                    # No more data or timeout
                    await asyncio.sleep(0.001)
        finally:
            await source.stop()
            
    async def stop(self):
        """Stop data processing."""
        # Cancel all data tasks
        for task in self._data_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        if self._data_tasks:
            await asyncio.wait(self._data_tasks, return_when=asyncio.ALL_COMPLETED)
            
        self._data_tasks = []
        await super().stop()
        
    async def get_latest_bar(self, symbol):
        """Get the latest bar for a symbol asynchronously."""
        # Implementation details...
        pass
```

### 2. Strategy

```python
class AsyncStrategy(AsyncComponentBase):
    """Asynchronous strategy implementation."""
    
    def __init__(self, name="async_strategy", parameters=None):
        super().__init__(name, parameters)
        self.indicators = {}
        
    async def initialize(self, context):
        """Initialize async strategy."""
        await super().initialize(context)
        
        # Get event bus from context
        self.event_bus = context.get("event_bus")
        if not self.event_bus:
            raise ValueError("Event bus not provided in context")
            
        # Subscribe to events
        await self.event_bus.subscribe("BAR", self.on_bar, is_async=True)
        
    async def on_bar(self, event):
        """Process bar event asynchronously."""
        bar_data = event.get("data", {})
        
        # Update indicators
        await self._update_indicators(bar_data)
        
        # Calculate signals
        signals = await self._calculate_signals(bar_data)
        
        # Emit signals
        for signal in signals:
            signal_event = {
                "type": "SIGNAL",
                "data": signal,
                "strategy": self.name
            }
            await self.event_bus.publish(signal_event)
            
    async def _update_indicators(self, bar_data):
        """Update indicators with new data asynchronously."""
        # Implementation details...
        pass
        
    async def _calculate_signals(self, bar_data):
        """Calculate signals based on indicators asynchronously."""
        # Implementation details...
        return []
```

### 3. Broker

```python
class AsyncBroker(AsyncComponentBase):
    """Asynchronous broker implementation."""
    
    def __init__(self, name="async_broker", parameters=None):
        super().__init__(name, parameters)
        self.orders = {}
        self.api_client = None
        
    async def initialize(self, context):
        """Initialize async broker."""
        await super().initialize(context)
        
        # Get event bus from context
        self.event_bus = context.get("event_bus")
        if not self.event_bus:
            raise ValueError("Event bus not provided in context")
            
        # Initialize API client
        api_config = self.parameters.get("api_config", {})
        self.api_client = await self._create_api_client(api_config)
        
        # Subscribe to events
        await self.event_bus.subscribe("ORDER", self.on_order, is_async=True)
        
    async def _create_api_client(self, config):
        """Create API client asynchronously."""
        # Implementation specific to broker API
        pass
        
    async def on_order(self, event):
        """Process order event asynchronously."""
        order_data = event.get("data", {})
        
        # Place order with broker
        try:
            result = await self.api_client.place_order(order_data)
            
            # Create fill event on success
            if result.get("status") == "filled":
                fill_event = {
                    "type": "FILL",
                    "data": {
                        "order_id": result.get("order_id"),
                        "fill_price": result.get("fill_price"),
                        "fill_quantity": result.get("fill_quantity"),
                        "timestamp": result.get("timestamp")
                    }
                }
                await self.event_bus.publish(fill_event)
                
        except Exception as e:
            # Handle order placement errors
            error_event = {
                "type": "ERROR",
                "data": {
                    "error_type": "order_placement",
                    "error_message": str(e),
                    "order_data": order_data
                }
            }
            await self.event_bus.publish(error_event)
```

## Compatibility With Existing Code

### 1. Async-Sync Bridge

```python
class AsyncSyncBridge:
    """Bridge between async and sync components."""
    
    def __init__(self, async_event_bus):
        self.async_event_bus = async_event_bus
        
    def create_sync_event_bus(self):
        """Create a synchronous API around the async event bus."""
        return SyncEventBusAdapter(self.async_event_bus)
        
class SyncEventBusAdapter:
    """Synchronous adapter for async event bus."""
    
    def __init__(self, async_event_bus):
        self.async_bus = async_event_bus
        
    def publish(self, event):
        """Synchronous publish method."""
        return self.async_bus.sync_publish(event)
        
    def subscribe(self, event_type, handler):
        """Synchronous subscribe method."""
        return self.async_bus.sync_subscribe(event_type, handler)
```

### 2. Execution Context Detection

```python
def is_async_context():
    """Detect if running in async context."""
    try:
        loop = asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
        
def get_or_create_loop():
    """Get current loop or create a new one if none is running."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()
```

## Guidelines for Async Implementation

1. **Use Async/Await Consistently**
   - Once a function is async, all callers should be async
   - Provide sync wrappers only at API boundaries

2. **Handle Cancellation Gracefully**
   - Check for `asyncio.CancelledError` in long-running tasks
   - Implement proper cleanup in `finally` blocks

3. **Avoid Blocking Operations**
   - Use `run_in_executor` for CPU-bound or blocking operations
   - Consider thread pools for heavy computations

4. **Manage Resources Properly**
   - Use async context managers (`async with`) for resource management
   - Ensure proper task cleanup on shutdown

5. **Error Handling**
   - Use structured exception handling with async/await
   - Propagate errors appropriately or convert to events

6. **Testing Async Code**
   - Use `asyncio.run()` for simple test cases
   - Consider `pytest-asyncio` for comprehensive testing

## Conclusion

The asynchronous architecture provides a foundation for building a high-performance trading system that can handle concurrent operations efficiently. By supporting both synchronous and asynchronous execution modes, the system maintains backward compatibility while enabling more efficient operation for live trading.

The architecture follows these key principles:
1. **Event-Driven Design**: All system components communicate through events
2. **Non-Blocking I/O**: Network and disk operations don't block the main thread
3. **Resource Efficiency**: CPU and memory are used efficiently
4. **Graceful Degradation**: System responds appropriately under load
5. **Dual-Mode Operation**: Components work in both sync and async contexts

This design provides a solid foundation for implementing a trading system that can scale from backtesting to live trading with minimal code changes.