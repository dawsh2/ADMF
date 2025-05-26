# Event System Scalability

## Overview

This document defines the design patterns and architecture for ensuring the ADMF-Trader event system scales effectively for high-frequency trading scenarios. It covers lock-free designs, partitioning strategies, batching mechanisms, and prioritization approaches to optimize event throughput and latency.

## Motivation

As trading frequency and data volume increase, the event system can become a bottleneck. High-frequency trading scenarios may generate thousands of events per second, requiring:

1. **Minimized Latency**: Events must be processed with microsecond-level latency
2. **High Throughput**: The system must handle bursts of thousands of events
3. **Efficient Resource Usage**: CPU and memory overhead must be minimized
4. **Predictable Performance**: Event processing times should have low variance
5. **Prioritization**: Critical events (orders, fills) must take precedence over less time-sensitive events

## Architecture

### 1. Event Bus Partitioning

The scalable event bus uses domain-based partitioning to distribute event processing:

```python
class PartitionedEventBus:
    """Partitioned event bus for high-frequency trading."""
    
    def __init__(self, partition_strategy="domain"):
        """
        Initialize partitioned event bus.
        
        Args:
            partition_strategy: Strategy for partitioning events
        """
        self.partitions = {
            "market_data": EventBusPartition("market_data"),
            "signal": EventBusPartition("signal"),
            "order": EventBusPartition("order"),
            "fill": EventBusPartition("fill"),
            "system": EventBusPartition("system")
        }
        
        # Define critical partitions
        self.critical_partitions = ["order", "fill"]
        
        # Initialize partition threads
        self._initialize_partitions()
        
    def _initialize_partitions(self):
        """Initialize partition threads."""
        for name, partition in self.partitions.items():
            # Critical partitions get higher priority
            is_critical = name in self.critical_partitions
            partition.initialize(is_critical)
            
    async def publish(self, event):
        """
        Publish event to appropriate partition.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: Success status
        """
        # Determine partition
        event_type = event.get("type", "")
        
        if event_type.startswith("MARKET_DATA"):
            partition = "market_data"
        elif event_type.startswith("SIGNAL"):
            partition = "signal"
        elif event_type.startswith("ORDER"):
            partition = "order"
        elif event_type.startswith("FILL"):
            partition = "fill"
        else:
            partition = "system"
            
        # Publish to partition
        return await self.partitions[partition].publish(event)
        
    async def subscribe(self, event_type, handler, is_async=True):
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Event handler function
            is_async: Whether handler is async
            
        Returns:
            Subscription ID
        """
        # Determine partition
        if event_type.startswith("MARKET_DATA"):
            partition = "market_data"
        elif event_type.startswith("SIGNAL"):
            partition = "signal"
        elif event_type.startswith("ORDER"):
            partition = "order"
        elif event_type.startswith("FILL"):
            partition = "fill"
        else:
            partition = "system"
            
        # Subscribe to partition
        return await self.partitions[partition].subscribe(event_type, handler, is_async)
        
    async def unsubscribe(self, subscription_id):
        """
        Unsubscribe from event.
        
        Args:
            subscription_id: Subscription ID from subscribe
            
        Returns:
            bool: Success status
        """
        # Extract partition from subscription ID
        partition_name, local_id = subscription_id.split(":", 1)
        
        # Unsubscribe from partition
        if partition_name in self.partitions:
            return await self.partitions[partition_name].unsubscribe(local_id)
        
        return False
```

### 2. Event Partition Implementation

Each partition is processed independently to maximize throughput:

```python
import asyncio
import threading
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor

class EventBusPartition:
    """Individual partition of the event bus."""
    
    def __init__(self, name):
        """
        Initialize event bus partition.
        
        Args:
            name: Partition name
        """
        self.name = name
        self.subscribers = {}  # event_type -> list of handlers
        self.subscription_map = {}  # subscription_id -> (event_type, handler)
        self._event_queue = asyncio.Queue()
        self._running = False
        self._task = None
        self._lock = asyncio.Lock()
        self._executor = None
        
    def initialize(self, is_critical=False):
        """
        Initialize partition.
        
        Args:
            is_critical: Whether this is a critical partition
        """
        self._running = True
        
        # Create thread pool
        max_workers = 2 if is_critical else 1
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"event_{self.name}_"
        )
        
        # Start event processing task
        self._task = asyncio.create_task(self._process_events())
        
    async def _process_events(self):
        """Process events from queue."""
        while self._running:
            try:
                # Get next event
                event = await self._event_queue.get()
                
                # Get handlers for this event type
                event_type = event.get("type")
                async with self._lock:
                    handlers = self.subscribers.get(event_type, []).copy()
                
                # Process event with all handlers
                for handler_info in handlers:
                    handler = handler_info["handler"]
                    is_async = handler_info["is_async"]
                    
                    try:
                        if is_async:
                            # Async handler
                            await handler(event)
                        else:
                            # Sync handler - run in executor
                            await asyncio.get_event_loop().run_in_executor(
                                self._executor, 
                                lambda: handler(event)
                            )
                    except Exception as e:
                        # Log but don't crash
                        print(f"Error in event handler: {str(e)}")
                        
                # Mark task done
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but don't crash
                print(f"Error in event processing: {str(e)}")
                
    async def publish(self, event):
        """
        Publish event to this partition.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: Success status
        """
        if not self._running:
            return False
            
        # Add partition info to event
        event["partition"] = self.name
        
        # Add to queue
        await self._event_queue.put(event)
        return True
        
    async def subscribe(self, event_type, handler, is_async=True):
        """
        Subscribe to event type in this partition.
        
        Args:
            event_type: Event type to subscribe to
            handler: Event handler function
            is_async: Whether handler is async
            
        Returns:
            str: Subscription ID
        """
        subscription_id = f"{self.name}:{str(uuid.uuid4())}"
        
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
                
            self.subscribers[event_type].append({
                "handler": handler,
                "is_async": is_async
            })
            
            self.subscription_map[subscription_id] = (event_type, handler)
            
        return subscription_id
        
    async def unsubscribe(self, subscription_id):
        """
        Unsubscribe from this partition.
        
        Args:
            subscription_id: Local subscription ID
            
        Returns:
            bool: Success status
        """
        if subscription_id not in self.subscription_map:
            return False
            
        event_type, handler = self.subscription_map[subscription_id]
        
        async with self._lock:
            if event_type in self.subscribers:
                self.subscribers[event_type] = [
                    h for h in self.subscribers[event_type]
                    if h["handler"] != handler
                ]
                
            # Remove from subscription map
            del self.subscription_map[subscription_id]
            
        return True
```

### 3. Lock-Free Event Queue

For maximum performance, a lock-free queue implementation can be used:

```python
class LockFreeEventQueue:
    """Lock-free event queue for high-performance scenarios."""
    
    def __init__(self, capacity=10000):
        """
        Initialize lock-free queue.
        
        Args:
            capacity: Maximum queue capacity
        """
        self._capacity = capacity
        self._queue = [None] * capacity
        self._head = 0
        self._tail = 0
        
        # Atomic variables for head and tail
        # Note: This uses Python's atomic primitives but would be implemented
        # with platform-specific atomics in production
        self._head_lock = threading.Lock()
        self._tail_lock = threading.Lock()
        
    def enqueue(self, event):
        """
        Add event to queue.
        
        Args:
            event: Event to enqueue
            
        Returns:
            bool: Success status
        """
        # Get current tail
        with self._tail_lock:
            tail = self._tail
            new_tail = (tail + 1) % self._capacity
            
            # Check if queue is full
            if new_tail == self._head:
                return False
                
            # Add event to queue
            self._queue[tail] = event
            self._tail = new_tail
            
        return True
        
    def dequeue(self):
        """
        Remove event from queue.
        
        Returns:
            Event or None if queue is empty
        """
        # Get current head
        with self._head_lock:
            if self._head == self._tail:
                return None
                
            head = self._head
            event = self._queue[head]
            self._queue[head] = None  # Clear reference
            self._head = (head + 1) % self._capacity
            
        return event
        
    def size(self):
        """
        Get current queue size.
        
        Returns:
            int: Queue size
        """
        # This is approximate due to concurrency
        head = self._head
        tail = self._tail
        
        if tail >= head:
            return tail - head
        else:
            return self._capacity - (head - tail)
            
    def is_empty(self):
        """
        Check if queue is empty.
        
        Returns:
            bool: True if empty
        """
        return self._head == self._tail
```

### 4. Event Batching

For high-volume scenarios, event batching improves throughput:

```python
class EventBatcher:
    """Batches events for improved throughput."""
    
    def __init__(self, max_batch_size=100, max_wait_time=0.01):
        """
        Initialize event batcher.
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time in seconds
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batches = {}  # event_type -> batch
        self._lock = asyncio.Lock()
        self._timers = {}  # event_type -> timer
        
    async def add_event(self, event, destination):
        """
        Add event to batch.
        
        Args:
            event: Event to add
            destination: Function to call with batch
            
        Returns:
            bool: Whether event was immediately processed
        """
        event_type = event.get("type")
        
        async with self._lock:
            # Create batch if not exists
            if event_type not in self.batches:
                self.batches[event_type] = []
                
                # Start timer for this batch
                self._timers[event_type] = asyncio.create_task(
                    self._flush_after_timeout(event_type, destination)
                )
                
            # Add to batch
            self.batches[event_type].append(event)
            
            # Check if batch is full
            if len(self.batches[event_type]) >= self.max_batch_size:
                # Cancel timer
                if self._timers[event_type]:
                    self._timers[event_type].cancel()
                    self._timers[event_type] = None
                    
                # Flush batch
                batch = self.batches[event_type]
                self.batches[event_type] = []
                
                # Process batch
                asyncio.create_task(destination(batch))
                return True
                
        return False
        
    async def _flush_after_timeout(self, event_type, destination):
        """
        Flush batch after timeout.
        
        Args:
            event_type: Event type to flush
            destination: Function to call with batch
        """
        try:
            # Wait for timeout
            await asyncio.sleep(self.max_wait_time)
            
            # Flush batch
            async with self._lock:
                if event_type in self.batches and self.batches[event_type]:
                    batch = self.batches[event_type]
                    self.batches[event_type] = []
                    self._timers[event_type] = None
                    
                    # Process batch
                    await destination(batch)
        except asyncio.CancelledError:
            # Timer was cancelled, batch will be processed elsewhere
            pass
```

### 5. Event Prioritization

Implementing prioritization for critical events:

```python
class PriorityEventQueue:
    """Event queue with priority levels."""
    
    def __init__(self):
        """Initialize priority event queue."""
        # Multiple queues for different priorities
        self._queues = {
            "high": asyncio.PriorityQueue(),
            "medium": asyncio.PriorityQueue(),
            "low": asyncio.PriorityQueue()
        }
        self._running = False
        self._task = None
        
    def start(self):
        """Start priority queue processing."""
        self._running = True
        self._task = asyncio.create_task(self._process_queues())
        
    def stop(self):
        """Stop priority queue processing."""
        self._running = False
        if self._task:
            self._task.cancel()
            
    async def enqueue(self, event, priority="medium"):
        """
        Add event to priority queue.
        
        Args:
            event: Event to enqueue
            priority: Priority level (high, medium, low)
            
        Returns:
            bool: Success status
        """
        if priority not in self._queues:
            priority = "medium"
            
        # Add timestamp to event
        event["enqueue_time"] = asyncio.get_event_loop().time()
        
        # Add to appropriate queue
        await self._queues[priority].put(event)
        return True
        
    async def _process_queues(self):
        """Process events from priority queues."""
        while self._running:
            try:
                # Check high priority queue first
                if not self._queues["high"].empty():
                    event = await self._queues["high"].get()
                    await self._process_event(event)
                    self._queues["high"].task_done()
                    continue
                    
                # Check medium priority queue
                if not self._queues["medium"].empty():
                    event = await self._queues["medium"].get()
                    await self._process_event(event)
                    self._queues["medium"].task_done()
                    continue
                    
                # Check low priority queue
                if not self._queues["low"].empty():
                    event = await self._queues["low"].get()
                    await self._process_event(event)
                    self._queues["low"].task_done()
                    continue
                    
                # No events, wait a bit
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break
                
    async def _process_event(self, event):
        """
        Process a single event.
        
        Args:
            event: Event to process
        """
        # Calculate latency
        latency = asyncio.get_event_loop().time() - event["enqueue_time"]
        event["queue_latency"] = latency
        
        # Process event (implementation-specific)
        # ...
```

## Benchmark-Driven Scaling

The event system should adapt based on observed performance metrics:

### 1. Event Throughput Benchmarking

```python
async def benchmark_event_throughput(event_bus, event_types, duration=10):
    """
    Benchmark event throughput.
    
    Args:
        event_bus: Event bus to benchmark
        event_types: List of event types to benchmark
        duration: Benchmark duration in seconds
        
    Returns:
        dict: Benchmark results
    """
    results = {}
    
    for event_type in event_types:
        # Create counters
        counter = {"count": 0}
        
        # Create handler
        async def handler(event):
            counter["count"] += 1
            
        # Subscribe to event
        await event_bus.subscribe(event_type, handler, is_async=True)
        
        # Create publisher
        async def publisher():
            start_time = asyncio.get_event_loop().time()
            publish_count = 0
            
            while asyncio.get_event_loop().time() - start_time < duration:
                # Create and publish event
                event = {
                    "type": event_type,
                    "data": {"value": publish_count},
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                await event_bus.publish(event)
                publish_count += 1
                
                # Short delay to prevent overwhelming
                await asyncio.sleep(0.0001)
                
            return publish_count
            
        # Run publisher
        publish_count = await publisher()
        
        # Calculate results
        results[event_type] = {
            "published": publish_count,
            "received": counter["count"],
            "loss_rate": (publish_count - counter["count"]) / publish_count if publish_count > 0 else 0,
            "throughput": counter["count"] / duration
        }
        
    return results
```

### 2. Event Latency Benchmarking

```python
async def benchmark_event_latency(event_bus, event_types, event_count=1000):
    """
    Benchmark event latency.
    
    Args:
        event_bus: Event bus to benchmark
        event_types: List of event types to benchmark
        event_count: Number of events to publish
        
    Returns:
        dict: Benchmark results
    """
    results = {}
    
    for event_type in event_types:
        # Create latency tracker
        latencies = []
        completion_event = asyncio.Event()
        
        # Create handler
        async def handler(event):
            # Calculate latency
            now = asyncio.get_event_loop().time()
            latency = now - event["timestamp"]
            latencies.append(latency)
            
            # Check if all events received
            if len(latencies) >= event_count:
                completion_event.set()
                
        # Subscribe to event
        await event_bus.subscribe(event_type, handler, is_async=True)
        
        # Create publisher
        async def publisher():
            for i in range(event_count):
                # Create and publish event
                event = {
                    "type": event_type,
                    "data": {"value": i},
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                await event_bus.publish(event)
                
                # Short delay to prevent overwhelming
                await asyncio.sleep(0.001)
                
        # Run publisher
        publisher_task = asyncio.create_task(publisher())
        
        # Wait for completion
        await asyncio.wait([completion_event.wait()], timeout=30)
        
        # Calculate results
        if latencies:
            results[event_type] = {
                "min_latency": min(latencies) * 1000,  # ms
                "max_latency": max(latencies) * 1000,  # ms
                "avg_latency": sum(latencies) / len(latencies) * 1000,  # ms
                "median_latency": sorted(latencies)[len(latencies) // 2] * 1000,  # ms
                "95th_percentile": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,  # ms
                "received_count": len(latencies)
            }
        else:
            results[event_type] = {
                "error": "No events received"
            }
            
    return results
```

### 3. Adaptive Scaling

```python
class AdaptiveEventBus:
    """Event bus that adapts based on load."""
    
    def __init__(self):
        """Initialize adaptive event bus."""
        self.partitions = {}
        self.metrics = {
            "throughput": {},
            "latency": {}
        }
        self._monitor_task = None
        
    async def start_monitoring(self):
        """Start performance monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_performance())
        
    async def _monitor_performance(self):
        """Monitor performance and adapt."""
        while True:
            try:
                # Collect metrics
                for partition_name, partition in self.partitions.items():
                    # Check queue backlog
                    backlog = await partition.get_queue_size()
                    
                    # Check processing rate
                    processing_rate = await partition.get_processing_rate()
                    
                    self.metrics["throughput"][partition_name] = processing_rate
                    
                    # Adapt based on metrics
                    if backlog > 1000 and processing_rate < 5000:
                        # High backlog, low processing rate
                        await self._scale_partition(partition_name, "up")
                    elif backlog < 10 and processing_rate > 100:
                        # Low backlog, high processing rate
                        await self._scale_partition(partition_name, "down")
                        
                # Wait before next check
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
                
    async def _scale_partition(self, partition_name, direction):
        """
        Scale partition.
        
        Args:
            partition_name: Partition to scale
            direction: "up" or "down"
        """
        if partition_name not in self.partitions:
            return
            
        partition = self.partitions[partition_name]
        
        if direction == "up":
            # Scale up
            await partition.increase_workers()
        else:
            # Scale down
            await partition.decrease_workers()
```

## Implementation Strategies

### 1. Incremental Adoption

The scalable event system can be adopted incrementally:

1. **Phase 1**: Implement partitioned event bus
2. **Phase 2**: Add event batching for high-volume events
3. **Phase 3**: Implement prioritization for critical events
4. **Phase 4**: Adopt lock-free data structures for critical paths
5. **Phase 5**: Add adaptive scaling based on metrics

### 2. Specialized Implementations

Different execution modes may use different event system implementations:

| Execution Mode | Recommended Implementation |
|----------------|----------------------------|
| Backtest       | Simple, single-threaded event bus |
| Optimization   | Multi-process event coordination |
| Live Trading   | Partitioned, lock-free event bus with prioritization |

### 3. Hybrid Approach

A hybrid approach combines the simplicity of the basic event bus with the performance of specialized implementations:

```python
class HybridEventBus:
    """Event bus that adapts to execution context."""
    
    def __init__(self, context):
        """
        Initialize hybrid event bus.
        
        Args:
            context: Execution context
        """
        self.context = context
        
        # Create appropriate implementation based on context
        if context.is_backtest and not context.is_multi_threaded:
            # Simple implementation for single-threaded backtest
            self._impl = SimpleEventBus()
        elif context.is_optimization:
            # Process-aware implementation for optimization
            self._impl = ProcessEventBus()
        elif context.is_live:
            # Scalable implementation for live trading
            self._impl = ScalableEventBus()
        else:
            # Default to simple async implementation
            self._impl = AsyncEventBus()
            
    async def publish(self, event):
        """
        Publish event.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: Success status
        """
        return await self._impl.publish(event)
        
    async def subscribe(self, event_type, handler, is_async=True):
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Event handler function
            is_async: Whether handler is async
            
        Returns:
            Subscription ID
        """
        return await self._impl.subscribe(event_type, handler, is_async)
        
    async def unsubscribe(self, subscription_id):
        """
        Unsubscribe from event.
        
        Args:
            subscription_id: Subscription ID from subscribe
            
        Returns:
            bool: Success status
        """
        return await self._impl.unsubscribe(subscription_id)
```

## Best Practices

### 1. Event Design

- Keep events small and focused
- Include only necessary data in events
- Use consistent naming conventions for event types
- Include timestamps in all events
- Design events to be serializable

### 2. Handler Design

- Keep handlers lightweight
- Move heavy processing to separate tasks
- Implement timeouts for all handlers
- Handle exceptions within handlers
- Log handler performance metrics

### 3. Performance Monitoring

- Track event throughput per event type
- Measure end-to-end latency for critical events
- Monitor queue backlog for early detection of bottlenecks
- Set up alerts for abnormal event processing patterns
- Periodically review event system performance metrics

### 4. Scaling Guidelines

| Event Volume | Recommended Architecture |
|--------------|--------------------------|
| < 100/sec    | Basic event bus is sufficient |
| 100-1,000/sec | Use partitioned event bus |
| 1,000-10,000/sec | Add batching and prioritization |
| > 10,000/sec | Implement full lock-free architecture with adaptive scaling |

## Conclusion

The scalable event system architecture provides the foundation for high-performance event processing in the ADMF-Trader system. By implementing partitioning, batching, prioritization, and lock-free designs, the system can handle the high event volumes required for high-frequency trading while maintaining low latency for critical operations.

Key benefits include:

1. **Scalability**: The system scales with increasing event volume
2. **Low Latency**: Critical events are processed with microsecond-level latency
3. **Prioritization**: Important events take precedence over less time-sensitive events
4. **Resource Efficiency**: CPU and memory usage are optimized
5. **Adaptability**: The system adapts to different execution modes and loads

For implementation, the recommended approach is to start with the basic event system and incrementally adopt scalability features as needed based on performance benchmarks.