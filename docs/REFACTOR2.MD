# ADMF-Trader Advanced Features: Protocol + Composition Architecture

## Table of Contents

1. [Advanced Event System](#1-advanced-event-system)
2. [Advanced Data Management](#2-advanced-data-management)
3. [Advanced Strategy Components](#3-advanced-strategy-components)
4. [Advanced Risk Management](#4-advanced-risk-management)
5. [Advanced Execution Features](#5-advanced-execution-features)
6. [Performance Optimization Framework](#6-performance-optimization-framework)
7. [Infrastructure Services (Advanced)](#7-infrastructure-services-advanced)
8. [Analytics & Reporting Framework](#8-analytics--reporting-framework)
9. [Concurrency & Asynchronous Architecture](#9-concurrency--asynchronous-architecture)

---

## 1. Advanced Event System

### 1.1 Event Context Isolation

```python
import threading
from typing import Optional, Any, Dict
from contextlib import contextmanager

class EventContext:
    """Defines scope for events to prevent cross-contamination"""
    
    def __init__(self, context_id: str, parent: Optional['EventContext'] = None):
        self.context_id = context_id
        self.parent = parent
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
    
    def __enter__(self):
        EventContextManager.set_current_context(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        EventContextManager.restore_previous_context()
    
    def is_compatible_with(self, other: 'EventContext') -> bool:
        """Check if events can flow between contexts"""
        if self == other:
            return True
        
        # Check parent hierarchy
        current = self.parent
        while current:
            if current == other:
                return True
            current = current.parent
        
        return False

class EventContextManager:
    """Thread-local context management"""
    
    _local = threading.local()
    
    @classmethod
    def get_current_context(cls) -> Optional[EventContext]:
        """Get current context for this thread"""
        return getattr(cls._local, 'current_context', None)
    
    @classmethod
    def set_current_context(cls, context: EventContext) -> None:
        """Set current context"""
        # Store previous context for restoration
        previous = getattr(cls._local, 'current_context', None)
        if not hasattr(cls._local, 'context_stack'):
            cls._local.context_stack = []
        cls._local.context_stack.append(previous)
        cls._local.current_context = context
    
    @classmethod
    def restore_previous_context(cls) -> None:
        """Restore previous context"""
        if hasattr(cls._local, 'context_stack') and cls._local.context_stack:
            cls._local.current_context = cls._local.context_stack.pop()

@contextmanager
def event_context(context_id: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for event isolation"""
    context = EventContext(context_id)
    if metadata:
        context.metadata.update(metadata)
    
    with context:
        yield context

class ContextAwareEventBus:
    """Event bus with context isolation support"""
    
    def __init__(self):
        self.global_subscribers: Dict[str, List[Callable]] = {}
        self.context_subscribers: Dict[str, Dict[str, List[Callable]]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, handler: Callable, 
                 context: Optional[EventContext] = None) -> None:
        """Subscribe with optional context isolation"""
        with self._lock:
            if context is None:
                # Global subscription
                if event_type not in self.global_subscribers:
                    self.global_subscribers[event_type] = []
                self.global_subscribers[event_type].append(handler)
            else:
                # Context-specific subscription
                context_id = context.context_id
                if context_id not in self.context_subscribers:
                    self.context_subscribers[context_id] = {}
                if event_type not in self.context_subscribers[context_id]:
                    self.context_subscribers[context_id][event_type] = []
                self.context_subscribers[context_id][event_type].append(handler)
    
    def publish(self, event: Event) -> None:
        """Publish event respecting context boundaries"""
        current_context = EventContextManager.get_current_context()
        
        # Add context to event if not present
        if event.context is None and current_context:
            event.context = current_context
        
        handlers = []
        
        with self._lock:
            # Always include global handlers
            event_type = str(event.event_type)
            handlers.extend(self.global_subscribers.get(event_type, []))
            
            # Include context-specific handlers if compatible
            if event.context:
                for context_id, context_subs in self.context_subscribers.items():
                    # Check if current context is compatible
                    if self._is_context_compatible(event.context, context_id):
                        handlers.extend(context_subs.get(event_type, []))
        
        # Deliver to all compatible handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler {handler}: {e}")
    
    def _is_context_compatible(self, event_context: EventContext, subscriber_context_id: str) -> bool:
        """Check if event context is compatible with subscriber context"""
        return event_context.context_id == subscriber_context_id
```

### 1.2 Event System Scalability

```python
import asyncio
import queue
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

class EventPartition:
    """Single partition for event processing"""
    
    def __init__(self, partition_id: str, worker_count: int = 1):
        self.partition_id = partition_id
        self.event_queue = queue.Queue()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=worker_count)
        self.running = False
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events in this partition"""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    def publish(self, event: Event) -> None:
        """Add event to partition queue"""
        if self.running:
            self.event_queue.put(event)
    
    def start(self) -> None:
        """Start processing events"""
        self.running = True
        for _ in range(self.worker_pool._max_workers):
            self.worker_pool.submit(self._process_events)
    
    def stop(self) -> None:
        """Stop processing events"""
        self.running = False
        # Add sentinel values to wake up workers
        for _ in range(self.worker_pool._max_workers):
            self.event_queue.put(None)
        self.worker_pool.shutdown(wait=True)
    
    def _process_events(self) -> None:
        """Process events from queue"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                if event is None:  # Sentinel value
                    break
                
                self._handle_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event in partition {self.partition_id}: {e}")
    
    def _handle_event(self, event: Event) -> None:
        """Handle individual event"""
        event_type = str(event.event_type)
        with self._lock:
            handlers = self.subscribers.get(event_type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in handler {handler}: {e}")

class PartitionedEventBus:
    """Event bus with partitioned processing for scalability"""
    
    def __init__(self, partition_config: Dict[str, int] = None):
        self.partitions: Dict[str, EventPartition] = {}
        self.partition_strategy = self._default_partition_strategy
        
        # Default partitions
        default_config = partition_config or {
            'market_data': 2,
            'orders': 1,
            'fills': 1,
            'signals': 2,
            'system': 1
        }
        
        for partition_name, worker_count in default_config.items():
            self.partitions[partition_name] = EventPartition(partition_name, worker_count)
    
    def set_partition_strategy(self, strategy: Callable[[Event], str]) -> None:
        """Set custom partition strategy"""
        self.partition_strategy = strategy
    
    def subscribe(self, event_type: str, handler: Callable, 
                 partition: Optional[str] = None) -> None:
        """Subscribe to events, optionally specifying partition"""
        if partition:
            if partition in self.partitions:
                self.partitions[partition].subscribe(event_type, handler)
        else:
            # Subscribe to all partitions
            for partition_obj in self.partitions.values():
                partition_obj.subscribe(event_type, handler)
    
    def publish(self, event: Event) -> None:
        """Publish event to appropriate partition"""
        partition_name = self.partition_strategy(event)
        if partition_name in self.partitions:
            self.partitions[partition_name].publish(event)
        else:
            print(f"Unknown partition: {partition_name}")
    
    def start(self) -> None:
        """Start all partitions"""
        for partition in self.partitions.values():
            partition.start()
    
    def stop(self) -> None:
        """Stop all partitions"""
        for partition in self.partitions.values():
            partition.stop()
    
    def _default_partition_strategy(self, event: Event) -> str:
        """Default partitioning strategy"""
        event_type = str(event.event_type)
        
        if event_type == 'BAR':
            return 'market_data'
        elif event_type in ['ORDER', 'ORDER_UPDATE']:
            return 'orders'
        elif event_type in ['FILL', 'TRADE']:
            return 'fills'
        elif event_type == 'SIGNAL':
            return 'signals'
        else:
            return 'system'

class LockFreeEventQueue:
    """Lock-free event queue for high-performance scenarios"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = [None] * max_size
        self.read_pos = 0
        self.write_pos = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
    
    def put(self, event: Event) -> bool:
        """Put event in queue, return False if full"""
        with self._write_lock:
            next_pos = (self.write_pos + 1) % self.max_size
            if next_pos == self.read_pos:  # Queue full
                return False
            
            self.buffer[self.write_pos] = event
            self.write_pos = next_pos
            return True
    
    def get(self) -> Optional[Event]:
        """Get event from queue, return None if empty"""
        with self._read_lock:
            if self.read_pos == self.write_pos:  # Queue empty
                return None
            
            event = self.buffer[self.read_pos]
            self.buffer[self.read_pos] = None  # Clear reference
            self.read_pos = (self.read_pos + 1) % self.max_size
            return event

class EventTracer:
    """Traces event flow for debugging"""
    
    def __init__(self, max_trace_length: int = 1000):
        self.max_trace_length = max_trace_length
        self.trace_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self.enabled = False
    
    def enable(self) -> None:
        """Enable event tracing"""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable event tracing"""
        self.enabled = False
    
    def trace_event(self, event: Event, action: str, component: str = None) -> None:
        """Trace an event action"""
        if not self.enabled:
            return
        
        with self._lock:
            trace_entry = {
                'timestamp': datetime.now(),
                'event_type': str(event.event_type),
                'event_id': id(event),
                'action': action,  # 'published', 'received', 'handled'
                'component': component,
                'context': event.context.context_id if event.context else None,
                'thread': threading.current_thread().name
            }
            
            self.trace_log.append(trace_entry)
            
            # Limit trace log size
            if len(self.trace_log) > self.max_trace_length:
                self.trace_log.pop(0)
    
    def get_trace_for_event(self, event_id: int) -> List[Dict[str, Any]]:
        """Get trace entries for specific event"""
        with self._lock:
            return [entry for entry in self.trace_log if entry['event_id'] == event_id]
    
    def get_recent_trace(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent trace entries"""
        with self._lock:
            return self.trace_log[-count:]
```

---

## 2. Advanced Data Management

### 2.1 Data Isolation Strategies

```python
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict, Protocol
from abc import abstractmethod
import multiprocessing.shared_memory as shared_memory
import pickle

@runtime_checkable
class DataView(Protocol):
    """Protocol for data views"""
    
    @abstractmethod
    def get_current_data(self) -> pd.DataFrame:
        """Get current data slice"""
        ...
    
    @abstractmethod
    def advance(self, steps: int = 1) -> bool:
        """Advance view by steps"""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset view to beginning"""
        ...

class IsolationMode(Enum):
    DEEP_COPY = "deep_copy"
    VIEW_BASED = "view_based"
    COPY_ON_WRITE = "copy_on_write"
    SHARED_MEMORY = "shared_memory"

class DataIsolationFactory:
    """Factory for creating appropriate data isolation strategy"""
    
    MEMORY_THRESHOLDS = {
        'small': 50 * 1024 * 1024,    # 50MB
        'medium': 200 * 1024 * 1024,  # 200MB
        'large': 1024 * 1024 * 1024   # 1GB
    }
    
    @classmethod
    def create_isolation(cls, data: pd.DataFrame, mode: Optional[IsolationMode] = None) -> Any:
        """Create appropriate isolation strategy"""
        if mode is None:
            mode = cls._recommend_mode(data)
        
        if mode == IsolationMode.DEEP_COPY:
            return DeepCopyIsolation(data)
        elif mode == IsolationMode.VIEW_BASED:
            return ViewBasedIsolation(data)
        elif mode == IsolationMode.COPY_ON_WRITE:
            return CopyOnWriteIsolation(data)
        elif mode == IsolationMode.SHARED_MEMORY:
            return SharedMemoryIsolation(data)
        else:
            raise ValueError(f"Unknown isolation mode: {mode}")
    
    @classmethod
    def _recommend_mode(cls, data: pd.DataFrame) -> IsolationMode:
        """Recommend isolation mode based on data size"""
        memory_usage = data.memory_usage(deep=True).sum()
        
        if memory_usage < cls.MEMORY_THRESHOLDS['small']:
            return IsolationMode.DEEP_COPY
        elif memory_usage < cls.MEMORY_THRESHOLDS['medium']:
            return IsolationMode.VIEW_BASED
        elif memory_usage < cls.MEMORY_THRESHOLDS['large']:
            return IsolationMode.COPY_ON_WRITE
        else:
            return IsolationMode.SHARED_MEMORY

class ViewBasedIsolation:
    """Read-only view isolation for memory efficiency"""
    
    def __init__(self, data: pd.DataFrame):
        self.original_data = data
        self.current_position = 0
        self.view_size = len(data)
        self._lock = threading.RLock()
    
    def get_train_test_split(self, train_ratio: float = 0.7) -> tuple['DataView', 'DataView']:
        """Get train/test views"""
        split_point = int(len(self.original_data) * train_ratio)
        
        train_view = DataFrameView(self.original_data, 0, split_point)
        test_view = DataFrameView(self.original_data, split_point, len(self.original_data))
        
        return train_view, test_view
    
    def verify_isolation(self) -> Dict[str, bool]:
        """Verify views don't share mutable state"""
        train_view, test_view = self.get_train_test_split()
        
        # Check that views have different ranges
        train_data = train_view.get_current_data()
        test_data = test_view.get_current_data()
        
        # Verify no overlap in indices
        train_indices = set(train_data.index)
        test_indices = set(test_data.index)
        overlap = train_indices.intersection(test_indices)
        
        return {
            'no_index_overlap': len(overlap) == 0,
            'different_memory_views': id(train_data) != id(test_data),
            'same_underlying_data': train_data.values.base is test_data.values.base
        }

class DataFrameView:
    """View into a pandas DataFrame"""
    
    def __init__(self, data: pd.DataFrame, start_idx: int, end_idx: int):
        self.data = data
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.current_idx = start_idx
    
    def get_current_data(self) -> pd.DataFrame:
        """Get current data slice"""
        return self.data.iloc[self.start_idx:self.current_idx + 1].copy()
    
    def advance(self, steps: int = 1) -> bool:
        """Advance view by steps"""
        new_idx = self.current_idx + steps
        if new_idx < self.end_idx:
            self.current_idx = new_idx
            return True
        return False
    
    def reset(self) -> None:
        """Reset view to beginning"""
        self.current_idx = self.start_idx
    
    def has_more_data(self) -> bool:
        """Check if more data available"""
        return self.current_idx < self.end_idx - 1

class CopyOnWriteIsolation:
    """Copy-on-write isolation for memory efficiency with modification support"""
    
    def __init__(self, data: pd.DataFrame):
        self.original_data = data
        self.modifications: Dict[str, pd.DataFrame] = {}
        self.access_log: List[str] = []
        self._lock = threading.RLock()
    
    def get_split_data(self, split_name: str, start_ratio: float, end_ratio: float) -> 'CopyOnWriteDataFrame':
        """Get split data with copy-on-write semantics"""
        start_idx = int(len(self.original_data) * start_ratio)
        end_idx = int(len(self.original_data) * end_ratio)
        
        return CopyOnWriteDataFrame(
            self.original_data.iloc[start_idx:end_idx],
            self,
            split_name
        )
    
    def mark_modified(self, split_name: str, data: pd.DataFrame) -> None:
        """Mark split as modified and store copy"""
        with self._lock:
            self.modifications[split_name] = data.copy()
            self.access_log.append(f"modified_{split_name}")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage breakdown"""
        with self._lock:
            original_size = self.original_data.memory_usage(deep=True).sum()
            modification_size = sum(
                df.memory_usage(deep=True).sum() 
                for df in self.modifications.values()
            )
            
            return {
                'original_data': original_size,
                'modifications': modification_size,
                'total': original_size + modification_size,
                'modification_count': len(self.modifications)
            }

class CopyOnWriteDataFrame:
    """DataFrame wrapper with copy-on-write semantics"""
    
    def __init__(self, data: pd.DataFrame, parent: CopyOnWriteIsolation, split_name: str):
        self._data = data
        self._parent = parent
        self._split_name = split_name
        self._is_copy = False
    
    def __getattr__(self, name):
        """Delegate read operations to underlying DataFrame"""
        return getattr(self._data, name)
    
    def __setitem__(self, key, value):
        """Trigger copy-on-write for modifications"""
        self._ensure_copy()
        self._data[key] = value
    
    def _ensure_copy(self):
        """Ensure we have a writable copy"""
        if not self._is_copy:
            self._data = self._data.copy()
            self._parent.mark_modified(self._split_name, self._data)
            self._is_copy = True

class SharedMemoryIsolation:
    """Shared memory isolation for large datasets"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.shared_blocks = {}
        self.metadata = {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'index_name': data.index.name,
            'column_names': list(data.columns)
        }
        self._create_shared_memory()
    
    def _create_shared_memory(self):
        """Create shared memory blocks for DataFrame data"""
        # Convert to numpy arrays and store in shared memory
        for col in self.data.columns:
            col_data = self.data[col].values
            
            # Create shared memory block
            nbytes = col_data.nbytes
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            
            # Copy data to shared memory
            shared_array = np.ndarray(col_data.shape, dtype=col_data.dtype, buffer=shm.buf)
            shared_array[:] = col_data[:]
            
            self.shared_blocks[col] = {
                'shm': shm,
                'shape': col_data.shape,
                'dtype': col_data.dtype
            }
    
    def create_view(self, start_idx: int, end_idx: int) -> 'SharedMemoryDataFrame':
        """Create view using existing shared memory"""
        return SharedMemoryDataFrame(self.shared_blocks, self.metadata, start_idx, end_idx)
    
    def cleanup(self):
        """Clean up shared memory blocks"""
        for block_info in self.shared_blocks.values():
            block_info['shm'].close()
            block_info['shm'].unlink()

class SharedMemoryDataFrame:
    """DataFrame view using shared memory"""
    
    def __init__(self, shared_blocks: Dict, metadata: Dict, start_idx: int, end_idx: int):
        self.shared_blocks = shared_blocks
        self.metadata = metadata
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert shared memory view to DataFrame"""
        columns = {}
        
        for col_name, block_info in self.shared_blocks.items():
            # Attach to existing shared memory
            shm = shared_memory.SharedMemory(block_info['shm'].name)
            
            # Create array view
            full_array = np.ndarray(
                block_info['shape'], 
                dtype=block_info['dtype'], 
                buffer=shm.buf
            )
            
            # Slice the view
            columns[col_name] = full_array[self.start_idx:self.end_idx]
        
        return pd.DataFrame(columns)

class MemoryTracker:
    """Tracks memory usage for data operations"""
    
    @staticmethod
    def get_process_memory() -> Dict[str, float]:
        """Get current process memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def get_dataframe_memory(df: pd.DataFrame) -> Dict[str, float]:
        """Get DataFrame memory usage"""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / 1024 / 1024,
            'per_column_mb': {col: memory_usage[col] / 1024 / 1024 
                            for col in memory_usage.index},
            'dtypes': df.dtypes.to_dict()
        }
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency"""
        optimized = df.copy()
        
        for col in optimized.columns:
            col_data = optimized[col]
            
            if pd.api.types.is_integer_dtype(col_data):
                # Downcast integers
                optimized[col] = pd.to_numeric(col_data, downcast='integer')
            elif pd.api.types.is_float_dtype(col_data):
                # Downcast floats
                optimized[col] = pd.to_numeric(col_data, downcast='float')
            elif pd.api.types.is_object_dtype(col_data):
                # Convert to category if beneficial
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized[col] = col_data.astype('category')
        
        return optimized

class TimeSeriesArray:
    """Memory-efficient time series storage"""
    
    def __init__(self, timestamps: np.ndarray, values: np.ndarray):
        self.timestamps = np.asarray(timestamps)
        self.values = np.asarray(values)
        
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have same length")
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, timestamp_col: str, value_col: str) -> 'TimeSeriesArray':
        """Create from DataFrame"""
        return cls(df[timestamp_col].values, df[value_col].values)
    
    def to_dataframe(self, timestamp_col: str = 'timestamp', value_col: str = 'value') -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame({
            timestamp_col: self.timestamps,
            value_col: self.values
        })
    
    def get_slice(self, start_idx: int, end_idx: int) -> 'TimeSeriesArray':
        """Get time series slice"""
        return TimeSeriesArray(
            self.timestamps[start_idx:end_idx],
            self.values[start_idx:end_idx]
        )
    
    def memory_usage(self) -> Dict[str, int]:
        """Get memory usage in bytes"""
        return {
            'timestamps': self.timestamps.nbytes,
            'values': self.values.nbytes,
            'total': self.timestamps.nbytes + self.values.nbytes
        }
```

### 2.2 Data Quality and Validation

```python
from typing import List, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ValidationResult:
    """Result of data validation"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class DataValidator(ABC):
    """Base class for data validators"""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data and return results"""
        ...

class ContinuityValidator(DataValidator):
    """Validates data continuity and completeness"""
    
    def __init__(self, timestamp_col: str = 'timestamp', expected_frequency: str = '1min'):
        self.timestamp_col = timestamp_col
        self.expected_frequency = expected_frequency
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check for gaps in time series data"""
        errors = []
        warnings = []
        metadata = {}
        
        if self.timestamp_col not in data.columns:
            errors.append(f"Timestamp column '{self.timestamp_col}' not found")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check for gaps
        timestamps = pd.to_datetime(data[self.timestamp_col])
        expected_range = pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq=self.expected_frequency
        )
        
        missing_timestamps = expected_range.difference(timestamps)
        if len(missing_timestamps) > 0:
            warnings.append(f"Found {len(missing_timestamps)} missing timestamps")
            metadata['missing_timestamps'] = missing_timestamps.tolist()
        
        # Check for duplicates
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate timestamps")
        
        # Calculate completeness ratio
        completeness = len(timestamps) / len(expected_range)
        metadata['completeness_ratio'] = completeness
        
        if completeness < 0.95:
            warnings.append(f"Data completeness is only {completeness:.2%}")
        
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

class OHLCValidator(DataValidator):
    """Validates OHLC data integrity"""
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate OHLC relationships"""
        errors = []
        warnings = []
        metadata = {}
        
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check OHLC relationships
        invalid_high = (data['high'] < data['open']) | (data['high'] < data['close'])
        invalid_low = (data['low'] > data['open']) | (data['low'] > data['close'])
        
        if invalid_high.any():
            count = invalid_high.sum()
            errors.append(f"Found {count} bars where high < open or high < close")
            metadata['invalid_high_indices'] = data[invalid_high].index.tolist()
        
        if invalid_low.any():
            count = invalid_low.sum()
            errors.append(f"Found {count} bars where low > open or low > close")
            metadata['invalid_low_indices'] = data[invalid_low].index.tolist()
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            zero_or_negative = (data[col] <= 0).sum()
            if zero_or_negative > 0:
                errors.append(f"Found {zero_or_negative} zero/negative values in {col}")
        
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

class OutlierDetector(DataValidator):
    """Detects outliers in price data"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 3.0):
        self.method = method
        self.threshold = threshold
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Detect outliers in data"""
        errors = []
        warnings = []
        metadata = {}
        
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
        
        for col in price_cols:
            outliers = self._detect_outliers(data[col])
            if len(outliers) > 0:
                warnings.append(f"Found {len(outliers)} outliers in {col}")
                metadata[f'{col}_outliers'] = outliers.tolist()
        
        return ValidationResult(
            passed=True,  # Outliers are warnings, not errors
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _detect_outliers(self, series: pd.Series) -> pd.Index:
        """Detect outliers using specified method"""
        if self.method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            return series[(series < lower_bound) | (series > upper_bound)].index
        
        elif self.method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return series[z_scores > self.threshold].index
        
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

class DataQualityManager:
    """Manages data quality validation"""
    
    def __init__(self):
        self.validators: List[DataValidator] = []
        self.validation_history: List[Dict[str, Any]] = []
    
    def add_validator(self, validator: DataValidator) -> None:
        """Add validator to the pipeline"""
        self.validators.append(validator)
    
    def validate_data(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, ValidationResult]:
        """Run all validators on data"""
        results = {}
        
        for i, validator in enumerate(self.validators):
            validator_name = validator.__class__.__name__
            try:
                result = validator.validate(data)
                results[validator_name] = result
            except Exception as e:
                results[validator_name] = ValidationResult(
                    passed=False,
                    errors=[f"Validator error: {str(e)}"],
                    warnings=[],
                    metadata={}
                )
        
        # Store validation history
        self.validation_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'data_shape': data.shape,
            'results': results
        })
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results"""
        total_validators = len(results)
        passed_validators = sum(1 for result in results.values() if result.passed)
        total_errors = sum(len(result.errors) for result in results.values())
        total_warnings = sum(len(result.warnings) for result in results.values())
        
        return {
            'overall_passed': total_errors == 0,
            'validators_passed': passed_validators,
            'total_validators': total_validators,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'pass_rate': passed_validators / total_validators if total_validators > 0 else 0
        }
```

---

## 3. Advanced Strategy Components

### 3.1 MetaComponent Framework

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

class AnalysisType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    FEATURE_EXTRACTION = "feature_extraction"

@runtime_checkable
class MetaComponent(Protocol):
    """Protocol for components that analyze without trading"""
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze data and return insights"""
        ...
    
    @abstractmethod
    def get_current_analysis(self) -> Optional[Dict[str, Any]]:
        """Get current analysis results"""
        ...
    
    @abstractmethod
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical analysis results"""
        ...

class MetaComponentBase:
    """Base implementation for meta components"""
    
    def __init__(self, name: str, analysis_type: AnalysisType):
        self.name = name
        self.analysis_type = analysis_type
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.analysis_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
        self._optimization: Optional[OptimizationSupport] = None
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze data and update state"""
        result = self._perform_analysis(data)
        
        # Add metadata
        result.update({
            'timestamp': datetime.now(),
            'component': self.name,
            'analysis_type': self.analysis_type.value
        })
        
        # Update state
        self.current_analysis = result
        self.analysis_history.append(result)
        
        # Limit history size
        if len(self.analysis_history) > self.max_history:
            self.analysis_history.pop(0)
        
        return result
    
    @abstractmethod
    def _perform_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform the actual analysis - implemented by subclasses"""
        ...
    
    def get_current_analysis(self) -> Optional[Dict[str, Any]]:
        """Get current analysis results"""
        return self.current_analysis
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical analysis results"""
        return self.analysis_history[-limit:]
    
    def reset(self) -> None:
        """Reset component state"""
        self.current_analysis = None
        self.analysis_history.clear()

class Classifier(MetaComponentBase):
    """Classifier for market regime detection"""
    
    def __init__(self, name: str, classification_categories: List[str]):
        super().__init__(name, AnalysisType.CLASSIFICATION)
        self.categories = classification_categories
        self.current_classification: Optional[str] = None
        self.classification_confidence: float = 0.0
        self.classification_history: List[Dict[str, Any]] = []
        
        # Stabilization parameters
        self.min_confidence_threshold = 0.6
        self.min_duration_bars = 5
        self.transition_buffer: List[str] = []
    
    def classify(self, data: Any) -> str:
        """Classify data into categories"""
        classification_result = self._perform_classification(data)
        category = classification_result['category']
        confidence = classification_result['confidence']
        
        # Apply stabilization logic
        stabilized_category = self._apply_stabilization(category, confidence)
        
        # Update state if category changed
        if stabilized_category != self.current_classification:
            self._handle_classification_change(stabilized_category, confidence)
        
        return stabilized_category
    
    @abstractmethod
    def _perform_classification(self, data: Any) -> Dict[str, Any]:
        """Perform actual classification - implemented by subclasses"""
        ...
    
    def _apply_stabilization(self, new_category: str, confidence: float) -> str:
        """Apply stabilization to prevent rapid switching"""
        # Check confidence threshold
        if confidence < self.min_confidence_threshold:
            return self.current_classification or new_category
        
        # Apply minimum duration requirement
        self.transition_buffer.append(new_category)
        if len(self.transition_buffer) > self.min_duration_bars:
            self.transition_buffer.pop(0)
        
        # Check if buffer is consistent
        if len(self.transition_buffer) >= self.min_duration_bars:
            if all(cat == new_category for cat in self.transition_buffer):
                return new_category
        
        return self.current_classification or new_category
    
    def _handle_classification_change(self, new_category: str, confidence: float) -> None:
        """Handle classification change"""
        old_category = self.current_classification
        self.current_classification = new_category
        self.classification_confidence = confidence
        
        # Record classification change
        change_record = {
            'timestamp': datetime.now(),
            'old_category': old_category,
            'new_category': new_category,
            'confidence': confidence
        }
        self.classification_history.append(change_record)
        
        # Emit classification event
        if hasattr(self, '_events') and self._events and self._events.event_bus:
            event = Event(EventType.CLASSIFICATION, {
                'classifier': self.name,
                'category': new_category,
                'previous_category': old_category,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            self._events.event_bus.publish(event)
    
    def get_current_classification(self) -> Optional[str]:
        """Get current classification"""
        return self.current_classification
    
    def get_classification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get classification change history"""
        return self.classification_history[-limit:]

class RegimeDetector(Classifier):
    """Market regime detector using technical indicators"""
    
    def __init__(self, name: str = "regime_detector", 
                 volatility_window: int = 20,
                 trend_window: int = 50,
                 volatility_threshold: float = 0.02):
        
        categories = ["low_vol_uptrend", "low_vol_downtrend", "low_vol_sideways",
                     "high_vol_uptrend", "high_vol_downtrend", "high_vol_sideways"]
        
        super().__init__(name, categories)
        
        # Parameters
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volatility_threshold = volatility_threshold
        
        # State
        self.price_history: List[float] = []
        
        # Set up optimization
        if hasattr(self, '_optimization'):
            self._optimization.set_parameter_space({
                'volatility_window': [10, 15, 20, 25, 30],
                'trend_window': [30, 40, 50, 60],
                'volatility_threshold': [0.01, 0.015, 0.02, 0.025, 0.03]
            })
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        """Handle new price bar"""
        bar_data = event.payload
        price = bar_data['close']
        
        self.price_history.append(price)
        
        # Limit history size
        max_history = max(self.volatility_window, self.trend_window) * 2
        if len(self.price_history) > max_history:
            self.price_history.pop(0)
        
        # Classify if we have enough data
        if len(self.price_history) >= max(self.volatility_window, self.trend_window):
            classification = self.classify(self.price_history)
    
    def _perform_classification(self, data: Any) -> Dict[str, Any]:
        """Classify market regime based on volatility and trend"""
        if isinstance(data, list):
            prices = np.array(data)
        else:
            prices = data
        
        # Calculate volatility (rolling standard deviation of returns)
        if len(prices) < self.volatility_window:
            return {'category': 'low_vol_sideways', 'confidence': 0.5}
        
        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-self.volatility_window:]
        volatility = np.std(recent_returns)
        
        # Calculate trend (slope of linear regression)
        if len(prices) < self.trend_window:
            trend_slope = 0
        else:
            recent_prices = prices[-self.trend_window:]
            x = np.arange(len(recent_prices))
            trend_slope = np.polyfit(x, recent_prices, 1)[0] / np.mean(recent_prices)
        
        # Classify volatility
        is_high_vol = volatility > self.volatility_threshold
        vol_category = "high_vol" if is_high_vol else "low_vol"
        
        # Classify trend
        if trend_slope > 0.001:  # 0.1% per bar
            trend_category = "uptrend"
        elif trend_slope < -0.001:
            trend_category = "downtrend"
        else:
            trend_category = "sideways"
        
        # Combine classifications
        category = f"{vol_category}_{trend_category}"
        
        # Calculate confidence based on how clear the signals are
        vol_confidence = min(abs(volatility - self.volatility_threshold) / self.volatility_threshold, 1.0)
        trend_confidence = min(abs(trend_slope) / 0.001, 1.0)
        overall_confidence = (vol_confidence + trend_confidence) / 2
        
        return {
            'category': category,
            'confidence': overall_confidence,
            'volatility': volatility,
            'trend_slope': trend_slope,
            'vol_category': vol_category,
            'trend_category': trend_category
        }
    
    def _perform_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform regime analysis"""
        classification_result = self._perform_classification(data)
        
        return {
            'regime': classification_result['category'],
            'confidence': classification_result['confidence'],
            'volatility': classification_result['volatility'],
            'trend_slope': classification_result['trend_slope'],
            'analysis_type': 'regime_detection'
        }

class MetaLabeler(MetaComponentBase):
    """Meta-labeling component for signal quality assessment"""
    
    def __init__(self, name: str = "meta_labeler", model_type: str = "xgboost"):
        super().__init__(name, AnalysisType.CLASSIFICATION)
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        # Feature engineering components
        self.feature_extractors: List[Callable] = []
        self.signal_history: List[Dict[str, Any]] = []
        self.trade_outcomes: List[Dict[str, Any]] = []
        
        # Training parameters
        self.lookback_window = 100
        self.min_samples_for_training = 50
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.SIGNAL, self.on_signal)
            self._events.subscribe(EventType.FILL, self.on_fill)
    
    def on_signal(self, event: Event) -> None:
        """Collect signal for analysis"""
        signal_data = event.payload
        
        # Extract features from signal and current market state
        features = self._extract_signal_features(signal_data)
        
        # Store signal with features
        signal_record = {
            'signal_id': id(event),
            'timestamp': signal_data.get('timestamp', datetime.now()),
            'features': features,
            'signal_data': signal_data
        }
        
        self.signal_history.append(signal_record)
        
        # Analyze signal quality if model is trained
        if self.is_trained and self.model:
            quality_score = self._assess_signal_quality(features)
            
            # Emit meta-label event
            meta_label_event = Event(EventType.META_LABEL, {
                'signal_id': id(event),
                'quality_score': quality_score,
                'confidence': self._calculate_confidence(features),
                'meta_labeler': self.name,
                'timestamp': datetime.now()
            })
            
            if self._events and self._events.event_bus:
                self._events.event_bus.publish(meta_label_event)
    
    def on_fill(self, event: Event) -> None:
        """Track trade outcomes for model training"""
        fill_data = event.payload
        
        # Find corresponding signal
        signal_id = fill_data.get('signal_id')  # Assuming this is tracked
        if signal_id:
            trade_record = {
                'signal_id': signal_id,
                'fill_timestamp': fill_data.get('timestamp'),
                'fill_price': fill_data.get('price'),
                'quantity': fill_data.get('quantity')
            }
            self.trade_outcomes.append(trade_record)
    
    def _extract_signal_features(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from signal for meta-labeling"""
        features = {}
        
        # Basic signal features
        features['signal_strength'] = signal_data.get('strength', 1.0)
        features['hour_of_day'] = signal_data.get('timestamp', datetime.now()).hour
        features['day_of_week'] = signal_data.get('timestamp', datetime.now()).weekday()
        
        # Market context features (would need access to current market data)
        # features['recent_volatility'] = self._calculate_recent_volatility()
        # features['recent_volume'] = self._calculate_recent_volume()
        # features['rsi'] = self._calculate_rsi()
        
        # Signal pattern features
        features['signals_in_last_hour'] = self._count_recent_signals(hours=1)
        features['signals_in_last_day'] = self._count_recent_signals(hours=24)
        
        # Strategy features
        strategy_name = signal_data.get('strategy', 'unknown')
        features[f'strategy_{strategy_name}'] = 1.0
        
        return features
    
    def _count_recent_signals(self, hours: int) -> int:
        """Count signals in recent time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return sum(1 for record in self.signal_history 
                  if record['timestamp'] > cutoff_time)
    
    def _assess_signal_quality(self, features: Dict[str, float]) -> float:
        """Assess signal quality using trained model"""
        if not self.is_trained or not self.model:
            return 0.5  # Neutral score
        
        # Convert features to model input format
        feature_vector = self._features_to_vector(features)
        
        # Get prediction from model
        try:
            if self.model_type == "xgboost":
                import xgboost as xgb
                quality_score = self.model.predict_proba([feature_vector])[0][1]  # Probability of positive class
            else:
                quality_score = self.model.predict_proba([feature_vector])[0][1]
            
            return quality_score
        except Exception as e:
            print(f"Error in meta-labeling prediction: {e}")
            return 0.5
    
    def _features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dict to vector for model input"""
        # This would need to match the feature order used during training
        return list(features.values())
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in meta-label prediction"""
        # Simple heuristic - could be more sophisticated
        signal_strength = features.get('signal_strength', 0.5)
        recent_signal_density = features.get('signals_in_last_hour', 0) / 10.0  # Normalize
        
        # Higher signal strength and moderate signal density indicate higher confidence
        confidence = signal_strength * (1.0 - min(recent_signal_density, 1.0))
        return max(0.1, min(0.9, confidence))
    
    def train_model(self, lookback_days: int = 30) -> bool:
        """Train meta-labeling model on historical data"""
        # This would implement the full training pipeline
        # Including feature extraction, label generation, model training
        
        print(f"Training meta-labeling model with {lookback_days} days of data...")
        
        # Placeholder for actual training implementation
        self.is_trained = True
        return True
    
    def _perform_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform meta-analysis on signals"""
        if not self.signal_history:
            return {'analysis_type': 'meta_labeling', 'status': 'no_data'}
        
        recent_signals = self.signal_history[-10:]  # Last 10 signals
        
        analysis = {
            'analysis_type': 'meta_labeling',
            'signal_count': len(recent_signals),
            'model_trained': self.is_trained,
            'avg_signal_strength': np.mean([s['signal_data'].get('strength', 1.0) 
                                          for s in recent_signals]),
            'signal_frequency': len(self.signal_history) / max(1, 
                (datetime.now() - self.signal_history[0]['timestamp']).total_seconds() / 3600)
        }
        
        return analysis
```

### 3.2 Regime-Adaptive Strategies

```python
class RegimeAdaptiveStrategy:
    """Strategy that adapts parameters based on market regime"""
    
    def __init__(self, name: str, base_strategy_class: type, 
                 regime_detector_name: str = "regime_detector"):
        self.name = name
        self.base_strategy_class = base_strategy_class
        self.regime_detector_name = regime_detector_name
        
        # Strategy instances per regime
        self.regime_strategies: Dict[str, Any] = {}
        self.regime_parameters: Dict[str, Dict[str, Any]] = {}
        
        # Current state
        self.current_regime: Optional[str] = None
        self.active_strategy: Optional[Any] = None
        self.regime_detector: Optional[Any] = None
        
        # Adaptation settings
        self.adaptation_mode = "immediate"  # "immediate", "delayed", "gradual"
        self.pending_regime_change: Optional[str] = None
        self.regime_change_delay = 5  # bars to wait before switching
        self.regime_change_counter = 0
        
        # Position management during regime changes
        self.close_positions_on_change = False
        self.trade_locked_parameters = False  # Use entry regime params for full trade
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
        self._optimization: Optional[OptimizationSupport] = None
    
    def initialize(self, context: SystemContext) -> None:
        """Initialize adaptive strategy"""
        # Get regime detector
        self.regime_detector = context.resolve(self.regime_detector_name)
        
        # Load regime-specific parameters
        self._load_regime_parameters()
        
        # Create strategy instances for each regime
        self._create_regime_strategies()
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.CLASSIFICATION, self.on_regime_change)
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def _load_regime_parameters(self) -> None:
        """Load optimized parameters for each regime"""
        # This would load from configuration or optimization results
        # Example structure:
        self.regime_parameters = {
            "low_vol_uptrend": {
                "fast_period": 5,
                "slow_period": 20,
                "signal_threshold": 0.01
            },
            "high_vol_uptrend": {
                "fast_period": 10,
                "slow_period": 30,
                "signal_threshold": 0.03
            },
            "low_vol_downtrend": {
                "fast_period": 8,
                "slow_period": 25,
                "signal_threshold": 0.015
            },
            # ... other regimes
        }
    
    def _create_regime_strategies(self) -> None:
        """Create strategy instances for each regime"""
        for regime, params in self.regime_parameters.items():
            strategy_instance = self.base_strategy_class(**params)
            
            # Add capabilities if needed
            if hasattr(strategy_instance, '_lifecycle'):
                strategy_instance._lifecycle = ComponentLifecycle(f"{self.name}_{regime}")
            
            self.regime_strategies[regime] = strategy_instance
    
    def on_regime_change(self, event: Event) -> None:
        """Handle regime change notification"""
        classification_data = event.payload
        
        # Check if this is from our regime detector
        if classification_data.get('classifier') != self.regime_detector_name:
            return
        
        new_regime = classification_data['category']
        confidence = classification_data['confidence']
        
        # Only switch if confidence is high enough
        if confidence < 0.7:
            return
        
        if new_regime != self.current_regime:
            self._handle_regime_transition(new_regime)
    
    def _handle_regime_transition(self, new_regime: str) -> None:
        """Handle transition to new regime"""
        if self.adaptation_mode == "immediate":
            self._switch_regime_immediately(new_regime)
        
        elif self.adaptation_mode == "delayed":
            if self.pending_regime_change != new_regime:
                self.pending_regime_change = new_regime
                self.regime_change_counter = 0
            else:
                self.regime_change_counter += 1
                if self.regime_change_counter >= self.regime_change_delay:
                    self._switch_regime_immediately(new_regime)
        
        elif self.adaptation_mode == "gradual":
            self._switch_regime_gradually(new_regime)
    
    def _switch_regime_immediately(self, new_regime: str) -> None:
        """Switch to new regime immediately"""
        old_regime = self.current_regime
        
        # Close positions if configured
        if self.close_positions_on_change and self.active_strategy:
            self._close_all_positions()
        
        # Switch active strategy
        if new_regime in self.regime_strategies:
            self.current_regime = new_regime
            self.active_strategy = self.regime_strategies[new_regime]
            
            # Reset pending change
            self.pending_regime_change = None
            self.regime_change_counter = 0
            
            # Log regime change
            if hasattr(self, 'logger'):
                self.logger.info(f"Regime switch: {old_regime} -> {new_regime}")
    
    def _switch_regime_gradually(self, new_regime: str) -> None:
        """Switch to new regime gradually over multiple bars"""
        # Implement gradual parameter blending
        # This is more complex and would involve interpolating parameters
        pass
    
    def _close_all_positions(self) -> None:
        """Close all open positions"""
        # This would emit close signals for all open positions
        # Implementation depends on how positions are tracked
        pass
    
    def on_bar(self, event: Event) -> None:
        """Forward bar events to active strategy"""
        if self.active_strategy and hasattr(self.active_strategy, 'on_bar'):
            self.active_strategy.on_bar(event)
    
    def generate_signal(self, data: Any) -> Optional[Dict[str, Any]]:
        """Generate signal using active strategy"""
        if self.active_strategy and hasattr(self.active_strategy, 'generate_signal'):
            signal = self.active_strategy.generate_signal(data)
            
            # Add regime information to signal
            if signal:
                signal['regime'] = self.current_regime
                signal['adaptive_strategy'] = self.name
            
            return signal
        
        return None
    
    def reset(self) -> None:
        """Reset adaptive strategy"""
        self.current_regime = None
        self.active_strategy = None
        self.pending_regime_change = None
        self.regime_change_counter = 0
        
        # Reset all regime strategies
        for strategy in self.regime_strategies.values():
            if hasattr(strategy, 'reset'):
                strategy.reset()
    
    # Optimization support
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for optimization"""
        # This would return the parameter space for regime-specific optimization
        space = {}
        
        for regime, strategy in self.regime_strategies.items():
            if hasattr(strategy, 'get_parameter_space'):
                regime_space = strategy.get_parameter_space()
                for param, values in regime_space.items():
                    space[f"{regime}.{param}"] = values
        
        return space
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters for regime strategies"""
        # Distribute parameters to appropriate regime strategies
        for param_name, value in params.items():
            if '.' in param_name:
                regime, actual_param = param_name.split('.', 1)
                if regime in self.regime_strategies:
                    strategy = self.regime_strategies[regime]
                    if hasattr(strategy, 'set_parameters'):
                        strategy.set_parameters({actual_param: value})
                    else:
                        # Set parameter directly if no set_parameters method
                        setattr(strategy, actual_param, value)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters for all regime strategies"""
        params = {}
        
        for regime, strategy in self.regime_strategies.items():
            if hasattr(strategy, 'get_parameters'):
                regime_params = strategy.get_parameters()
                for param, value in regime_params.items():
                    params[f"{regime}.{param}"] = value
        
        return params

class BoundaryTradeTracker:
    """Tracks trades that span regime boundaries"""
    
    def __init__(self):
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.boundary_trades: List[Dict[str, Any]] = []
        self.boundary_stats: Dict[str, Any] = {}
    
    def track_trade_entry(self, trade_id: str, entry_regime: str, 
                         entry_signal: Dict[str, Any]) -> None:
        """Track trade entry with regime information"""
        self.active_trades[trade_id] = {
            'entry_regime': entry_regime,
            'entry_signal': entry_signal,
            'entry_timestamp': entry_signal.get('timestamp', datetime.now()),
            'regime_changes': []
        }
    
    def track_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Track regime changes for active trades"""
        timestamp = datetime.now()
        
        for trade_id, trade_info in self.active_trades.items():
            trade_info['regime_changes'].append({
                'from_regime': old_regime,
                'to_regime': new_regime,
                'timestamp': timestamp
            })
    
    def track_trade_exit(self, trade_id: str, exit_regime: str, 
                        exit_result: Dict[str, Any]) -> None:
        """Track trade exit and determine if it's a boundary trade"""
        if trade_id not in self.active_trades:
            return
        
        trade_info = self.active_trades[trade_id]
        entry_regime = trade_info['entry_regime']
        
        # Create trade record
        trade_record = {
            'trade_id': trade_id,
            'entry_regime': entry_regime,
            'exit_regime': exit_regime,
            'is_boundary_trade': entry_regime != exit_regime,
            'regime_changes': trade_info['regime_changes'],
            'entry_timestamp': trade_info['entry_timestamp'],
            'exit_timestamp': exit_result.get('timestamp', datetime.now()),
            'pnl': exit_result.get('pnl', 0),
            'duration': (exit_result.get('timestamp', datetime.now()) - 
                        trade_info['entry_timestamp']).total_seconds() / 60  # minutes
        }
        
        # Store if boundary trade
        if trade_record['is_boundary_trade']:
            self.boundary_trades.append(trade_record)
            self._update_boundary_stats(trade_record)
        
        # Remove from active trades
        del self.active_trades[trade_id]
    
    def _update_boundary_stats(self, trade_record: Dict[str, Any]) -> None:
        """Update boundary trade statistics"""
        transition = f"{trade_record['entry_regime']}_to_{trade_record['exit_regime']}"
        
        if transition not in self.boundary_stats:
            self.boundary_stats[transition] = {
                'count': 0,
                'total_pnl': 0,
                'winning_trades': 0,
                'avg_duration': 0
            }
        
        stats = self.boundary_stats[transition]
        stats['count'] += 1
        stats['total_pnl'] += trade_record['pnl']
        if trade_record['pnl'] > 0:
            stats['winning_trades'] += 1
        
        # Update average duration
        total_duration = stats['avg_duration'] * (stats['count'] - 1) + trade_record['duration']
        stats['avg_duration'] = total_duration / stats['count']
    
    def get_boundary_trade_ratio(self) -> float:
        """Get ratio of boundary trades to total trades"""
        total_trades = len(self.boundary_trades) + len([t for t in self.active_trades.values()])
        if total_trades == 0:
            return 0.0
        return len(self.boundary_trades) / total_trades
    
    def get_worst_transitions(self, top_n: int = 5) -> List[tuple]:
        """Get worst performing regime transitions"""
        transition_performance = []
        
        for transition, stats in self.boundary_stats.items():
            if stats['count'] > 0:
                avg_pnl = stats['total_pnl'] / stats['count']
                win_rate = stats['winning_trades'] / stats['count']
                transition_performance.append((transition, avg_pnl, win_rate, stats['count']))
        
        # Sort by average PnL (worst first)
        transition_performance.sort(key=lambda x: x[1])
        return transition_performance[:top_n]

### 3.3 Signal Processing Pipeline

```python
class SignalProcessor(ABC):
    """Base class for signal processing components"""
    
    @abstractmethod
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process signal, return modified signal or None to filter"""
        ...
    
    @abstractmethod
    def get_name(self) -> str:
        """Get processor name"""
        ...

class SignalProcessingPipeline:
    """Pipeline for processing signals through multiple stages"""
    
    def __init__(self, name: str = "signal_pipeline"):
        self.name = name
        self.processors: List[SignalProcessor] = []
        self.processing_stats: Dict[str, Dict[str, int]] = {}
    
    def add_processor(self, processor: SignalProcessor) -> None:
        """Add processor to pipeline"""
        self.processors.append(processor)
        self.processing_stats[processor.get_name()] = {
            'signals_processed': 0,
            'signals_passed': 0,
            'signals_filtered': 0,
            'signals_modified': 0
        }
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process signal through all processors"""
        current_signal = signal.copy()
        
        for processor in self.processors:
            processor_name = processor.get_name()
            stats = self.processing_stats[processor_name]
            stats['signals_processed'] += 1
            
            try:
                original_signal = current_signal.copy()
                processed_signal = processor.process_signal(current_signal)
                
                if processed_signal is None:
                    # Signal was filtered out
                    stats['signals_filtered'] += 1
                    return None
                
                # Check if signal was modified
                if processed_signal != original_signal:
                    stats['signals_modified'] += 1
                
                stats['signals_passed'] += 1
                current_signal = processed_signal
                
            except Exception as e:
                print(f"Error in signal processor {processor_name}: {e}")
                stats['signals_filtered'] += 1
                return None
        
        return current_signal
    
    def get_processing_stats(self) -> Dict[str, Dict[str, int]]:
        """Get processing statistics"""
        return self.processing_stats.copy()

class RegimeFilter(SignalProcessor):
    """Filters signals based on regime compatibility"""
    
    def __init__(self, regime_detector_name: str, 
                 regime_rules: Dict[str, List[str]]):
        self.regime_detector_name = regime_detector_name
        self.regime_rules = regime_rules  # regime -> allowed signal types
        self.regime_detector: Optional[Any] = None
        self.current_regime: Optional[str] = None
    
    def initialize(self, context: SystemContext) -> None:
        """Initialize with regime detector"""
        self.regime_detector = context.resolve(self.regime_detector_name)
    
    def set_current_regime(self, regime: str) -> None:
        """Update current regime"""
        self.current_regime = regime
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter signal based on regime compatibility"""
        if not self.current_regime:
            return signal  # Pass through if no regime info
        
        signal_direction = signal.get('direction', '').upper()
        
        # Check if signal type is allowed in current regime
        if self.current_regime in self.regime_rules:
            allowed_signals = self.regime_rules[self.current_regime]
            if signal_direction not in allowed_signals:
                # Filter out incompatible signal
                return None
        
        # Add regime information to signal
        signal['regime_filter'] = {
            'regime': self.current_regime,
            'allowed': True,
            'processor': self.get_name()
        }
        
        return signal
    
    def get_name(self) -> str:
        return "regime_filter"

class ConfidenceFilter(SignalProcessor):
    """Filters signals based on confidence scores"""
    
    def __init__(self, min_confidence: float = 0.6,
                 confidence_source: str = "meta_labeler"):
        self.min_confidence = min_confidence
        self.confidence_source = confidence_source
        self.meta_labeler: Optional[Any] = None
    
    def initialize(self, context: SystemContext) -> None:
        """Initialize with meta labeler"""
        try:
            self.meta_labeler = context.resolve(self.confidence_source)
        except:
            print(f"Warning: Could not resolve {self.confidence_source}")
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter signal based on confidence score"""
        # Get confidence from meta labeler or signal itself
        confidence = signal.get('confidence', 0.5)
        
        if self.meta_labeler and hasattr(self.meta_labeler, 'assess_signal_quality'):
            # Use meta labeler for confidence assessment
            try:
                features = self._extract_features(signal)
                confidence = self.meta_labeler._assess_signal_quality(features)
            except Exception as e:
                print(f"Error getting confidence from meta labeler: {e}")
        
        # Filter based on confidence threshold
        if confidence < self.min_confidence:
            return None
        
        # Add confidence information to signal
        signal['confidence_filter'] = {
            'confidence': confidence,
            'threshold': self.min_confidence,
            'passed': True,
            'processor': self.get_name()
        }
        
        return signal
    
    def _extract_features(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for confidence assessment"""
        return {
            'signal_strength': signal.get('strength', 1.0),
            'hour_of_day': signal.get('timestamp', datetime.now()).hour,
            'day_of_week': signal.get('timestamp', datetime.now()).weekday()
        }
    
    def get_name(self) -> str:
        return "confidence_filter"

class SignalSmoother(SignalProcessor):
    """Smooths signals to reduce noise and excessive trading"""
    
    def __init__(self, window_size: int = 3, 
                 consensus_threshold: float = 0.6):
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self.signal_history: Dict[str, List[str]] = {}  # symbol -> signal directions
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Smooth signal based on recent signal history"""
        symbol = signal.get('symbol', 'default')
        direction = signal.get('direction', '').upper()
        
        # Initialize history for symbol if needed
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        history = self.signal_history[symbol]
        history.append(direction)
        
        # Limit history size
        if len(history) > self.window_size:
            history.pop(0)
        
        # Check consensus if we have enough history
        if len(history) >= self.window_size:
            direction_counts = {}
            for hist_direction in history:
                direction_counts[hist_direction] = direction_counts.get(hist_direction, 0) + 1
            
            # Check if current direction has consensus
            current_count = direction_counts.get(direction, 0)
            consensus_ratio = current_count / len(history)
            
            if consensus_ratio < self.consensus_threshold:
                return None  # Filter out signal lacking consensus
        
        # Add smoothing information to signal
        signal['signal_smoother'] = {
            'consensus_ratio': consensus_ratio if 'consensus_ratio' in locals() else 1.0,
            'threshold': self.consensus_threshold,
            'window_size': len(history),
            'processor': self.get_name()
        }
        
        return signal
    
    def get_name(self) -> str:
        return "signal_smoother"
    
    def reset(self) -> None:
        """Reset signal history"""
        self.signal_history.clear()

class SignalQualityAnalyzer(SignalProcessor):
    """Analyzes signal quality across multiple dimensions"""
    
    def __init__(self):
        self.quality_metrics = [
            'strength', 'consistency', 'timeliness', 
            'alignment', 'anomaly_score'
        ]
        self.metric_weights = {
            'strength': 0.3,
            'consistency': 0.2,
            'timeliness': 0.2,
            'alignment': 0.2,
            'anomaly_score': 0.1
        }
    
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze signal quality across multiple dimensions"""
        quality_metrics = {}
        
        # Calculate individual quality metrics
        quality_metrics['strength'] = self._calculate_strength(signal)
        quality_metrics['consistency'] = self._calculate_consistency(signal)
        quality_metrics['timeliness'] = self._calculate_timeliness(signal)
        quality_metrics['alignment'] = self._calculate_market_alignment(signal)
        quality_metrics['anomaly_score'] = self._calculate_anomaly_score(signal)
        
        # Calculate composite quality score
        composite_score = sum(
            quality_metrics[metric] * self.metric_weights[metric]
            for metric in self.quality_metrics
        )
        
        # Add quality analysis to signal
        signal['quality_analysis'] = {
            'metrics': quality_metrics,
            'composite_score': composite_score,
            'processor': self.get_name()
        }
        
        return signal
    
    def _calculate_strength(self, signal: Dict[str, Any]) -> float:
        """Calculate signal strength metric"""
        raw_strength = signal.get('strength', 1.0)
        # Normalize to 0-1 range
        return min(1.0, max(0.0, raw_strength))
    
    def _calculate_consistency(self, signal: Dict[str, Any]) -> float:
        """Calculate signal consistency metric"""
        # This would check agreement across multiple indicators/timeframes
        # Simplified implementation
        return 0.7  # Placeholder
    
    def _calculate_timeliness(self, signal: Dict[str, Any]) -> float:
        """Calculate signal timeliness metric"""
        timestamp = signal.get('timestamp', datetime.now())
        hour = timestamp.hour
        
        # Market hours scoring (simplified)
        if 8 <= hour <= 16:  # Market hours
            return 1.0
        elif 6 <= hour <= 8 or 16 <= hour <= 18:  # Pre/post market
            return 0.7
        else:  # Off hours
            return 0.3
    
    def _calculate_market_alignment(self, signal: Dict[str, Any]) -> float:
        """Calculate market alignment metric"""
        # This would assess how well signal aligns with broader market conditions
        # Simplified implementation
        return 0.8  # Placeholder
    
    def _calculate_anomaly_score(self, signal: Dict[str, Any]) -> float:
        """Calculate anomaly score (higher = more normal)"""
        # This would detect unusual signal patterns
        # Simplified implementation
        return 0.9  # Placeholder - most signals are normal
    
    def get_name(self) -> str:
        return "quality_analyzer"
```

---

## 4. Advanced Risk Management

### 4.1 Signal Processing Framework Integration

```python
class EnhancedRiskManager:
    """Risk manager with advanced signal processing capabilities"""
    
    def __init__(self, default_position_sizer: str = "fixed",
                 max_position_size: int = 1000,
                 enable_signal_processing: bool = True):
        self.default_position_sizer = default_position_sizer
        self.max_position_size = max_position_size
        self.enable_signal_processing = enable_signal_processing
        
        # Signal processing pipeline
        self.signal_pipeline: Optional[SignalProcessingPipeline] = None
        
        # Enhanced state tracking
        self.order_id_counter = 0
        self.signal_history: List[Dict[str, Any]] = []
        self.rejected_signals: List[Dict[str, Any]] = []
        
        # Risk limit composition
        self.risk_limit_registry = RiskLimitRegistry()
        self.risk_evaluator: Optional[RiskLimitEvaluator] = None
        
        # Dependencies
        self.portfolio: Optional['Portfolio'] = None
        self.regime_detector: Optional[Any] = None
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
    
    def initialize(self, context: SystemContext) -> None:
        """Initialize enhanced risk manager"""
        # Get dependencies
        self.portfolio = context.resolve("portfolio_manager")
        
        try:
            self.regime_detector = context.resolve("regime_detector")
        except:
            print("Warning: No regime detector found")
        
        # Set up signal processing pipeline
        if self.enable_signal_processing:
            self._setup_signal_pipeline()
        
        # Set up risk limits and evaluator
        self._setup_risk_limits()
        self._setup_risk_evaluator()
    
    def _setup_signal_pipeline(self) -> None:
        """Set up signal processing pipeline"""
        self.signal_pipeline = SignalProcessingPipeline("risk_signal_pipeline")
        
        # Add regime filter if regime detector available
        if self.regime_detector:
            regime_rules = {
                "low_vol_uptrend": ["BUY"],
                "low_vol_downtrend": ["SELL"],
                "high_vol_uptrend": ["BUY", "SELL"],
                "high_vol_downtrend": ["BUY", "SELL"],
                "low_vol_sideways": [],  # No trades in sideways market
                "high_vol_sideways": ["BUY", "SELL"]
            }
            regime_filter = RegimeFilter("regime_detector", regime_rules)
            self.signal_pipeline.add_processor(regime_filter)
        
        # Add confidence filter
        confidence_filter = ConfidenceFilter(min_confidence=0.6)
        self.signal_pipeline.add_processor(confidence_filter)
        
        # Add signal smoother
        signal_smoother = SignalSmoother(window_size=3, consensus_threshold=0.6)
        self.signal_pipeline.add_processor(signal_smoother)
        
        # Add quality analyzer
        quality_analyzer = SignalQualityAnalyzer()
        self.signal_pipeline.add_processor(quality_analyzer)
    
    def _setup_risk_limits(self) -> None:
        """Set up risk limits"""
        # Register standard risk limits
        self.risk_limit_registry.register(
            "max_position", 
            MaxPositionLimit(self.max_position_size, priority=100)
        )
        self.risk_limit_registry.register(
            "max_exposure",
            MaxExposureLimit(max_exposure_pct=10.0, priority=90)
        )
        self.risk_limit_registry.register(
            "max_drawdown",
            MaxDrawdownLimit(max_drawdown_pct=15.0, priority=80)
        )
    
    def _setup_risk_evaluator(self) -> None:
        """Set up risk limit evaluator"""
        composition_strategy = AllPassStrategy()  # All limits must pass
        self.risk_evaluator = RiskLimitEvaluator(
            self.risk_limit_registry,
            composition_strategy
        )
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.SIGNAL, self.on_signal)
            if self.regime_detector:
                self._events.subscribe(EventType.CLASSIFICATION, self.on_regime_change)
    
    def on_regime_change(self, event: Event) -> None:
        """Handle regime change for signal processing"""
        if self.signal_pipeline:
            classification_data = event.payload
            new_regime = classification_data['category']
            
            # Update regime filter
            for processor in self.signal_pipeline.processors:
                if isinstance(processor, RegimeFilter):
                    processor.set_current_regime(new_regime)
    
    def on_signal(self, event: Event) -> None:
        """Process enhanced signal with pipeline"""
        original_signal = event.payload
        
        try:
            # Store original signal
            self.signal_history.append({
                'timestamp': datetime.now(),
                'original_signal': original_signal.copy(),
                'processed': False
            })
            
            # Process signal through pipeline if enabled
            if self.signal_pipeline:
                processed_signal = self.signal_pipeline.process_signal(original_signal)
                
                if processed_signal is None:
                    # Signal was filtered out
                    self.rejected_signals.append({
                        'timestamp': datetime.now(),
                        'signal': original_signal,
                        'reason': 'filtered_by_pipeline'
                    })
                    return
                
                signal = processed_signal
            else:
                signal = original_signal
            
            # Continue with standard risk management
            self._process_validated_signal(signal)
            
        except Exception as e:
            print(f"Error processing signal: {e}")
            self.rejected_signals.append({
                'timestamp': datetime.now(),
                'signal': original_signal,
                'reason': f'processing_error: {str(e)}'
            })
    
    def _process_validated_signal(self, signal: Dict[str, Any]) -> None:
        """Process signal that passed pipeline validation"""
        # Standard signal processing
        if not self._validate_signal(signal):
            return
        
        # Calculate position size
        symbol = signal['symbol']
        current_position = self.portfolio.get_position(symbol).quantity
        
        # Get position sizer (could be regime-specific)
        sizer = self._get_position_sizer(symbol, signal)
        quantity = sizer.calculate_size(signal, self.portfolio, current_position)
        
        if quantity == 0:
            return
        
        # Enhanced risk limit checking
        risk_result = self.risk_evaluator.evaluate(signal, quantity, self.portfolio)
        
        if not risk_result.passed:
            # Check if we can use modified quantity
            if risk_result.modified_quantity is not None:
                quantity = risk_result.modified_quantity
            else:
                self.rejected_signals.append({
                    'timestamp': datetime.now(),
                    'signal': signal,
                    'reason': f'risk_limits: {risk_result.violations}'
                })
                return
        
        # Create and emit order
        order = self._create_enhanced_order(signal, quantity, risk_result)
        self._emit_order(order)
    
    def _get_position_sizer(self, symbol: str, signal: Dict[str, Any]) -> Any:
        """Get position sizer, potentially regime-specific"""
        # Could select different sizers based on regime, signal quality, etc.
        regime = signal.get('regime')
        quality_score = signal.get('quality_analysis', {}).get('composite_score', 0.5)
        
        # Adjust size based on quality
        base_sizer = FixedPositionSizer(self.max_position_size)
        
        # Scale size based on signal quality
        if quality_score > 0.8:
            scaled_size = int(self.max_position_size * 1.2)  # Increase size for high quality
        elif quality_score < 0.4:
            scaled_size = int(self.max_position_size * 0.5)  # Reduce size for low quality
        else:
            scaled_size = self.max_position_size
        
        return FixedPositionSizer(scaled_size)
    
    def _create_enhanced_order(self, signal: Dict[str, Any], quantity: int, 
                             risk_result: Any) -> Dict[str, Any]:
        """Create order with enhanced metadata"""
        self.order_id_counter += 1
        
        # Adjust quantity for direction
        if signal['direction'] == 'SELL':
            quantity = -quantity
        
        order = {
            'order_id': f"ORDER_{self.order_id_counter:06d}",
            'symbol': signal['symbol'],
            'quantity': quantity,
            'order_type': 'MARKET',
            'price': signal['price'],
            'timestamp': signal['timestamp'],
            'strategy': signal.get('strategy', 'unknown'),
            'signal_strength': signal.get('strength', 1.0),
            
            # Enhanced metadata
            'regime': signal.get('regime'),
            'signal_quality': signal.get('quality_analysis'),
            'confidence_score': signal.get('confidence_filter', {}).get('confidence'),
            'risk_analysis': {
                'limits_passed': risk_result.passed,
                'original_quantity': abs(quantity),
                'modified_quantity': risk_result.modified_quantity,
                'violations': risk_result.violations
            },
            'processing_chain': self._get_processing_chain_info(signal)
        }
        
        return order
    
    def _get_processing_chain_info(self, signal: Dict[str, Any]) -> List[str]:
        """Get list of processors that handled the signal"""
        processors = []
        
        if 'regime_filter' in signal:
            processors.append('regime_filter')
        if 'confidence_filter' in signal:
            processors.append('confidence_filter')
        if 'signal_smoother' in signal:
            processors.append('signal_smoother')
        if 'quality_analysis' in signal:
            processors.append('quality_analyzer')
        
        return processors
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        total_signals = len(self.signal_history)
        rejected_signals = len(self.rejected_signals)
        processed_signals = total_signals - rejected_signals
        
        stats = {
            'total_signals_received': total_signals,
            'signals_processed': processed_signals,
            'signals_rejected': rejected_signals,
            'processing_rate': processed_signals / total_signals if total_signals > 0 else 0,
            'rejection_reasons': {}
        }
        
        # Analyze rejection reasons
        for rejection in self.rejected_signals:
            reason = rejection['reason']
            stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
        
        # Add pipeline stats if available
        if self.signal_pipeline:
            stats['pipeline_stats'] = self.signal_pipeline.get_processing_stats()
        
        return stats

class RiskLimitComposition:
    """Advanced risk limit composition framework"""
    
    def __init__(self):
        pass

class RiskLimitRegistry:
    """Registry for risk limits with metadata"""
    
    def __init__(self):
        self.limits: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, limit: Any, metadata: Dict[str, Any] = None) -> None:
        """Register risk limit with metadata"""
        self.limits[name] = limit
        self.metadata[name] = metadata or {}
    
    def get_limit(self, name: str) -> Any:
        """Get risk limit by name"""
        return self.limits.get(name)
    
    def get_all_limits(self) -> List[Any]:
        """Get all registered limits"""
        return list(self.limits.values())
    
    def get_limits_by_priority(self) -> List[Any]:
        """Get limits sorted by priority"""
        return sorted(self.limits.values(), 
                     key=lambda limit: getattr(limit, 'get_priority', lambda: 100)(),
                     reverse=True)

class CompositionResult:
    """Result of risk limit composition evaluation"""
    
    def __init__(self, passed: bool, violations: List[str] = None, 
                 modified_quantity: Optional[int] = None, 
                 metadata: Dict[str, Any] = None):
        self.passed = passed
        self.violations = violations or []
        self.modified_quantity = modified_quantity
        self.metadata = metadata or {}

class AllPassStrategy:
    """Composition strategy requiring all limits to pass"""
    
    def evaluate(self, limits: List[Any], signal: Dict[str, Any], 
                quantity: int, portfolio: Any) -> CompositionResult:
        """Evaluate all limits - all must pass"""
        violations = []
        most_restrictive_quantity = quantity
        
        for limit in limits:
            passed, reason = limit.check(signal, quantity, portfolio)
            
            if not passed:
                violations.append(f"{limit.__class__.__name__}: {reason}")
                
                # Try to get modified quantity
                if hasattr(limit, 'modify_quantity'):
                    modified_qty = limit.modify_quantity(signal, quantity, portfolio)
                    if modified_qty is not None and abs(modified_qty) < abs(most_restrictive_quantity):
                        most_restrictive_quantity = modified_qty
        
        # If violations but we have a modified quantity, suggest it
        if violations and most_restrictive_quantity != quantity:
            return CompositionResult(
                passed=False,
                violations=violations,
                modified_quantity=most_restrictive_quantity
            )
        
        return CompositionResult(
            passed=len(violations) == 0,
            violations=violations
        )

class RiskLimitEvaluator:
    """Evaluates signals against risk limits using composition strategy"""
    
    def __init__(self, registry: RiskLimitRegistry, 
                 composition_strategy: Any):
        self.registry = registry
        self.composition_strategy = composition_strategy
    
    def evaluate(self, signal: Dict[str, Any], quantity: int, 
                portfolio: Any) -> CompositionResult:
        """Evaluate signal against all risk limits"""
        limits = self.registry.get_all_limits()
        return self.composition_strategy.evaluate(limits, signal, quantity, portfolio)

class MaxDrawdownLimit:
    """Enhanced drawdown limit with modification capability"""
    
    def __init__(self, max_drawdown_pct: float, priority: int = 80):
        self.max_drawdown_pct = max_drawdown_pct
        self.priority = priority
    
    def check(self, signal: Dict[str, Any], quantity: int, 
             portfolio: Any) -> tuple[bool, str]:
        """Check drawdown limit"""
        current_equity = portfolio.get_portfolio_value()
        peak_equity = getattr(portfolio, 'peak_equity', current_equity)
        
        if current_equity > peak_equity:
            portfolio.peak_equity = current_equity
            return True, ""
        
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
        
        if drawdown_pct >= self.max_drawdown_pct:
            return False, f"Drawdown {drawdown_pct:.1f}% exceeds limit {self.max_drawdown_pct:.1f}%"
        
        return True, ""
    
    def modify_quantity(self, signal: Dict[str, Any], quantity: int, 
                       portfolio: Any) -> Optional[int]:
        """Suggest modified quantity based on drawdown"""
        current_equity = portfolio.get_portfolio_value()
        peak_equity = getattr(portfolio, 'peak_equity', current_equity)
        
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
        
        if drawdown_pct >= self.max_drawdown_pct * 0.8:  # Reduce size when approaching limit
            reduction_factor = 1.0 - (drawdown_pct / self.max_drawdown_pct)
            return int(quantity * max(0.1, reduction_factor))
        
        return None
    
    def get_priority(self) -> int:
        return self.priority

class MaxExposureLimit:
    """Enhanced exposure limit with modification capability"""
    
    def __init__(self, max_exposure_pct: float, priority: int = 90):
        self.max_exposure_pct = max_exposure_pct
        self.priority = priority
    
    def check(self, signal: Dict[str, Any], quantity: int, 
             portfolio: Any) -> tuple[bool, str]:
        """Check exposure limit"""
        symbol = signal['symbol']
        price = signal['price']
        current_position = portfolio.get_position(symbol).quantity
        new_position = current_position + quantity
        
        position_value = abs(new_position * price)
        portfolio_value = portfolio.get_portfolio_value()
        
        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative"
        
        exposure_pct = (position_value / portfolio_value) * 100
        
        if exposure_pct > self.max_exposure_pct:
            return False, f"Exposure {exposure_pct:.1f}% exceeds limit {self.max_exposure_pct:.1f}%"
        
        return True, ""
    
    def modify_quantity(self, signal: Dict[str, Any], quantity: int, 
                       portfolio: Any) -> Optional[int]:
        """Suggest modified quantity to stay within exposure limit"""
        symbol = signal['symbol']
        price = signal['price']
        current_position = portfolio.get_position(symbol).quantity
        portfolio_value = portfolio.get_portfolio_value()
        
        if portfolio_value <= 0:
            return 0
        
        # Calculate max allowed position value
        max_position_value = (self.max_exposure_pct / 100.0) * portfolio_value
        max_position_quantity = int(max_position_value / price)
        
        # Account for current position
        max_additional_quantity = max_position_quantity - abs(current_position)
        
        # Return the smaller of requested quantity or max allowed
        if quantity > 0:
            return min(quantity, max(0, max_additional_quantity))
        else:
            return max(quantity, min(0, -max_additional_quantity))
    
    def get_priority(self) -> int:
        return self.priority
```

---

## 5. Advanced Execution Features

### 5.1 Execution Modes and Threading

```python
from enum import Enum
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

class ExecutionMode(Enum):
    BACKTEST_SINGLE = "backtest_single"
    BACKTEST_PARALLEL = "backtest_parallel"
    OPTIMIZATION = "optimization"
    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"
    REPLAY = "replay"

class ThreadModel(Enum):
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    PROCESS_PARALLEL = "process_parallel"
    ASYNC_SINGLE = "async_single"
    ASYNC_MULTI = "async_multi"
    MIXED = "mixed"

class ExecutionContext:
    """Manages execution environment and threading model"""
    
    # Default mappings
    DEFAULT_THREAD_MODELS = {
        ExecutionMode.BACKTEST_SINGLE: ThreadModel.SINGLE_THREADED,
        ExecutionMode.BACKTEST_PARALLEL: ThreadModel.MULTI_THREADED,
        ExecutionMode.OPTIMIZATION: ThreadModel.PROCESS_PARALLEL,
        ExecutionMode.LIVE_TRADING: ThreadModel.ASYNC_MULTI,
        ExecutionMode.PAPER_TRADING: ThreadModel.MULTI_THREADED,
        ExecutionMode.REPLAY: ThreadModel.SINGLE_THREADED,
    }
    
    _local = threading.local()
    
    def __init__(self, execution_mode: ExecutionMode, 
                 thread_model: Optional[ThreadModel] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.execution_mode = execution_mode
        self.thread_model = thread_model or self.DEFAULT_THREAD_MODELS[execution_mode]
        self.metadata = metadata or {}
        self.created_at = datetime.now()
    
    def __enter__(self):
        self.set_current(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_current()
    
    @classmethod
    def set_current(cls, context: 'ExecutionContext') -> None:
        """Set current execution context for this thread"""
        cls._local.current_context = context
    
    @classmethod
    def get_current(cls) -> Optional['ExecutionContext']:
        """Get current execution context"""
        return getattr(cls._local, 'current_context', None)
    
    @classmethod
    def clear_current(cls) -> None:
        """Clear current execution context"""
        if hasattr(cls._local, 'current_context'):
            delattr(cls._local, 'current_context')
    
    def is_single_threaded(self) -> bool:
        """Check if running in single-threaded mode"""
        return self.thread_model == ThreadModel.SINGLE_THREADED
    
    def is_multi_threaded(self) -> bool:
        """Check if running in multi-threaded mode"""
        return self.thread_model in [ThreadModel.MULTI_THREADED, ThreadModel.MIXED]
    
    def is_async(self) -> bool:
        """Check if running in async mode"""
        return self.thread_model in [ThreadModel.ASYNC_SINGLE, ThreadModel.ASYNC_MULTI]
    
    def is_live(self) -> bool:
        """Check if running in live mode"""
        return self.execution_mode in [ExecutionMode.LIVE_TRADING, ExecutionMode.PAPER_TRADING]

class ThreadPoolManager:
    """Manages thread pools based on execution context"""
    
    def __init__(self):
        self.thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.async_loops: Dict[str, asyncio.AbstractEventLoop] = {}
    
    def initialize_for_context(self, context: ExecutionContext) -> None:
        """Initialize pools based on execution context"""
        if context.thread_model == ThreadModel.SINGLE_THREADED:
            # No pools needed
            pass
        
        elif context.thread_model == ThreadModel.MULTI_THREADED:
            if context.execution_mode == ExecutionMode.BACKTEST_PARALLEL:
                # Limited pools for backtesting
                self.thread_pools['data'] = ThreadPoolExecutor(max_workers=2)
                self.thread_pools['compute'] = ThreadPoolExecutor(max_workers=4)
            
            elif context.is_live():
                # Separate pools for live trading
                self.thread_pools['market_data'] = ThreadPoolExecutor(max_workers=2)
                self.thread_pools['order_processing'] = ThreadPoolExecutor(max_workers=1)
                self.thread_pools['strategy'] = ThreadPoolExecutor(max_workers=4)
        
        elif context.thread_model == ThreadModel.PROCESS_PARALLEL:
            # For CPU-intensive optimization
            import multiprocessing
            max_workers = min(8, multiprocessing.cpu_count())
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        elif context.thread_model in [ThreadModel.ASYNC_SINGLE, ThreadModel.ASYNC_MULTI]:
            # Set up async event loops
            if context.thread_model == ThreadModel.ASYNC_MULTI:
                self.async_loops['main'] = asyncio.new_event_loop()
                self.async_loops['data'] = asyncio.new_event_loop()
                self.async_loops['orders'] = asyncio.new_event_loop()
    
    def get_executor(self, pool_name: str = 'default') -> Optional[Any]:
        """Get executor for specified pool"""
        context = ExecutionContext.get_current()
        if not context:
            return None
        
        if context.thread_model == ThreadModel.PROCESS_PARALLEL:
            return self.process_pool
        
        return self.thread_pools.get(pool_name)
    
    def shutdown(self) -> None:
        """Shutdown all pools"""
        for pool in self.thread_pools.values():
            pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Close async loops
        for loop in self.async_loops.values():
            if loop.is_running():
                loop.close()

class ThreadAffinityManager:
    """Manages CPU affinity for performance optimization"""
    
    @staticmethod
    def set_thread_affinity(thread_type: str, cpu_cores: List[int]) -> None:
        """Set CPU affinity for thread type"""
        try:
            import os
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, cpu_cores)
        except Exception as e:
            print(f"Could not set CPU affinity: {e}")
    
    @staticmethod
    def get_recommended_affinity(context: ExecutionContext) -> Dict[str, List[int]]:
        """Get recommended CPU affinity for different thread types"""
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        
        if context.is_live():
            # Dedicate cores for critical live trading tasks
            return {
                'market_data': [0, 1],  # Dedicated cores for market data
                'order_processing': [2],  # Dedicated core for orders
                'strategy': list(range(3, min(7, total_cores))),  # Multiple cores for strategy
                'background': list(range(7, total_cores))  # Remaining cores
            }
        else:
            # For backtesting, spread load evenly
            return {
                'compute': list(range(total_cores)),
                'data': list(range(min(2, total_cores)))
            }

class ThreadIsolationGuidelines:
    """Provides guidelines for component isolation"""
    
    @staticmethod
    def get_isolation_level(context: ExecutionContext, component_type: str) -> str:
        """Get recommended isolation level for component type"""
        if context.is_single_threaded():
            return 'shared'  # No isolation needed
        
        elif context.execution_mode == ExecutionMode.OPTIMIZATION:
            if component_type in ['strategy', 'portfolio', 'data_handler']:
                return 'process_isolated'  # Full process isolation
            else:
                return 'thread_isolated'
        
        elif context.is_live():
            if component_type in ['order_manager', 'portfolio']:
                return 'thread_isolated'  # Critical components isolated
            else:
                return 'shared'
        
        else:
            return 'isolated'  # Default to some isolation
    
    @staticmethod
    def should_use_locks(context: ExecutionContext, component_type: str) -> bool:
        """Determine if component should use locks"""
        if context.is_single_threaded():
            return False
        
        if context.is_live():
            return True  # Always use locks in live trading
        
        isolation_level = ThreadIsolationGuidelines.get_isolation_level(context, component_type)
        return isolation_level in ['shared', 'thread_isolated']

class ThreadSynchronizationGuidelines:
    """Provides synchronization recommendations"""
    
    @staticmethod
    def get_sync_primitive(context: ExecutionContext, component_type: str, 
                          access_pattern: str) -> str:
        """Get recommended synchronization primitive"""
        if context.is_single_threaded():
            return 'none'
        
        if access_pattern == 'read_heavy':
            return 'reader_writer_lock'
        elif access_pattern == 'write_heavy':
            return 'lock'
        elif access_pattern == 'producer_consumer':
            return 'queue'
        elif access_pattern == 'event_driven':
            return 'event'
        else:
            return 'lock'  # Default
    
    @staticmethod
    def get_locking_strategy(context: ExecutionContext, component_type: str) -> str:
        """Get recommended locking strategy"""
        if context.is_single_threaded():
            return 'none'
        
        if component_type in ['portfolio', 'order_manager']:
            return 'fine_grained_lock'  # Fine-grained locking for critical components
        elif context.is_live():
            return 'reentrant_lock'
        else:
            return 'lock'
```

### 5.2 Asynchronous Architecture

```python
import asyncio
from typing import Coroutine, Awaitable
from abc import ABC, abstractmethod

@runtime_checkable
class AsyncComponent(Protocol):
    """Protocol for async-capable components"""
    
    @abstractmethod
    async def initialize_async(self, context: SystemContext) -> None:
        """Async initialization"""
        ...
    
    @abstractmethod
    async def start_async(self) -> None:
        """Async start"""
        ...
    
    @abstractmethod
    async def stop_async(self) -> None:
        """Async stop"""
        ...

class AsyncEventBus:
    """Async event bus for high-concurrency scenarios"""
    
    def __init__(self, buffer_size: int = 1000):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events (async handler supported)"""
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    async def publish(self, event: Event) -> None:
        """Publish event to queue"""
        if self.running:
            try:
                await self.event_queue.put(event)
            except asyncio.QueueFull:
                print(f"Event queue full, dropping event: {event.event_type}")
    
    async def start(self) -> None:
        """Start async event processing"""
        self.running = True
        self.processing_task = asyncio.create_task(self._process_events())
    
    async def stop(self) -> None:
        """Stop async event processing"""
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def _process_events(self) -> None:
        """Process events from queue"""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._handle_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """Handle individual event"""
        event_type = str(event.event_type)
        
        async with self._lock:
            handlers = self.subscribers.get(event_type, []).copy()
        
        # Process handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                task = asyncio.create_task(handler(event))
            else:
                # Run sync handler in executor
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, handler, event)
            tasks.append(task)
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"Error in event handlers: {e}")

class AsyncDataHandler:
    """Async data handler for live data feeds"""
    
    def __init__(self, data_source: str):
        self.data_source = data_source
        self.websocket: Optional[Any] = None
        self.data_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
    
    async def initialize_async(self, context: SystemContext) -> None:
        """Async initialization"""
        # Set up async event subscriptions
        if hasattr(context, 'async_event_bus'):
            self._async_event_bus = context.async_event_bus
    
    async def start_async(self) -> None:
        """Start async data processing"""
        await self._connect_websocket()
        self.processing_task = asyncio.create_task(self._process_data_stream())
    
    async def stop_async(self) -> None:
        """Stop async data processing"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
    
    async def _connect_websocket(self) -> None:
        """Connect to data source websocket"""
        # Placeholder for websocket connection
        print(f"Connecting to {self.data_source}")
    
    async def _process_data_stream(self) -> None:
        """Process incoming data stream"""
        while True:
            try:
                # Simulate receiving data
                await asyncio.sleep(0.1)  # 100ms
                
                # Create bar event
                bar_data = {
                    'symbol': 'EURUSD',
                    'timestamp': datetime.now(),
                    'close': 1.1000 + (asyncio.get_event_loop().time() % 0.01),
                    'volume': 1000
                }
                
                event = Event(EventType.BAR, bar_data)
                
                # Publish to async event bus
                if hasattr(self, '_async_event_bus'):
                    await self._async_event_bus.publish(event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in data stream processing: {e}")

class AsyncBroker:
    """Async broker for live order execution"""
    
    def __init__(self, broker_api_url: str):
        self.broker_api_url = broker_api_url
        self.session: Optional[Any] = None
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.execution_task: Optional[asyncio.Task] = None
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
    
    async def setup_subscriptions_async(self) -> None:
        """Set up async event subscriptions"""
        if hasattr(self, '_async_event_bus'):
            await self._async_event_bus.subscribe(EventType.ORDER, self.on_order_async)
    
    async def on_order_async(self, event: Event) -> None:
        """Handle order event asynchronously"""
        order = event.payload
        await self.order_queue.put(order)
    
    async def start_async(self) -> None:
        """Start async order processing"""
        await self._initialize_session()
        self.execution_task = asyncio.create_task(self._process_orders())
    
    async def stop_async(self) -> None:
        """Stop async order processing"""
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
    
    async def _initialize_session(self) -> None:
        """Initialize HTTP session for broker API"""
        import aiohttp
        self.session = aiohttp.ClientSession()
    
    async def _process_orders(self) -> None:
        """Process orders from queue"""
        while True:
            try:
                order = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                await self._execute_order_async(order)
                self.order_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing order: {e}")
    
    async def _execute_order_async(self, order: Dict[str, Any]) -> None:
        """Execute order asynchronously"""
        try:
            # Simulate API call
            await asyncio.sleep(0.1)  # Simulate network latency
            
            # Create fill event
            fill = {
                'fill_id': f"FILL_{order['order_id']}",
                'order_id': order['order_id'],
                'symbol': order['symbol'],
                'quantity': order['quantity'],
                'price': order['price'],
                'commission': 2.0,
                'timestamp': datetime.now()
            }
            
            # Publish fill event
            if hasattr(self, '_async_event_bus'):
                event = Event(EventType.FILL, fill)
                await self._async_event_bus.publish(event)
                
        except Exception as e:
            print(f"Error executing order {order['order_id']}: {e}")

class AsyncBootstrap:
    """Bootstrap for async-capable systems"""
    
    def __init__(self, config_path: str):
        self.config = Config.load(config_path)
        self.factory = ComponentFactory(self.config)
        self.components: Dict[str, Any] = {}
        self.async_components: List[Any] = []
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def initialize_async_system(self) -> SystemContext:
        """Initialize async system"""
        # Create async event bus
        async_event_bus = AsyncEventBus()
        
        # Create system context
        context = SystemContext(
            config=self.config,
            event_bus=async_event_bus,  # Use async event bus
            container=Container(),
            logger=StructuredLogger("async_system"),
            run_mode=RunMode.LIVE_TRADING
        )
        
        # Build components
        await self._build_async_components(context)
        
        return context
    
    async def _build_async_components(self, context: SystemContext) -> None:
        """Build and initialize async components"""
        component_specs = self.config.get('components', {})
        
        for name, spec in component_specs.items():
            component = self.factory.create_component(spec)
            self.components[name] = component
            
            # Initialize async components
            if isinstance(component, AsyncComponent):
                await component.initialize_async(context)
                self.async_components.append(component)
                
                # Set up async subscriptions
                if hasattr(component, 'setup_subscriptions_async'):
                    await component.setup_subscriptions_async()
    
    async def start_async_system(self) -> None:
        """Start all async components"""
        # Start async event bus
        if hasattr(self, 'async_event_bus'):
            await self.async_event_bus.start()
        
        # Start all async components
        for component in self.async_components:
            if hasattr(component, 'start_async'):
                await component.start_async()
    
    async def stop_async_system(self) -> None:
        """Stop all async components"""
        # Stop components in reverse order
        for component in reversed(self.async_components):
            if hasattr(component, 'stop_async'):
                await component.stop_async()
        
        # Stop async event bus
        if hasattr(self, 'async_event_bus'):
            await self.async_event_bus.stop()

class HybridExecutionManager:
    """Manages hybrid sync/async execution based on context"""
    
    def __init__(self):
        self.sync_bootstrap: Optional[Bootstrap] = None
        self.async_bootstrap: Optional[AsyncBootstrap] = None
        self.current_mode: Optional[ExecutionMode] = None
    
    async def initialize_for_mode(self, execution_mode: ExecutionMode, 
                                 config_path: str) -> SystemContext:
        """Initialize system for specific execution mode"""
        self.current_mode = execution_mode
        
        if execution_mode in [ExecutionMode.LIVE_TRADING, ExecutionMode.PAPER_TRADING]:
            # Use async architecture for live trading
            self.async_bootstrap = AsyncBootstrap(config_path)
            return await self.async_bootstrap.initialize_async_system()
        
        else:
            # Use sync architecture for backtesting/optimization
            self.sync_bootstrap = Bootstrap(config_path)
            return self.sync_bootstrap.initialize()
    
    async def start_system(self) -> None:
        """Start system based on current mode"""
        if self.async_bootstrap:
            await self.async_bootstrap.start_async_system()
        elif self.sync_bootstrap:
            self.sync_bootstrap.start_system()
    
    async def stop_system(self) -> None:
        """Stop system based on current mode"""
        if self.async_bootstrap:
            await self.async_bootstrap.stop_async_system()
        elif self.sync_bootstrap:
            self.sync_bootstrap.stop_system()
```

---

This completes the advanced features documentation. The document now covers:

1. **Advanced Event System** - Context isolation, scalability, partitioning, lock-free queues, tracing
2. **Advanced Data Management** - Multiple isolation strategies, memory optimization, data quality validation
3. **Advanced Strategy Components** - MetaComponent framework, regime detection, adaptive strategies, boundary trade handling, signal processing pipelines
4. **Advanced Risk Management** - Signal processing integration, risk limit composition, enhanced position tracking
5. **Advanced Execution** - Execution modes, threading models, asynchronous architecture

Combined with the first document, this provides comprehensive coverage equivalent to your original ADMF-Trader documentation, but architected using the Protocol + Composition approach. This gives you a complete blueprint for implementing a modern, flexible, and highly capable algorithmic trading system.
