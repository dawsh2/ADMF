# Strategic Caching

## Overview

This document outlines the design for strategic caching mechanisms in the ADMF-Trader system. These mechanisms optimize performance for computation-heavy operations by intelligently caching and reusing results.

## Problem Statement

Several components in the ADMF-Trader system perform computationally expensive operations that may be redundantly executed:

1. **Repeated Calculations**: Some operations (e.g., indicators, position metrics, portfolio statistics) are recalculated frequently with the same inputs

2. **Incremental Updates**: Many calculations could be updated incrementally rather than being fully recomputed (e.g., moving averages, position values)

3. **Performance Bottlenecks**: Certain code paths consume disproportionate computational resources, becoming performance bottlenecks during backtesting and optimization

We need a strategic caching approach that:
- Identifies and optimizes computation-heavy code paths
- Provides a consistent framework for caching decisions
- Balances memory usage with performance gains
- Ensures cache correctness through proper invalidation

## Design Solution

### 1. Caching Decorator Framework

The foundation of our strategic caching solution is a flexible decorator framework:

```python
import functools
import time
import weakref
import inspect
from typing import Dict, Any, Optional, Callable, Tuple, List, Union, Set
from enum import Enum, auto

class CacheStrategy(Enum):
    """Cache strategy options."""
    LRU = auto()        # Least Recently Used
    LFU = auto()        # Least Frequently Used
    TLRU = auto()       # Time-aware Least Recently Used
    FIFO = auto()       # First In, First Out
    UNBOUNDED = auto()  # No eviction policy

class CacheKey:
    """Cache key generator and manipulator."""
    
    @staticmethod
    def generate_key(args, kwargs, include_self=True):
        """
        Generate a cache key from function arguments.
        
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            include_self: Whether to include self/cls in key
            
        Returns:
            Hashable key
        """
        if not include_self and args and inspect.isclass(args[0]):
            args = args[1:]
            
        # Convert args to hashable representation
        arg_key = tuple(CacheKey._make_hashable(arg) for arg in args)
        
        # Convert kwargs to hashable representation
        kwarg_items = sorted(kwargs.items())
        kwarg_key = tuple((k, CacheKey._make_hashable(v)) for k, v in kwarg_items)
        
        # Combine keys
        return (arg_key, kwarg_key)
        
    @staticmethod
    def _make_hashable(obj):
        """
        Convert an object to a hashable representation.
        
        Args:
            obj: Object to convert
            
        Returns:
            Hashable representation
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return tuple(CacheKey._make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, CacheKey._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, set):
            return frozenset(CacheKey._make_hashable(item) for item in obj)
        elif hasattr(obj, 'cache_key'):
            # Allow objects to define their own cache key
            return obj.cache_key()
        else:
            # Fall back to object ID for non-hashable objects
            return id(obj)

class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, value, args=None, kwargs=None):
        """
        Initialize cache entry.
        
        Args:
            value: Cached value
            args: Arguments used to generate value
            kwargs: Keyword arguments used to generate value
        """
        self.value = value
        self.args = args
        self.kwargs = kwargs
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        
    def access(self):
        """Record access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def age(self):
        """Get age of entry in seconds."""
        return time.time() - self.created_at
        
    def idle_time(self):
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed

class Cache:
    """Cache implementation with configurable strategy."""
    
    def __init__(self, max_size=128, strategy=CacheStrategy.LRU, ttl=None):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            strategy: Cache eviction strategy
            ttl: Time-to-live in seconds (None for no TTL)
        """
        self.max_size = max_size
        self.strategy = strategy
        self.ttl = ttl
        self.entries = {}  # key -> CacheEntry
        self.hits = 0
        self.misses = 0
        
    def get(self, key, default=None):
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        # Check if key exists
        if key in self.entries:
            entry = self.entries[key]
            
            # Check TTL
            if self.ttl is not None and entry.age() > self.ttl:
                # Entry expired
                del self.entries[key]
                self.misses += 1
                return default
                
            # Update metadata
            entry.access()
            self.hits += 1
            
            return entry.value
        else:
            self.misses += 1
            return default
            
    def set(self, key, value, args=None, kwargs=None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            args: Arguments used to generate value
            kwargs: Keyword arguments used to generate value
            
        Returns:
            None
        """
        # Check if we need to evict an entry
        if len(self.entries) >= self.max_size and self.max_size > 0 and key not in self.entries:
            self._evict_entry()
            
        # Add new entry
        self.entries[key] = CacheEntry(value, args, kwargs)
        
    def invalidate(self, key=None, pattern=None):
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate
            pattern: Pattern to match for invalidation
            
        Returns:
            int: Number of invalidated entries
        """
        if key is not None:
            # Invalidate specific key
            if key in self.entries:
                del self.entries[key]
                return 1
            return 0
            
        elif pattern is not None:
            # Invalidate by pattern
            to_remove = []
            for k in self.entries.keys():
                if self._matches_pattern(k, pattern):
                    to_remove.append(k)
                    
            for k in to_remove:
                del self.entries[k]
                
            return len(to_remove)
            
        else:
            # Invalidate all
            count = len(self.entries)
            self.entries.clear()
            return count
            
    def _evict_entry(self):
        """Evict an entry based on strategy."""
        if not self.entries:
            return
            
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key_to_evict = min(self.entries.items(), key=lambda x: x[1].last_accessed)[0]
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key_to_evict = min(self.entries.items(), key=lambda x: x[1].access_count)[0]
        elif self.strategy == CacheStrategy.TLRU:
            # Time-aware LRU - consider both age and access time
            key_to_evict = min(
                self.entries.items(),
                key=lambda x: (x[1].access_count / max(1, x[1].age()), x[1].last_accessed)
            )[0]
        elif self.strategy == CacheStrategy.FIFO:
            # Evict oldest entry
            key_to_evict = min(self.entries.items(), key=lambda x: x[1].created_at)[0]
        elif self.strategy == CacheStrategy.UNBOUNDED:
            # No eviction
            return
        else:
            # Default to LRU
            key_to_evict = min(self.entries.items(), key=lambda x: x[1].last_accessed)[0]
            
        del self.entries[key_to_evict]
        
    def _matches_pattern(self, key, pattern):
        """Check if key matches pattern."""
        # Pattern matching could be implemented in various ways
        # Simple implementation: check if pattern is a prefix of key's string representation
        return str(key).startswith(str(pattern))
        
    def stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        return {
            'size': len(self.entries),
            'max_size': self.max_size,
            'strategy': self.strategy.name,
            'ttl': self.ttl,
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hits / max(1, self.hits + self.misses)
        }
        
    def clear(self):
        """Clear all cache entries."""
        self.entries.clear()
        
    def keys(self):
        """Get cache keys."""
        return list(self.entries.keys())

def cached(max_size=128, strategy=CacheStrategy.LRU, ttl=None, key_maker=None):
    """
    Decorator for caching function results.
    
    Args:
        max_size: Maximum cache size
        strategy: Cache eviction strategy
        ttl: Time-to-live in seconds
        key_maker: Custom function to generate cache keys
        
    Returns:
        Decorated function
    """
    cache = Cache(max_size=max_size, strategy=strategy, ttl=ttl)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_maker:
                key = key_maker(*args, **kwargs)
            else:
                key = CacheKey.generate_key(args, kwargs)
                
            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
                
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, args, kwargs)
            
            return result
            
        # Attach cache to function for management
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.invalidate_cache = cache.invalidate
        wrapper.cache_stats = cache.stats
        
        return wrapper
        
    return decorator
```

### 2. Incremental Calculation Framework

For calculations that can be updated incrementally:

```python
class IncrementalCalculator:
    """Base class for incremental calculations."""
    
    def __init__(self):
        """Initialize incremental calculator."""
        self._state = {}
        self._last_inputs = {}
        self._last_result = None
        
    def calculate(self, *args, **kwargs):
        """
        Calculate result, using incremental update if possible.
        
        Returns:
            Calculation result
        """
        # Check if we can do an incremental update
        if self._can_update_incrementally(*args, **kwargs):
            # Update incrementally
            self._last_result = self._update_incrementally(*args, **kwargs)
        else:
            # Calculate from scratch
            self._last_result = self._calculate_full(*args, **kwargs)
            
        # Update last inputs
        self._last_inputs = {
            'args': args,
            'kwargs': kwargs
        }
        
        return self._last_result
        
    def _can_update_incrementally(self, *args, **kwargs):
        """
        Check if incremental update is possible.
        
        Returns:
            bool: Whether incremental update is possible
        """
        # No previous calculation
        if not self._last_inputs:
            return False
            
        # Implementation-specific logic
        return False
        
    def _update_incrementally(self, *args, **kwargs):
        """
        Update calculation incrementally.
        
        Returns:
            Updated result
        """
        raise NotImplementedError
        
    def _calculate_full(self, *args, **kwargs):
        """
        Calculate result from scratch.
        
        Returns:
            Full calculation result
        """
        raise NotImplementedError
        
    def reset(self):
        """Reset calculator state."""
        self._state = {}
        self._last_inputs = {}
        self._last_result = None


class MovingAverageCalculator(IncrementalCalculator):
    """Incremental calculator for moving averages."""
    
    def __init__(self, window_size):
        """
        Initialize moving average calculator.
        
        Args:
            window_size: Size of moving window
        """
        super().__init__()
        self._window_size = window_size
        self._values = []
        self._sum = 0.0
        
    def _can_update_incrementally(self, new_value):
        """Check if we can update incrementally."""
        # We can update incrementally if we have previous values
        return len(self._values) > 0
        
    def _update_incrementally(self, new_value):
        """Update moving average incrementally."""
        # Add new value
        self._values.append(new_value)
        self._sum += new_value
        
        # Remove oldest value if window is full
        if len(self._values) > self._window_size:
            old_value = self._values.pop(0)
            self._sum -= old_value
            
        # Calculate average
        return self._sum / len(self._values)
        
    def _calculate_full(self, new_value):
        """Calculate moving average from scratch."""
        # Reset state
        self._values = [new_value]
        self._sum = new_value
        
        # Calculate average
        return new_value
        
    def reset(self):
        """Reset calculator."""
        super().reset()
        self._values = []
        self._sum = 0.0
```

### 3. Strategy and Indicator Caching

Efficient caching for strategy calculations and indicators:

```python
class CachedIndicator:
    """Base class for cached indicators."""
    
    def __init__(self, name):
        """
        Initialize cached indicator.
        
        Args:
            name: Indicator name
        """
        self.name = name
        self._cache = {}  # timestamp -> value
        self._last_timestamp = None
        self._last_value = None
        
    def calculate(self, data, timestamp=None):
        """
        Calculate indicator value with caching.
        
        Args:
            data: Price data
            timestamp: Optional timestamp for lookup
            
        Returns:
            Indicator value
        """
        # Use current timestamp as default
        if timestamp is None:
            timestamp = data.index[-1]
            
        # Check cache
        if timestamp in self._cache:
            self._last_timestamp = timestamp
            self._last_value = self._cache[timestamp]
            return self._last_value
            
        # Calculate value
        value = self._calculate(data, timestamp)
        
        # Cache value
        self._cache[timestamp] = value
        self._last_timestamp = timestamp
        self._last_value = value
        
        return value
        
    def _calculate(self, data, timestamp):
        """
        Calculate indicator value.
        
        Args:
            data: Price data
            timestamp: Timestamp for calculation
            
        Returns:
            Indicator value
        """
        raise NotImplementedError
        
    def invalidate(self, timestamp=None):
        """
        Invalidate cached values.
        
        Args:
            timestamp: Specific timestamp to invalidate (None for all)
        """
        if timestamp is None:
            self._cache.clear()
        elif timestamp in self._cache:
            del self._cache[timestamp]
            
    def clear_cache(self):
        """Clear indicator cache."""
        self._cache.clear()
        self._last_timestamp = None
        self._last_value = None


class CachedMovingAverage(CachedIndicator):
    """Cached moving average indicator."""
    
    def __init__(self, window_size, column='close'):
        """
        Initialize moving average indicator.
        
        Args:
            window_size: Window size for moving average
            column: Column to calculate average for
        """
        super().__init__(f"MA{window_size}")
        self.window_size = window_size
        self.column = column
        
    def _calculate(self, data, timestamp):
        """Calculate moving average."""
        # Find position of timestamp
        timestamp_idx = data.index.get_loc(timestamp)
        
        # Get window of data
        start_idx = max(0, timestamp_idx - self.window_size + 1)
        window_data = data.iloc[start_idx:timestamp_idx + 1]
        
        # Calculate average
        if len(window_data) < self.window_size:
            # Not enough data
            return None
            
        return window_data[self.column].mean()
```

### 4. Position Tracking with Incremental Updates

Incremental position value tracking:

```python
class IncrementalPositionTracker:
    """Position tracker with incremental updates."""
    
    def __init__(self):
        """Initialize position tracker."""
        self._positions = {}  # symbol -> quantity
        self._prices = {}  # symbol -> price
        self._values = {}  # symbol -> value
        self._total_value = 0.0
        
    def update_position(self, symbol, quantity_change, price):
        """
        Update position with incremental calculation.
        
        Args:
            symbol: Symbol to update
            quantity_change: Change in quantity
            price: Current price
            
        Returns:
            Updated position value
        """
        # Get current position
        current_quantity = self._positions.get(symbol, 0.0)
        current_value = self._values.get(symbol, 0.0)
        
        # Calculate new position
        new_quantity = current_quantity + quantity_change
        
        # Update position
        self._positions[symbol] = new_quantity
        self._prices[symbol] = price
        
        # Calculate new value
        new_value = new_quantity * price
        value_change = new_value - current_value
        
        # Update values
        self._values[symbol] = new_value
        self._total_value += value_change
        
        return new_value
        
    def update_price(self, symbol, price):
        """
        Update price with incremental calculation.
        
        Args:
            symbol: Symbol to update
            price: New price
            
        Returns:
            Updated position value
        """
        # Get current position
        current_quantity = self._positions.get(symbol, 0.0)
        current_price = self._prices.get(symbol, price)
        current_value = self._values.get(symbol, 0.0)
        
        # Skip if price unchanged
        if price == current_price:
            return current_value
            
        # Update price
        self._prices[symbol] = price
        
        # Calculate new value
        new_value = current_quantity * price
        value_change = new_value - current_value
        
        # Update values
        self._values[symbol] = new_value
        self._total_value += value_change
        
        return new_value
        
    def get_position_value(self, symbol):
        """
        Get position value.
        
        Args:
            symbol: Symbol to get value for
            
        Returns:
            Position value
        """
        return self._values.get(symbol, 0.0)
        
    def get_total_value(self):
        """
        Get total portfolio value.
        
        Returns:
            Total value
        """
        return self._total_value
        
    def reset(self):
        """Reset position tracker."""
        self._positions.clear()
        self._prices.clear()
        self._values.clear()
        self._total_value = 0.0
```

### 5. Caching Decorators for Strategy Methods

Specialized caching for strategy methods:

```python
def cached_indicator(window_size=None, max_size=1000, ttl=None):
    """
    Decorator for caching indicators.
    
    Args:
        window_size: Window size for indicator (if any)
        max_size: Maximum cache size
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def custom_key_maker(self, data, *args, **kwargs):
        """
        Generate cache key for indicator.
        
        Args:
            self: Strategy instance
            data: Price data
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Cache key
        """
        # Use last timestamp of data as part of key
        last_timestamp = data.index[-1]
        
        # Include window size in key if provided
        window_key = f"window={window_size}" if window_size is not None else ""
        
        # Generate key components
        components = [
            self.__class__.__name__,
            last_timestamp,
            window_key
        ]
        
        # Add additional args and kwargs
        components.extend(args)
        components.extend(sorted(kwargs.items()))
        
        return tuple(CacheKey._make_hashable(component) for component in components)
        
    return cached(max_size=max_size, ttl=ttl, key_maker=custom_key_maker)


def cached_signal(max_size=100, ttl=None):
    """
    Decorator for caching signal generation.
    
    Args:
        max_size: Maximum cache size
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def custom_key_maker(self, data, *args, **kwargs):
        """
        Generate cache key for signal generation.
        
        Args:
            self: Strategy instance
            data: Price data
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Cache key
        """
        # Use last timestamp of data as part of key
        last_timestamp = data.index[-1]
        
        # Generate key components
        components = [
            self.__class__.__name__,
            "signal",
            last_timestamp
        ]
        
        # Add strategy parameters to key
        params = getattr(self, 'parameters', {})
        if params:
            components.append(tuple(sorted(params.items())))
            
        return tuple(CacheKey._make_hashable(component) for component in components)
        
    return cached(max_size=max_size, ttl=ttl, key_maker=custom_key_maker)
```

### 6. Cache Management System

A central cache management system for monitoring and control:

```python
class CacheManager:
    """Centralized cache management system."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        """Initialize cache manager."""
        self._caches = {}  # name -> Cache
        self._calculators = {}  # name -> IncrementalCalculator
        self._cached_functions = {}  # name -> decorated function
        self._enabled = True
        
    def register_cache(self, name, cache):
        """
        Register a cache.
        
        Args:
            name: Cache name
            cache: Cache instance
        """
        self._caches[name] = cache
        
    def register_calculator(self, name, calculator):
        """
        Register an incremental calculator.
        
        Args:
            name: Calculator name
            calculator: Calculator instance
        """
        self._calculators[name] = calculator
        
    def register_cached_function(self, name, func):
        """
        Register a cached function.
        
        Args:
            name: Function name
            func: Decorated function
        """
        self._cached_functions[name] = func
        
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        stats = {
            'caches': {},
            'total_hits': 0,
            'total_misses': 0,
            'total_entries': 0
        }
        
        # Collect cache stats
        for name, cache in self._caches.items():
            cache_stats = cache.stats()
            stats['caches'][name] = cache_stats
            stats['total_hits'] += cache_stats['hits']
            stats['total_misses'] += cache_stats['misses']
            stats['total_entries'] += cache_stats['size']
            
        # Collect cached function stats
        for name, func in self._cached_functions.items():
            if hasattr(func, 'cache_stats'):
                func_stats = func.cache_stats()
                stats['caches'][name] = func_stats
                stats['total_hits'] += func_stats['hits']
                stats['total_misses'] += func_stats['misses']
                stats['total_entries'] += func_stats['size']
                
        # Calculate overall hit ratio
        total_requests = stats['total_hits'] + stats['total_misses']
        stats['hit_ratio'] = stats['total_hits'] / max(1, total_requests)
        
        return stats
        
    def clear_all(self):
        """Clear all caches."""
        # Clear registered caches
        for cache in self._caches.values():
            cache.clear()
            
        # Clear cached function caches
        for func in self._cached_functions.values():
            if hasattr(func, 'clear_cache'):
                func.clear_cache()
                
        # Reset calculators
        for calculator in self._calculators.values():
            calculator.reset()
            
    def invalidate_by_pattern(self, pattern):
        """
        Invalidate cache entries by pattern.
        
        Args:
            pattern: Pattern to match for invalidation
            
        Returns:
            int: Number of invalidated entries
        """
        count = 0
        
        # Invalidate in registered caches
        for cache in self._caches.values():
            count += cache.invalidate(pattern=pattern)
            
        # Invalidate in cached function caches
        for func in self._cached_functions.values():
            if hasattr(func, 'invalidate_cache'):
                count += func.invalidate_cache(pattern=pattern)
                
        return count
        
    def enable(self):
        """Enable caching."""
        self._enabled = True
        
    def disable(self):
        """Disable caching."""
        self._enabled = False
        
    @property
    def enabled(self):
        """Get enabled state."""
        return self._enabled
```

## Performance-Critical Components

We've identified several performance-critical components that will benefit from strategic caching:

### 1. Indicator Calculation

Technical indicators are frequently recalculated with the same inputs:

```python
class CachingIndicatorManager:
    """Manager for cached indicators."""
    
    def __init__(self):
        """Initialize indicator manager."""
        self._indicators = {}  # name -> CachedIndicator
        self._cache_manager = CacheManager.get_instance()
        
    def get_indicator(self, indicator_type, *args, **kwargs):
        """
        Get indicator with caching.
        
        Args:
            indicator_type: Type of indicator
            *args: Arguments for indicator
            **kwargs: Keyword arguments for indicator
            
        Returns:
            Cached indicator instance
        """
        # Generate indicator name
        name = self._generate_indicator_name(indicator_type, *args, **kwargs)
        
        # Check if indicator exists
        if name not in self._indicators:
            # Create indicator
            indicator = self._create_indicator(indicator_type, *args, **kwargs)
            self._indicators[name] = indicator
            
            # Register with cache manager
            self._cache_manager.register_cache(name, indicator)
            
        return self._indicators[name]
        
    def _generate_indicator_name(self, indicator_type, *args, **kwargs):
        """Generate indicator name."""
        if indicator_type == 'MA':
            window_size = args[0] if args else kwargs.get('window_size')
            column = kwargs.get('column', 'close')
            return f"MA({window_size},{column})"
        elif indicator_type == 'RSI':
            window_size = args[0] if args else kwargs.get('window_size')
            return f"RSI({window_size})"
        elif indicator_type == 'BBANDS':
            window_size = args[0] if args else kwargs.get('window_size')
            num_std = args[1] if len(args) > 1 else kwargs.get('num_std', 2)
            return f"BBANDS({window_size},{num_std})"
        else:
            return f"{indicator_type}({args},{kwargs})"
            
    def _create_indicator(self, indicator_type, *args, **kwargs):
        """Create indicator instance."""
        if indicator_type == 'MA':
            window_size = args[0] if args else kwargs.get('window_size')
            column = kwargs.get('column', 'close')
            return CachedMovingAverage(window_size, column)
        elif indicator_type == 'RSI':
            window_size = args[0] if args else kwargs.get('window_size')
            return CachedRSI(window_size)
        elif indicator_type == 'BBANDS':
            window_size = args[0] if args else kwargs.get('window_size')
            num_std = args[1] if len(args) > 1 else kwargs.get('num_std', 2)
            return CachedBollingerBands(window_size, num_std)
        else:
            raise ValueError(f"Unsupported indicator type: {indicator_type}")
            
    def clear_caches(self):
        """Clear all indicator caches."""
        for indicator in self._indicators.values():
            indicator.clear_cache()
            
    def invalidate_timestamp(self, timestamp):
        """
        Invalidate cached values for a timestamp.
        
        Args:
            timestamp: Timestamp to invalidate
        """
        for indicator in self._indicators.values():
            indicator.invalidate(timestamp)
```

### 2. Portfolio Valuation

Portfolio valuation is frequently performed with minimal changes:

```python
class CachingPortfolio:
    """Portfolio implementation with caching."""
    
    def __init__(self, initial_cash=100000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Initial cash amount
        """
        self._cash = initial_cash
        self._positions = {}  # symbol -> quantity
        self._position_tracker = IncrementalPositionTracker()
        self._equity_curve = []
        self._trades = []
        
        # Cache key for position value calculation
        self._last_valuation_time = None
        self._cached_portfolio_value = None
        self._price_cache = {}  # symbol -> (timestamp, price)
        
    def update_position(self, symbol, quantity_change, price, timestamp):
        """
        Update position.
        
        Args:
            symbol: Symbol to update
            quantity_change: Change in quantity
            price: Current price
            timestamp: Update timestamp
        """
        # Update position tracker
        position_value = self._position_tracker.update_position(symbol, quantity_change, price)
        
        # Update cash
        self._cash -= quantity_change * price
        
        # Update current position
        current_quantity = self._positions.get(symbol, 0.0)
        new_quantity = current_quantity + quantity_change
        
        if new_quantity == 0:
            # Position closed
            if symbol in self._positions:
                del self._positions[symbol]
        else:
            # Position updated
            self._positions[symbol] = new_quantity
            
        # Record trade
        if quantity_change != 0:
            trade = {
                'symbol': symbol,
                'quantity': quantity_change,
                'price': price,
                'timestamp': timestamp
            }
            self._trades.append(trade)
            
        # Update price cache
        self._price_cache[symbol] = (timestamp, price)
        
        # Invalidate portfolio value cache
        self._last_valuation_time = None
        self._cached_portfolio_value = None
        
        return position_value
        
    def update_price(self, symbol, price, timestamp):
        """
        Update price.
        
        Args:
            symbol: Symbol to update
            price: New price
            timestamp: Update timestamp
        """
        # Skip if no position
        if symbol not in self._positions and symbol not in self._position_tracker._positions:
            return
            
        # Skip if price unchanged
        if symbol in self._price_cache:
            cache_timestamp, cache_price = self._price_cache[symbol]
            if price == cache_price and timestamp <= cache_timestamp:
                return
                
        # Update position tracker
        self._position_tracker.update_price(symbol, price)
        
        # Update price cache
        self._price_cache[symbol] = (timestamp, price)
        
        # Invalidate portfolio value cache
        self._last_valuation_time = None
        self._cached_portfolio_value = None
        
    @cached(max_size=1, strategy=CacheStrategy.TLRU)
    def get_portfolio_value(self, timestamp=None):
        """
        Get portfolio value.
        
        Args:
            timestamp: Valuation timestamp
            
        Returns:
            Portfolio value
        """
        # Check if we have a cached value with same timestamp
        if timestamp == self._last_valuation_time and self._cached_portfolio_value is not None:
            return self._cached_portfolio_value
            
        # Use position tracker for position values
        total_position_value = self._position_tracker.get_total_value()
        
        # Add cash to get portfolio value
        portfolio_value = self._cash + total_position_value
        
        # Cache value
        self._last_valuation_time = timestamp
        self._cached_portfolio_value = portfolio_value
        
        return portfolio_value
        
    def get_position(self, symbol):
        """
        Get position.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position quantity
        """
        return self._positions.get(symbol, 0.0)
        
    def get_position_value(self, symbol):
        """
        Get position value.
        
        Args:
            symbol: Symbol to get value for
            
        Returns:
            Position value
        """
        return self._position_tracker.get_position_value(symbol)
        
    def update_equity_curve(self, timestamp):
        """
        Update equity curve.
        
        Args:
            timestamp: Update timestamp
        """
        portfolio_value = self.get_portfolio_value(timestamp)
        
        point = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self._cash
        }
        
        self._equity_curve.append(point)
        
    def get_equity_curve(self):
        """
        Get equity curve.
        
        Returns:
            List of equity curve points
        """
        return self._equity_curve
        
    def get_trades(self):
        """
        Get trades.
        
        Returns:
            List of trades
        """
        return self._trades
        
    def reset(self):
        """Reset portfolio."""
        self._positions.clear()
        self._position_tracker.reset()
        self._equity_curve.clear()
        self._trades.clear()
        self._price_cache.clear()
        self._last_valuation_time = None
        self._cached_portfolio_value = None
        
        # Reset cash to initial value
        initial_cash = self._cash  # Store initial cash value
        self._cash = initial_cash
```

### 3. Risk Metrics Calculation

Risk metrics are computationally expensive but change infrequently:

```python
class CachingRiskCalculator:
    """Risk metrics calculator with caching."""
    
    def __init__(self):
        """Initialize risk calculator."""
        self._equity_curve = None
        self._metrics_cache = {}
        self._last_calculation_size = 0
        
    @cached(max_size=1, strategy=CacheStrategy.TLRU)
    def calculate_metrics(self, equity_curve):
        """
        Calculate risk metrics with caching.
        
        Args:
            equity_curve: Equity curve for calculation
            
        Returns:
            Dict with risk metrics
        """
        # Check if we can update incrementally
        if (self._equity_curve is not None and 
            len(self._equity_curve) < len(equity_curve) and
            self._last_calculation_size > 0):
            # Try incremental update
            return self._update_incrementally(equity_curve)
            
        # Calculate from scratch
        returns = self._calculate_returns(equity_curve)
        
        metrics = {
            'total_return': self._calculate_total_return(equity_curve),
            'annualized_return': self._calculate_annualized_return(equity_curve, returns),
            'volatility': self._calculate_volatility(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': self._calculate_win_rate(returns),
            'profit_factor': self._calculate_profit_factor(returns)
        }
        
        # Cache calculation
        self._equity_curve = equity_curve.copy()
        self._metrics_cache = metrics.copy()
        self._last_calculation_size = len(equity_curve)
        
        return metrics
        
    def _update_incrementally(self, equity_curve):
        """
        Update metrics incrementally.
        
        Args:
            equity_curve: New equity curve
            
        Returns:
            Dict with updated metrics
        """
        # Extract new data points
        new_points = equity_curve[self._last_calculation_size:]
        
        # Calculate returns for new points
        new_returns = self._calculate_returns(
            [self._equity_curve[-1]] + new_points
        )[1:]  # Skip first return which compares to last cached point
        
        # Combine with existing returns
        all_returns = self._calculate_returns(self._equity_curve) + new_returns
        
        # Update metrics
        metrics = {
            'total_return': self._calculate_total_return(equity_curve),
            'annualized_return': self._calculate_annualized_return(equity_curve, all_returns),
            'volatility': self._calculate_volatility(all_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(all_returns),
            'max_drawdown': self._update_max_drawdown(
                self._metrics_cache['max_drawdown'],
                self._equity_curve,
                new_points
            ),
            'win_rate': self._update_win_rate(
                self._metrics_cache['win_rate'],
                len(all_returns) - len(new_returns),
                new_returns
            ),
            'profit_factor': self._update_profit_factor(
                self._metrics_cache['profit_factor'],
                all_returns,
                new_returns
            )
        }
        
        # Cache calculation
        self._equity_curve = equity_curve.copy()
        self._metrics_cache = metrics.copy()
        self._last_calculation_size = len(equity_curve)
        
        return metrics
        
    def _calculate_returns(self, equity_curve):
        """Calculate returns from equity curve."""
        returns = []
        
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1]['portfolio_value']
            curr_value = equity_curve[i]['portfolio_value']
            
            if prev_value > 0:
                ret = (curr_value / prev_value) - 1.0
            else:
                ret = 0.0
                
            returns.append(ret)
            
        return returns
        
    def _calculate_total_return(self, equity_curve):
        """Calculate total return."""
        if len(equity_curve) < 2:
            return 0.0
            
        start_value = equity_curve[0]['portfolio_value']
        end_value = equity_curve[-1]['portfolio_value']
        
        if start_value > 0:
            return (end_value / start_value) - 1.0
        else:
            return 0.0
            
    def _calculate_annualized_return(self, equity_curve, returns):
        """Calculate annualized return."""
        if len(equity_curve) < 2:
            return 0.0
            
        # Calculate days between start and end
        start_date = equity_curve[0]['timestamp']
        end_date = equity_curve[-1]['timestamp']
        days = (end_date - start_date).days
        
        if days < 1:
            days = 1
            
        # Calculate annualized return
        total_return = self._calculate_total_return(equity_curve)
        return ((1 + total_return) ** (365 / days)) - 1.0
        
    def _calculate_volatility(self, returns):
        """Calculate return volatility."""
        if not returns:
            return 0.0
            
        return np.std(returns) * np.sqrt(252)  # Annualized
        
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio."""
        if not returns:
            return 0.0
            
        excess_returns = [r - risk_free_rate / 252 for r in returns]  # Daily risk-free rate
        
        avg_excess_return = np.mean(excess_returns)
        std_dev = np.std(excess_returns)
        
        if std_dev == 0:
            return 0.0
            
        sharpe = avg_excess_return / std_dev * np.sqrt(252)  # Annualized
        
        return sharpe
        
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
            
        values = [point['portfolio_value'] for point in equity_curve]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
                
        return max_dd
        
    def _update_max_drawdown(self, current_max_dd, old_curve, new_points):
        """Update maximum drawdown incrementally."""
        if not new_points:
            return current_max_dd
            
        # Get portfolio values
        old_values = [point['portfolio_value'] for point in old_curve]
        new_values = [point['portfolio_value'] for point in new_points]
        all_values = old_values + new_values
        
        # Find peak up to current point
        peak = max(old_values)
        
        # Check for new peak and drawdowns in new data
        for value in new_values:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0.0
                current_max_dd = max(current_max_dd, dd)
                
        return current_max_dd
        
    def _calculate_win_rate(self, returns):
        """Calculate win rate."""
        if not returns:
            return 0.0
            
        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)
        
    def _update_win_rate(self, current_win_rate, old_count, new_returns):
        """Update win rate incrementally."""
        if not new_returns:
            return current_win_rate
            
        # Calculate wins in new returns
        new_wins = sum(1 for r in new_returns if r > 0)
        
        # Calculate old wins from win rate
        old_wins = int(current_win_rate * old_count)
        
        # Calculate new win rate
        total_count = old_count + len(new_returns)
        total_wins = old_wins + new_wins
        
        return total_wins / total_count
        
    def _calculate_profit_factor(self, returns):
        """Calculate profit factor."""
        if not returns:
            return 0.0
            
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = sum(abs(r) for r in returns if r < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
        
    def _update_profit_factor(self, current_profit_factor, old_returns, new_returns):
        """
        Update profit factor incrementally.
        
        Note: This is an approximation as we don't store old returns.
        A more accurate approach would store gross profit/loss directly.
        """
        # Calculate from all returns for accuracy
        return self._calculate_profit_factor(old_returns)
        
    def clear_cache(self):
        """Clear calculation cache."""
        self._equity_curve = None
        self._metrics_cache = {}
        self._last_calculation_size = 0
```

## Implementation Strategy

### 1. Core Implementation

1. Implement caching decorator framework:
   - `cached` decorator
   - `Cache` class with eviction strategies
   - `CacheKey` utility

2. Implement incremental calculation framework:
   - `IncrementalCalculator` base class
   - Specialized calculator implementations

3. Implement cache management:
   - `CacheManager` for centralized control
   - Cache statistics gathering

### 2. Component Integration

1. Integrate caching into indicators:
   - Implement `CachedIndicator` base class
   - Create specialized cached indicators

2. Enhance portfolio with caching:
   - Implement `IncrementalPositionTracker`
   - Add caching to portfolio valuation

3. Optimize risk calculations:
   - Implement `CachingRiskCalculator`
   - Add incremental updates for risk metrics

### 3. Testing and Benchmarking

1. Create benchmark suite:
   - Measure performance with and without caching
   - Compare different caching strategies
   - Test memory usage impact

2. Verify calculation correctness:
   - Ensure cached and non-cached results match
   - Test cache invalidation scenarios
   - Verify incremental updates

## Best Practices

### 1. Caching Guidelines

- **Cache Selectively**: Cache only computationally expensive operations:
  ```python
  # Good candidate for caching (expensive calculation)
  @cached(max_size=100)
  def calculate_correlation_matrix(self, returns):
      return returns.corr()
      
  # Poor candidate for caching (simple operation)
  def get_position_direction(self, symbol):
      quantity = self.positions.get(symbol, 0)
      return 'LONG' if quantity > 0 else 'SHORT' if quantity < 0 else 'FLAT'
  ```

- **Consider Memory Impact**: Balance performance gain with memory usage:
  ```python
  # Limit cache size for memory-intensive calculations
  @cached(max_size=10, strategy=CacheStrategy.LRU)
  def calculate_large_covariance_matrix(self, returns):
      return returns.cov()
  ```

- **Set TTL for Time-Sensitive Data**: Use time-to-live for dynamic data:
  ```python
  # Cache market data with TTL
  @cached(ttl=60)  # 60 seconds TTL
  def get_market_data(self, symbol):
      return self.api.get_market_data(symbol)
  ```

### 2. Incremental Calculation Guidelines

- **Track Input Changes**: Carefully track input changes for incremental updates:
  ```python
  def _can_update_incrementally(self, *args, **kwargs):
      # Check if only new data points are added
      new_data = args[0]
      if (self._last_data is not None and
          len(self._last_data) < len(new_data) and
          all(self._last_data[i] == new_data[i] for i in range(len(self._last_data)))):
          return True
      return False
  ```

- **Validate Incremental Results**: Periodically verify incremental calculations:
  ```python
  def calculate(self, *args, **kwargs):
      result = super().calculate(*args, **kwargs)
      
      # Periodically verify against full calculation
      if random.random() < 0.01:  # 1% chance
          full_result = self._calculate_full(*args, **kwargs)
          if not np.isclose(result, full_result, rtol=1e-5):
              # Log warning and use full calculation
              logging.warning("Incremental calculation drift detected")
              result = full_result
              
      return result
  ```

### 3. Cache Invalidation Guidelines

- **Invalidate Proactively**: Invalidate caches when inputs change significantly:
  ```python
  def update_strategy_parameters(self, new_params):
      self.parameters.update(new_params)
      
      # Invalidate affected caches
      if hasattr(self.calculate_signals, 'invalidate_cache'):
          self.calculate_signals.invalidate_cache()
  ```

- **Use Patterns for Related Invalidation**: Invalidate related caches together:
  ```python
  def update_price(self, symbol, price):
      self._price_cache[symbol] = price
      
      # Invalidate all caches related to this symbol
      CacheManager.get_instance().invalidate_by_pattern(symbol)
  ```

- **Clear All on Reset**: Clear all caches when resetting state:
  ```python
  def reset(self):
      super().reset()
      
      # Clear caches
      CacheManager.get_instance().clear_all()
  ```

## Conclusion

The strategic caching mechanisms presented in this document provide a robust framework for optimizing performance-critical operations in the ADMF-Trader system. By implementing caching decorators, incremental calculations, and a centralized cache management system, we can significantly improve performance while keeping memory usage under control.

The approach targets specific high-impact areas like indicator calculation, portfolio valuation, and risk metrics, providing tailored caching solutions for each. The implementation is designed to be flexible and configurable, allowing users to adjust caching behavior based on their specific requirements and constraints.

These improvements address the performance bottlenecks identified in the current system, making it more efficient for both backtesting and optimization scenarios.