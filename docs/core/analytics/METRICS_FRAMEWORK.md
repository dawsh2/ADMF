# Metrics Framework

## Overview

The Metrics Framework provides a comprehensive system for collecting, calculating, and analyzing key performance indicators (KPIs) across the ADMF-Trader system. This framework enables consistent measurement of trading strategies, system performance, and operational efficiency.

## Problem Statement

Trading systems require robust metrics to evaluate performance across multiple dimensions:

1. **Strategy Performance Metrics**: Traders need accurate measurements of returns, risk, and various performance ratios to evaluate trading strategies

2. **System Performance Metrics**: Engineers need to monitor execution speed, resource utilization, and system efficiency

3. **Operational Metrics**: Operations teams need to track reliability, uptime, and error rates

4. **Consistency and Standardization**: Different parts of the system should use consistent calculation methodologies to ensure comparability

5. **Real-time vs. Historical Analysis**: Metrics must support both real-time monitoring and historical analysis

## Design Solution

### 1. Core Metrics Framework

The foundation of the metrics system is a flexible, extensible framework:

```python
import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Any, Callable, Optional, Union, Type
from enum import Enum, auto
from datetime import datetime, timedelta

class MetricType(Enum):
    """Types of metrics supported by the framework."""
    COUNTER = auto()          # Monotonically increasing counter
    GAUGE = auto()            # Value that can go up and down
    HISTOGRAM = auto()        # Distribution of values
    TIMER = auto()            # Duration measurements
    RATIO = auto()            # Ratio between two metrics
    COMPOSITE = auto()        # Metric composed of multiple others
    
class MetricDimension(Enum):
    """Dimensions for categorizing metrics."""
    PERFORMANCE = auto()      # Strategy performance metrics
    RISK = auto()             # Risk-related metrics
    SYSTEM = auto()           # System performance metrics
    OPERATIONAL = auto()      # Operational metrics
    CUSTOM = auto()           # User-defined metrics

class MetricPeriod(Enum):
    """Time periods for calculating metrics."""
    TICK = auto()             # Per-tick metrics
    MINUTE = auto()           # Per-minute metrics
    HOUR = auto()             # Hourly metrics
    DAY = auto()              # Daily metrics
    WEEK = auto()             # Weekly metrics
    MONTH = auto()            # Monthly metrics
    YEAR = auto()             # Yearly metrics
    CUSTOM = auto()           # Custom time period
    ALL = auto()              # All-time metrics

class MetricValue:
    """Value container for metrics with metadata."""
    
    def __init__(self, value, timestamp=None, metadata=None):
        """
        Initialize metric value.
        
        Args:
            value: Metric value
            timestamp: Optional timestamp
            metadata: Optional metadata
        """
        self.value = value
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        
    def __float__(self):
        """Convert to float."""
        return float(self.value)
        
    def __int__(self):
        """Convert to int."""
        return int(self.value)
        
    def __str__(self):
        """Convert to string."""
        return str(self.value)
        
    def __repr__(self):
        """Object representation."""
        return f"MetricValue({self.value}, {self.timestamp}, {self.metadata})"
        
    def as_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'value': self.value,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

class Metric:
    """Base class for all metrics."""
    
    def __init__(self, name, description=None, 
               metric_type=MetricType.GAUGE,
               dimension=MetricDimension.CUSTOM,
               period=MetricPeriod.ALL,
               unit=None,
               tags=None):
        """
        Initialize metric.
        
        Args:
            name: Metric name
            description: Optional description
            metric_type: Metric type
            dimension: Metric dimension
            period: Metric period
            unit: Optional unit of measurement
            tags: Optional tags for categorization
        """
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.dimension = dimension
        self.period = period
        self.unit = unit
        self.tags = tags or {}
        self.values = []
        self._lock = threading.RLock()
        
    def record(self, value, timestamp=None, metadata=None):
        """
        Record metric value.
        
        Args:
            value: Value to record
            timestamp: Optional timestamp
            metadata: Optional metadata
            
        Returns:
            Recorded MetricValue
        """
        metric_value = MetricValue(value, timestamp, metadata)
        
        with self._lock:
            self.values.append(metric_value)
            self._on_new_value(metric_value)
            
        return metric_value
        
    def _on_new_value(self, value):
        """
        Handle new metric value.
        
        Args:
            value: New MetricValue
        """
        # Base implementation does nothing
        pass
        
    def get_latest(self):
        """
        Get latest metric value.
        
        Returns:
            Latest MetricValue or None
        """
        with self._lock:
            if not self.values:
                return None
            return self.values[-1]
            
    def get_values(self, start_time=None, end_time=None):
        """
        Get metric values within time range.
        
        Args:
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            List of MetricValues
        """
        with self._lock:
            if not self.values:
                return []
                
            if start_time is None and end_time is None:
                return self.values.copy()
                
            filtered_values = []
            for value in self.values:
                if start_time is not None and value.timestamp < start_time:
                    continue
                if end_time is not None and value.timestamp > end_time:
                    continue
                filtered_values.append(value)
                
            return filtered_values
            
    def get_value_at(self, timestamp):
        """
        Get metric value at specific time.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Closest MetricValue or None
        """
        with self._lock:
            if not self.values:
                return None
                
            # Find closest value
            closest_value = None
            min_diff = float('inf')
            
            for value in self.values:
                diff = abs(value.timestamp - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_value = value
                    
            return closest_value
            
    def reset(self):
        """Reset metric state."""
        with self._lock:
            self.values = []
            
    def as_series(self):
        """
        Convert to pandas Series.
        
        Returns:
            pandas Series with timestamps as index
        """
        with self._lock:
            if not self.values:
                return pd.Series()
                
            data = [v.value for v in self.values]
            index = [datetime.fromtimestamp(v.timestamp) for v in self.values]
            
            return pd.Series(data, index=index, name=self.name)
            
    def as_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        latest = self.get_latest()
        
        result = {
            'name': self.name,
            'description': self.description,
            'type': self.metric_type.name,
            'dimension': self.dimension.name,
            'period': self.period.name,
            'unit': self.unit,
            'tags': self.tags,
            'latest': latest.as_dict() if latest else None
        }
        
        return result

class CounterMetric(Metric):
    """Monotonically increasing counter metric."""
    
    def __init__(self, name, description=None, 
               dimension=MetricDimension.CUSTOM,
               period=MetricPeriod.ALL,
               unit=None,
               tags=None):
        """Initialize counter metric."""
        super().__init__(
            name=name,
            description=description,
            metric_type=MetricType.COUNTER,
            dimension=dimension,
            period=period,
            unit=unit,
            tags=tags
        )
        self._count = 0
        
    def increment(self, amount=1, timestamp=None, metadata=None):
        """
        Increment counter.
        
        Args:
            amount: Amount to increment
            timestamp: Optional timestamp
            metadata: Optional metadata
            
        Returns:
            New counter value
        """
        with self._lock:
            self._count += amount
            return self.record(self._count, timestamp, metadata)
            
    def get_count(self):
        """
        Get current count.
        
        Returns:
            Current count value
        """
        with self._lock:
            return self._count
            
    def reset(self):
        """Reset counter."""
        with self._lock:
            super().reset()
            self._count = 0

class GaugeMetric(Metric):
    """Gauge metric that can go up and down."""
    
    def __init__(self, name, description=None, 
               dimension=MetricDimension.CUSTOM,
               period=MetricPeriod.ALL,
               unit=None,
               tags=None):
        """Initialize gauge metric."""
        super().__init__(
            name=name,
            description=description,
            metric_type=MetricType.GAUGE,
            dimension=dimension,
            period=period,
            unit=unit,
            tags=tags
        )
        
    def set(self, value, timestamp=None, metadata=None):
        """
        Set gauge value.
        
        Args:
            value: New value
            timestamp: Optional timestamp
            metadata: Optional metadata
            
        Returns:
            Recorded MetricValue
        """
        return self.record(value, timestamp, metadata)
        
    def get_average(self, start_time=None, end_time=None):
        """
        Get average value within time range.
        
        Args:
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Average value or None
        """
        values = self.get_values(start_time, end_time)
        if not values:
            return None
            
        return sum(float(v.value) for v in values) / len(values)
        
    def get_min(self, start_time=None, end_time=None):
        """
        Get minimum value within time range.
        
        Args:
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Minimum value or None
        """
        values = self.get_values(start_time, end_time)
        if not values:
            return None
            
        return min(float(v.value) for v in values)
        
    def get_max(self, start_time=None, end_time=None):
        """
        Get maximum value within time range.
        
        Args:
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Maximum value or None
        """
        values = self.get_values(start_time, end_time)
        if not values:
            return None
            
        return max(float(v.value) for v in values)

class HistogramMetric(Metric):
    """Histogram metric for value distributions."""
    
    def __init__(self, name, description=None, 
               dimension=MetricDimension.CUSTOM,
               period=MetricPeriod.ALL,
               unit=None,
               tags=None,
               buckets=None):
        """
        Initialize histogram metric.
        
        Args:
            name: Metric name
            description: Optional description
            dimension: Metric dimension
            period: Metric period
            unit: Optional unit of measurement
            tags: Optional tags for categorization
            buckets: Optional histogram buckets
        """
        super().__init__(
            name=name,
            description=description,
            metric_type=MetricType.HISTOGRAM,
            dimension=dimension,
            period=period,
            unit=unit,
            tags=tags
        )
        self.buckets = buckets or [0, 10, 100, 1000, float('inf')]
        self._bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._count = 0
        self._sum = 0
        
    def observe(self, value, timestamp=None, metadata=None):
        """
        Observe value for histogram.
        
        Args:
            value: Value to observe
            timestamp: Optional timestamp
            metadata: Optional metadata
            
        Returns:
            Recorded MetricValue
        """
        with self._lock:
            self._count += 1
            self._sum += value
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
                    
            return self.record(value, timestamp, metadata)
            
    def get_count(self):
        """
        Get observation count.
        
        Returns:
            Number of observations
        """
        with self._lock:
            return self._count
            
    def get_sum(self):
        """
        Get sum of observed values.
        
        Returns:
            Sum of observed values
        """
        with self._lock:
            return self._sum
            
    def get_average(self):
        """
        Get average of observed values.
        
        Returns:
            Average value or 0
        """
        with self._lock:
            if self._count == 0:
                return 0
            return self._sum / self._count
            
    def get_bucket_counts(self):
        """
        Get bucket counts.
        
        Returns:
            Dictionary with bucket counts
        """
        with self._lock:
            return self._bucket_counts.copy()
            
    def get_percentile(self, p):
        """
        Get approximate percentile.
        
        Args:
            p: Percentile (0-100)
            
        Returns:
            Approximate percentile value
        """
        with self._lock:
            if not self.values:
                return None
                
            # Sort values for percentile calculation
            sorted_values = sorted(float(v.value) for v in self.values)
            
            # Calculate percentile index
            idx = int(len(sorted_values) * p / 100)
            
            # Return percentile value
            return sorted_values[idx]
            
    def reset(self):
        """Reset histogram."""
        with self._lock:
            super().reset()
            self._bucket_counts = {bucket: 0 for bucket in self.buckets}
            self._count = 0
            self._sum = 0
            
    def as_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = super().as_dict()
        
        with self._lock:
            result.update({
                'count': self._count,
                'sum': self._sum,
                'avg': self.get_average(),
                'buckets': self.get_bucket_counts()
            })
            
        return result

class TimerMetric(Metric):
    """Timer metric for duration measurements."""
    
    def __init__(self, name, description=None, 
               dimension=MetricDimension.CUSTOM,
               period=MetricPeriod.ALL,
               tags=None):
        """Initialize timer metric."""
        super().__init__(
            name=name,
            description=description,
            metric_type=MetricType.TIMER,
            dimension=dimension,
            period=period,
            unit="seconds",
            tags=tags
        )
        self._histogram = HistogramMetric(
            name=f"{name}_histogram",
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0, float('inf')]
        )
        
    def start(self):
        """
        Start timing.
        
        Returns:
            Timer context manager
        """
        return TimerContext(self)
        
    def record_duration(self, duration, timestamp=None, metadata=None):
        """
        Record duration manually.
        
        Args:
            duration: Duration in seconds
            timestamp: Optional timestamp
            metadata: Optional metadata
            
        Returns:
            Recorded MetricValue
        """
        self._histogram.observe(duration)
        return self.record(duration, timestamp, metadata)
        
    def get_count(self):
        """
        Get observation count.
        
        Returns:
            Number of observations
        """
        return self._histogram.get_count()
        
    def get_average_duration(self):
        """
        Get average duration.
        
        Returns:
            Average duration
        """
        return self._histogram.get_average()
        
    def get_percentile(self, p):
        """
        Get duration percentile.
        
        Args:
            p: Percentile (0-100)
            
        Returns:
            Percentile duration
        """
        return self._histogram.get_percentile(p)
        
    def reset(self):
        """Reset timer."""
        with self._lock:
            super().reset()
            self._histogram.reset()
            
    def as_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = super().as_dict()
        
        result.update({
            'count': self.get_count(),
            'avg_duration': self.get_average_duration(),
            'p50': self.get_percentile(50),
            'p90': self.get_percentile(90),
            'p99': self.get_percentile(99)
        })
        
        return result

class TimerContext:
    """Context manager for timer metrics."""
    
    def __init__(self, timer):
        """
        Initialize timer context.
        
        Args:
            timer: Timer metric
        """
        self.timer = timer
        self.start_time = None
        self.metadata = {}
        
    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            # Add exception info to metadata if available
            if exc_type is not None:
                self.metadata['exception'] = f"{exc_type.__name__}: {exc_val}"
                
            self.timer.record_duration(duration, metadata=self.metadata)
            
    def add_metadata(self, key, value):
        """
        Add metadata to timer.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

class MetricsRegistry:
    """Central registry for metrics."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        """Initialize metrics registry."""
        self._metrics = {}
        self._lock = threading.RLock()
        
    def create_counter(self, name, description=None, 
                     dimension=MetricDimension.CUSTOM,
                     period=MetricPeriod.ALL,
                     unit=None,
                     tags=None):
        """
        Create or get counter metric.
        
        Args:
            name: Metric name
            description: Optional description
            dimension: Metric dimension
            period: Metric period
            unit: Optional unit of measurement
            tags: Optional tags
            
        Returns:
            CounterMetric
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, CounterMetric):
                    return metric
                else:
                    raise ValueError(f"Metric '{name}' already exists with different type")
                    
            metric = CounterMetric(
                name=name,
                description=description,
                dimension=dimension,
                period=period,
                unit=unit,
                tags=tags
            )
            
            self._metrics[name] = metric
            return metric
            
    def create_gauge(self, name, description=None, 
                   dimension=MetricDimension.CUSTOM,
                   period=MetricPeriod.ALL,
                   unit=None,
                   tags=None):
        """
        Create or get gauge metric.
        
        Args:
            name: Metric name
            description: Optional description
            dimension: Metric dimension
            period: Metric period
            unit: Optional unit of measurement
            tags: Optional tags
            
        Returns:
            GaugeMetric
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, GaugeMetric):
                    return metric
                else:
                    raise ValueError(f"Metric '{name}' already exists with different type")
                    
            metric = GaugeMetric(
                name=name,
                description=description,
                dimension=dimension,
                period=period,
                unit=unit,
                tags=tags
            )
            
            self._metrics[name] = metric
            return metric
            
    def create_histogram(self, name, description=None, 
                       dimension=MetricDimension.CUSTOM,
                       period=MetricPeriod.ALL,
                       unit=None,
                       tags=None,
                       buckets=None):
        """
        Create or get histogram metric.
        
        Args:
            name: Metric name
            description: Optional description
            dimension: Metric dimension
            period: Metric period
            unit: Optional unit of measurement
            tags: Optional tags
            buckets: Optional histogram buckets
            
        Returns:
            HistogramMetric
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, HistogramMetric):
                    return metric
                else:
                    raise ValueError(f"Metric '{name}' already exists with different type")
                    
            metric = HistogramMetric(
                name=name,
                description=description,
                dimension=dimension,
                period=period,
                unit=unit,
                tags=tags,
                buckets=buckets
            )
            
            self._metrics[name] = metric
            return metric
            
    def create_timer(self, name, description=None, 
                   dimension=MetricDimension.CUSTOM,
                   period=MetricPeriod.ALL,
                   tags=None):
        """
        Create or get timer metric.
        
        Args:
            name: Metric name
            description: Optional description
            dimension: Metric dimension
            period: Metric period
            tags: Optional tags
            
        Returns:
            TimerMetric
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, TimerMetric):
                    return metric
                else:
                    raise ValueError(f"Metric '{name}' already exists with different type")
                    
            metric = TimerMetric(
                name=name,
                description=description,
                dimension=dimension,
                period=period,
                tags=tags
            )
            
            self._metrics[name] = metric
            return metric
            
    def get_metric(self, name):
        """
        Get metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric or None
        """
        with self._lock:
            return self._metrics.get(name)
            
    def get_metrics_by_dimension(self, dimension):
        """
        Get metrics by dimension.
        
        Args:
            dimension: Metric dimension
            
        Returns:
            List of metrics
        """
        with self._lock:
            return [m for m in self._metrics.values() if m.dimension == dimension]
            
    def get_metrics_by_tags(self, tags):
        """
        Get metrics by tags.
        
        Args:
            tags: Dictionary of tags to match
            
        Returns:
            List of metrics
        """
        with self._lock:
            result = []
            
            for metric in self._metrics.values():
                match = True
                
                for key, value in tags.items():
                    if key not in metric.tags or metric.tags[key] != value:
                        match = False
                        break
                        
                if match:
                    result.append(metric)
                    
            return result
            
    def get_all_metrics(self):
        """
        Get all metrics.
        
        Returns:
            List of all metrics
        """
        with self._lock:
            return list(self._metrics.values())
            
    def reset_all_metrics(self):
        """Reset all metrics."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()
                
    def remove_metric(self, name):
        """
        Remove metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
            return False
            
    def as_dict(self):
        """
        Convert all metrics to dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            return {name: metric.as_dict() for name, metric in self._metrics.items()}
```

### 2. Strategy Performance Metrics

Specialized metrics for evaluating trading strategy performance:

```python
class StrategyMetrics:
    """Metrics for trading strategy performance."""
    
    def __init__(self, strategy_id, strategy_name=None):
        """
        Initialize strategy metrics.
        
        Args:
            strategy_id: Strategy identifier
            strategy_name: Optional strategy name
        """
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name or strategy_id
        
        # Create metrics registry
        self._registry = MetricsRegistry.get_instance()
        
        # Common tags for all strategy metrics
        self._tags = {
            'strategy_id': strategy_id,
            'strategy_name': self.strategy_name
        }
        
        # Initialize metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize strategy metrics."""
        # Performance metrics
        self.total_return = self._registry.create_gauge(
            name=f"{self.strategy_id}.total_return",
            description="Total return",
            dimension=MetricDimension.PERFORMANCE,
            unit="percent",
            tags=self._tags
        )
        
        self.daily_returns = self._registry.create_histogram(
            name=f"{self.strategy_id}.daily_returns",
            description="Daily returns distribution",
            dimension=MetricDimension.PERFORMANCE,
            period=MetricPeriod.DAY,
            unit="percent",
            tags=self._tags,
            buckets=[-10, -5, -3, -1, 0, 1, 3, 5, 10, float('inf')]
        )
        
        self.annualized_return = self._registry.create_gauge(
            name=f"{self.strategy_id}.annualized_return",
            description="Annualized return",
            dimension=MetricDimension.PERFORMANCE,
            unit="percent",
            tags=self._tags
        )
        
        # Risk metrics
        self.volatility = self._registry.create_gauge(
            name=f"{self.strategy_id}.volatility",
            description="Return volatility",
            dimension=MetricDimension.RISK,
            unit="percent",
            tags=self._tags
        )
        
        self.sharpe_ratio = self._registry.create_gauge(
            name=f"{self.strategy_id}.sharpe_ratio",
            description="Sharpe ratio",
            dimension=MetricDimension.RISK,
            tags=self._tags
        )
        
        self.max_drawdown = self._registry.create_gauge(
            name=f"{self.strategy_id}.max_drawdown",
            description="Maximum drawdown",
            dimension=MetricDimension.RISK,
            unit="percent",
            tags=self._tags
        )
        
        # Trading metrics
        self.trade_count = self._registry.create_counter(
            name=f"{self.strategy_id}.trade_count",
            description="Number of trades",
            dimension=MetricDimension.PERFORMANCE,
            tags=self._tags
        )
        
        self.win_rate = self._registry.create_gauge(
            name=f"{self.strategy_id}.win_rate",
            description="Win rate",
            dimension=MetricDimension.PERFORMANCE,
            unit="percent",
            tags=self._tags
        )
        
        self.average_trade_return = self._registry.create_gauge(
            name=f"{self.strategy_id}.average_trade_return",
            description="Average trade return",
            dimension=MetricDimension.PERFORMANCE,
            unit="percent",
            tags=self._tags
        )
        
        self.profit_factor = self._registry.create_gauge(
            name=f"{self.strategy_id}.profit_factor",
            description="Profit factor (gross profit / gross loss)",
            dimension=MetricDimension.PERFORMANCE,
            tags=self._tags
        )
        
        # Execution metrics
        self.execution_time = self._registry.create_timer(
            name=f"{self.strategy_id}.execution_time",
            description="Strategy execution time",
            dimension=MetricDimension.SYSTEM,
            tags=self._tags
        )
        
    def record_trade(self, trade):
        """
        Record trade information.
        
        Args:
            trade: Trade object with details
        """
        # Increment trade count
        self.trade_count.increment()
        
        # Record trade return
        trade_return = trade.get('return', 0)
        
        # Update average trade return
        current_count = self.trade_count.get_count()
        current_avg = self.average_trade_return.get_latest()
        
        if current_avg is None:
            new_avg = trade_return
        else:
            new_avg = ((current_avg.value * (current_count - 1)) + trade_return) / current_count
            
        self.average_trade_return.set(new_avg)
        
        # Update win rate
        win_count = self.win_rate.get_latest()
        
        if win_count is None:
            win_count = 0
        else:
            win_count = win_count.value * (current_count - 1) / 100
            
        if trade_return > 0:
            win_count += 1
            
        self.win_rate.set((win_count / current_count) * 100)
        
    def update_returns(self, total_return, daily_returns=None):
        """
        Update return metrics.
        
        Args:
            total_return: Total return percentage
            daily_returns: Optional list of daily returns
        """
        self.total_return.set(total_return)
        
        if daily_returns is not None:
            for daily_return in daily_returns:
                self.daily_returns.observe(daily_return)
                
    def update_risk_metrics(self, volatility=None, sharpe_ratio=None, max_drawdown=None):
        """
        Update risk metrics.
        
        Args:
            volatility: Return volatility
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
        """
        if volatility is not None:
            self.volatility.set(volatility)
            
        if sharpe_ratio is not None:
            self.sharpe_ratio.set(sharpe_ratio)
            
        if max_drawdown is not None:
            self.max_drawdown.set(max_drawdown)
            
    def start_execution_timer(self):
        """
        Start execution timer.
        
        Returns:
            Timer context manager
        """
        return self.execution_time.start()
        
    def get_summary(self):
        """
        Get strategy metrics summary.
        
        Returns:
            Dictionary with metrics summary
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'performance': {
                'total_return': self.total_return.get_latest().value if self.total_return.get_latest() else None,
                'annualized_return': self.annualized_return.get_latest().value if self.annualized_return.get_latest() else None,
                'trade_count': self.trade_count.get_count(),
                'win_rate': self.win_rate.get_latest().value if self.win_rate.get_latest() else None,
                'average_trade_return': self.average_trade_return.get_latest().value if self.average_trade_return.get_latest() else None,
                'profit_factor': self.profit_factor.get_latest().value if self.profit_factor.get_latest() else None
            },
            'risk': {
                'volatility': self.volatility.get_latest().value if self.volatility.get_latest() else None,
                'sharpe_ratio': self.sharpe_ratio.get_latest().value if self.sharpe_ratio.get_latest() else None,
                'max_drawdown': self.max_drawdown.get_latest().value if self.max_drawdown.get_latest() else None
            },
            'system': {
                'avg_execution_time': self.execution_time.get_average_duration(),
                'p90_execution_time': self.execution_time.get_percentile(90),
                'p99_execution_time': self.execution_time.get_percentile(99)
            }
        }
        
    def reset(self):
        """Reset all strategy metrics."""
        self.total_return.reset()
        self.daily_returns.reset()
        self.annualized_return.reset()
        self.volatility.reset()
        self.sharpe_ratio.reset()
        self.max_drawdown.reset()
        self.trade_count.reset()
        self.win_rate.reset()
        self.average_trade_return.reset()
        self.profit_factor.reset()
        self.execution_time.reset()
```

### 3. System Performance Metrics

Metrics for tracking system performance:

```python
class SystemMetrics:
    """Metrics for system performance monitoring."""
    
    def __init__(self):
        """Initialize system metrics."""
        # Create metrics registry
        self._registry = MetricsRegistry.get_instance()
        
        # Common tags for all system metrics
        self._tags = {
            'component': 'system'
        }
        
        # Initialize metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize system metrics."""
        # CPU metrics
        self.cpu_usage = self._registry.create_gauge(
            name="system.cpu_usage",
            description="CPU usage percentage",
            dimension=MetricDimension.SYSTEM,
            unit="percent",
            tags=self._tags
        )
        
        self.cpu_time = self._registry.create_timer(
            name="system.cpu_time",
            description="CPU time used by operations",
            dimension=MetricDimension.SYSTEM,
            tags=self._tags
        )
        
        # Memory metrics
        self.memory_usage = self._registry.create_gauge(
            name="system.memory_usage",
            description="Memory usage",
            dimension=MetricDimension.SYSTEM,
            unit="bytes",
            tags=self._tags
        )
        
        self.memory_usage_percent = self._registry.create_gauge(
            name="system.memory_usage_percent",
            description="Memory usage percentage",
            dimension=MetricDimension.SYSTEM,
            unit="percent",
            tags=self._tags
        )
        
        # I/O metrics
        self.disk_read_bytes = self._registry.create_counter(
            name="system.disk_read_bytes",
            description="Bytes read from disk",
            dimension=MetricDimension.SYSTEM,
            unit="bytes",
            tags=self._tags
        )
        
        self.disk_write_bytes = self._registry.create_counter(
            name="system.disk_write_bytes",
            description="Bytes written to disk",
            dimension=MetricDimension.SYSTEM,
            unit="bytes",
            tags=self._tags
        )
        
        self.disk_io_time = self._registry.create_timer(
            name="system.disk_io_time",
            description="Time spent on disk I/O operations",
            dimension=MetricDimension.SYSTEM,
            tags=self._tags
        )
        
        # Network metrics
        self.network_received_bytes = self._registry.create_counter(
            name="system.network_received_bytes",
            description="Bytes received over network",
            dimension=MetricDimension.SYSTEM,
            unit="bytes",
            tags=self._tags
        )
        
        self.network_sent_bytes = self._registry.create_counter(
            name="system.network_sent_bytes",
            description="Bytes sent over network",
            dimension=MetricDimension.SYSTEM,
            unit="bytes",
            tags=self._tags
        )
        
        self.network_request_time = self._registry.create_timer(
            name="system.network_request_time",
            description="Time spent on network requests",
            dimension=MetricDimension.SYSTEM,
            tags=self._tags
        )
        
        # Queue metrics
        self.queue_size = self._registry.create_gauge(
            name="system.queue_size",
            description="Number of items in queue",
            dimension=MetricDimension.SYSTEM,
            tags=self._tags
        )
        
        self.queue_latency = self._registry.create_timer(
            name="system.queue_latency",
            description="Time items spend in queue",
            dimension=MetricDimension.SYSTEM,
            tags=self._tags
        )
        
    def update_cpu_metrics(self, cpu_percent):
        """
        Update CPU usage metrics.
        
        Args:
            cpu_percent: CPU usage percentage
        """
        self.cpu_usage.set(cpu_percent)
        
    def start_cpu_timer(self, operation=None):
        """
        Start CPU time timer.
        
        Args:
            operation: Optional operation name
            
        Returns:
            Timer context manager
        """
        timer = self.cpu_time.start()
        if operation:
            timer.add_metadata('operation', operation)
        return timer
        
    def update_memory_metrics(self, memory_bytes, memory_percent):
        """
        Update memory usage metrics.
        
        Args:
            memory_bytes: Memory usage in bytes
            memory_percent: Memory usage percentage
        """
        self.memory_usage.set(memory_bytes)
        self.memory_usage_percent.set(memory_percent)
        
    def record_disk_read(self, bytes_read):
        """
        Record disk read operation.
        
        Args:
            bytes_read: Number of bytes read
            
        Returns:
            New counter value
        """
        return self.disk_read_bytes.increment(bytes_read)
        
    def record_disk_write(self, bytes_written):
        """
        Record disk write operation.
        
        Args:
            bytes_written: Number of bytes written
            
        Returns:
            New counter value
        """
        return self.disk_write_bytes.increment(bytes_written)
        
    def start_disk_io_timer(self, operation=None):
        """
        Start disk I/O timer.
        
        Args:
            operation: Optional operation name
            
        Returns:
            Timer context manager
        """
        timer = self.disk_io_time.start()
        if operation:
            timer.add_metadata('operation', operation)
        return timer
        
    def record_network_received(self, bytes_received):
        """
        Record network receive operation.
        
        Args:
            bytes_received: Number of bytes received
            
        Returns:
            New counter value
        """
        return self.network_received_bytes.increment(bytes_received)
        
    def record_network_sent(self, bytes_sent):
        """
        Record network send operation.
        
        Args:
            bytes_sent: Number of bytes sent
            
        Returns:
            New counter value
        """
        return self.network_sent_bytes.increment(bytes_sent)
        
    def start_network_timer(self, endpoint=None):
        """
        Start network request timer.
        
        Args:
            endpoint: Optional endpoint name
            
        Returns:
            Timer context manager
        """
        timer = self.network_request_time.start()
        if endpoint:
            timer.add_metadata('endpoint', endpoint)
        return timer
        
    def update_queue_size(self, queue_name, size):
        """
        Update queue size.
        
        Args:
            queue_name: Queue name
            size: Current queue size
            
        Returns:
            Recorded metric value
        """
        return self.queue_size.set(size, metadata={'queue': queue_name})
        
    def record_queue_latency(self, queue_name, latency):
        """
        Record queue latency.
        
        Args:
            queue_name: Queue name
            latency: Latency in seconds
            
        Returns:
            Recorded metric value
        """
        return self.queue_latency.record_duration(
            latency,
            metadata={'queue': queue_name}
        )
        
    def get_summary(self):
        """
        Get system metrics summary.
        
        Returns:
            Dictionary with metrics summary
        """
        return {
            'cpu': {
                'usage': self.cpu_usage.get_latest().value if self.cpu_usage.get_latest() else None,
                'avg_time': self.cpu_time.get_average_duration(),
                'p90_time': self.cpu_time.get_percentile(90)
            },
            'memory': {
                'usage_bytes': self.memory_usage.get_latest().value if self.memory_usage.get_latest() else None,
                'usage_percent': self.memory_usage_percent.get_latest().value if self.memory_usage_percent.get_latest() else None
            },
            'disk': {
                'read_bytes': self.disk_read_bytes.get_count(),
                'write_bytes': self.disk_write_bytes.get_count(),
                'avg_io_time': self.disk_io_time.get_average_duration()
            },
            'network': {
                'received_bytes': self.network_received_bytes.get_count(),
                'sent_bytes': self.network_sent_bytes.get_count(),
                'avg_request_time': self.network_request_time.get_average_duration(),
                'p90_request_time': self.network_request_time.get_percentile(90)
            },
            'queue': {
                'avg_latency': self.queue_latency.get_average_duration(),
                'p90_latency': self.queue_latency.get_percentile(90)
            }
        }
        
    def reset(self):
        """Reset all system metrics."""
        self.cpu_usage.reset()
        self.cpu_time.reset()
        self.memory_usage.reset()
        self.memory_usage_percent.reset()
        self.disk_read_bytes.reset()
        self.disk_write_bytes.reset()
        self.disk_io_time.reset()
        self.network_received_bytes.reset()
        self.network_sent_bytes.reset()
        self.network_request_time.reset()
        self.queue_size.reset()
        self.queue_latency.reset()
```

### 4. Operational Metrics

Metrics for tracking system reliability and operational health:

```python
class OperationalMetrics:
    """Metrics for operational monitoring."""
    
    def __init__(self):
        """Initialize operational metrics."""
        # Create metrics registry
        self._registry = MetricsRegistry.get_instance()
        
        # Common tags for all operational metrics
        self._tags = {
            'component': 'operations'
        }
        
        # Initialize metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize operational metrics."""
        # Availability metrics
        self.uptime = self._registry.create_gauge(
            name="ops.uptime",
            description="System uptime",
            dimension=MetricDimension.OPERATIONAL,
            unit="seconds",
            tags=self._tags
        )
        
        self.availability = self._registry.create_gauge(
            name="ops.availability",
            description="System availability",
            dimension=MetricDimension.OPERATIONAL,
            unit="percent",
            tags=self._tags
        )
        
        # Error metrics
        self.error_count = self._registry.create_counter(
            name="ops.error_count",
            description="Number of errors",
            dimension=MetricDimension.OPERATIONAL,
            tags=self._tags
        )
        
        self.error_rate = self._registry.create_gauge(
            name="ops.error_rate",
            description="Error rate",
            dimension=MetricDimension.OPERATIONAL,
            unit="percent",
            tags=self._tags
        )
        
        # Request metrics
        self.request_count = self._registry.create_counter(
            name="ops.request_count",
            description="Number of requests",
            dimension=MetricDimension.OPERATIONAL,
            tags=self._tags
        )
        
        self.request_duration = self._registry.create_timer(
            name="ops.request_duration",
            description="Request duration",
            dimension=MetricDimension.OPERATIONAL,
            tags=self._tags
        )
        
        # Component metrics
        self.component_health = self._registry.create_gauge(
            name="ops.component_health",
            description="Component health status",
            dimension=MetricDimension.OPERATIONAL,
            tags=self._tags
        )
        
        self.dependency_health = self._registry.create_gauge(
            name="ops.dependency_health",
            description="Dependency health status",
            dimension=MetricDimension.OPERATIONAL,
            tags=self._tags
        )
        
        # Data metrics
        self.data_latency = self._registry.create_gauge(
            name="ops.data_latency",
            description="Data latency (time between generation and processing)",
            dimension=MetricDimension.OPERATIONAL,
            unit="seconds",
            tags=self._tags
        )
        
        self.data_quality = self._registry.create_gauge(
            name="ops.data_quality",
            description="Data quality score",
            dimension=MetricDimension.OPERATIONAL,
            unit="percent",
            tags=self._tags
        )
        
    def update_uptime(self, uptime_seconds):
        """
        Update system uptime.
        
        Args:
            uptime_seconds: Uptime in seconds
            
        Returns:
            Recorded metric value
        """
        return self.uptime.set(uptime_seconds)
        
    def update_availability(self, availability_percent):
        """
        Update system availability.
        
        Args:
            availability_percent: Availability percentage
            
        Returns:
            Recorded metric value
        """
        return self.availability.set(availability_percent)
        
    def record_error(self, error_type=None, source=None):
        """
        Record error occurrence.
        
        Args:
            error_type: Optional error type
            source: Optional error source
            
        Returns:
            New error count
        """
        metadata = {}
        if error_type:
            metadata['error_type'] = error_type
        if source:
            metadata['source'] = source
            
        # Increment error count
        error_count = self.error_count.increment(metadata=metadata)
        
        # Update error rate
        request_count = self.request_count.get_count()
        
        if request_count > 0:
            error_rate = (self.error_count.get_count() / request_count) * 100
            self.error_rate.set(error_rate)
            
        return error_count
        
    def record_request(self, endpoint=None, status=None):
        """
        Record request.
        
        Args:
            endpoint: Optional endpoint
            status: Optional status code
            
        Returns:
            New request count
        """
        metadata = {}
        if endpoint:
            metadata['endpoint'] = endpoint
        if status:
            metadata['status'] = status
            
        # Increment request count
        request_count = self.request_count.increment(metadata=metadata)
        
        # Update error rate if status indicates error
        if status and status >= 400:
            error_count = self.error_count.get_count()
            error_rate = (error_count / request_count) * 100
            self.error_rate.set(error_rate)
            
        return request_count
        
    def start_request_timer(self, endpoint=None):
        """
        Start request timer.
        
        Args:
            endpoint: Optional endpoint
            
        Returns:
            Timer context manager
        """
        timer = self.request_duration.start()
        if endpoint:
            timer.add_metadata('endpoint', endpoint)
        return timer
        
    def update_component_health(self, component, health_score):
        """
        Update component health.
        
        Args:
            component: Component name
            health_score: Health score (0-100)
            
        Returns:
            Recorded metric value
        """
        return self.component_health.set(health_score, metadata={'component': component})
        
    def update_dependency_health(self, dependency, health_score):
        """
        Update dependency health.
        
        Args:
            dependency: Dependency name
            health_score: Health score (0-100)
            
        Returns:
            Recorded metric value
        """
        return self.dependency_health.set(health_score, metadata={'dependency': dependency})
        
    def update_data_latency(self, data_source, latency_seconds):
        """
        Update data latency.
        
        Args:
            data_source: Data source name
            latency_seconds: Latency in seconds
            
        Returns:
            Recorded metric value
        """
        return self.data_latency.set(latency_seconds, metadata={'source': data_source})
        
    def update_data_quality(self, data_source, quality_score):
        """
        Update data quality.
        
        Args:
            data_source: Data source name
            quality_score: Quality score (0-100)
            
        Returns:
            Recorded metric value
        """
        return self.data_quality.set(quality_score, metadata={'source': data_source})
        
    def get_summary(self):
        """
        Get operational metrics summary.
        
        Returns:
            Dictionary with metrics summary
        """
        return {
            'availability': {
                'uptime': self.uptime.get_latest().value if self.uptime.get_latest() else None,
                'availability': self.availability.get_latest().value if self.availability.get_latest() else None
            },
            'errors': {
                'count': self.error_count.get_count(),
                'rate': self.error_rate.get_latest().value if self.error_rate.get_latest() else None
            },
            'requests': {
                'count': self.request_count.get_count(),
                'avg_duration': self.request_duration.get_average_duration(),
                'p90_duration': self.request_duration.get_percentile(90)
            },
            'health': {
                'component': self._get_component_health_summary(),
                'dependency': self._get_dependency_health_summary()
            },
            'data': {
                'latency': self._get_data_latency_summary(),
                'quality': self._get_data_quality_summary()
            }
        }
        
    def _get_component_health_summary(self):
        """
        Get component health summary.
        
        Returns:
            Dictionary with component health
        """
        result = {}
        
        # Get all component health values
        for value in self.component_health.get_values():
            if 'component' in value.metadata:
                component = value.metadata['component']
                result[component] = value.value
                
        return result
        
    def _get_dependency_health_summary(self):
        """
        Get dependency health summary.
        
        Returns:
            Dictionary with dependency health
        """
        result = {}
        
        # Get all dependency health values
        for value in self.dependency_health.get_values():
            if 'dependency' in value.metadata:
                dependency = value.metadata['dependency']
                result[dependency] = value.value
                
        return result
        
    def _get_data_latency_summary(self):
        """
        Get data latency summary.
        
        Returns:
            Dictionary with data latency
        """
        result = {}
        
        # Get all data latency values
        for value in self.data_latency.get_values():
            if 'source' in value.metadata:
                source = value.metadata['source']
                result[source] = value.value
                
        return result
        
    def _get_data_quality_summary(self):
        """
        Get data quality summary.
        
        Returns:
            Dictionary with data quality
        """
        result = {}
        
        # Get all data quality values
        for value in self.data_quality.get_values():
            if 'source' in value.metadata:
                source = value.metadata['source']
                result[source] = value.value
                
        return result
        
    def reset(self):
        """Reset all operational metrics."""
        self.uptime.reset()
        self.availability.reset()
        self.error_count.reset()
        self.error_rate.reset()
        self.request_count.reset()
        self.request_duration.reset()
        self.component_health.reset()
        self.dependency_health.reset()
        self.data_latency.reset()
        self.data_quality.reset()
```

### 5. Custom Metrics API

A flexible API for defining custom metrics:

```python
class CustomMetricsBuilder:
    """Builder for custom metrics."""
    
    def __init__(self, namespace, tags=None):
        """
        Initialize custom metrics builder.
        
        Args:
            namespace: Metrics namespace
            tags: Optional common tags
        """
        self.namespace = namespace
        self.common_tags = tags or {}
        self._registry = MetricsRegistry.get_instance()
        self.metrics = {}
        
    def counter(self, name, description=None, unit=None, tags=None):
        """
        Create counter metric.
        
        Args:
            name: Metric name
            description: Optional description
            unit: Optional unit
            tags: Optional tags
            
        Returns:
            CounterMetric
        """
        full_name = f"{self.namespace}.{name}"
        merged_tags = self._merge_tags(tags)
        
        metric = self._registry.create_counter(
            name=full_name,
            description=description,
            dimension=MetricDimension.CUSTOM,
            unit=unit,
            tags=merged_tags
        )
        
        self.metrics[name] = metric
        return metric
        
    def gauge(self, name, description=None, unit=None, tags=None):
        """
        Create gauge metric.
        
        Args:
            name: Metric name
            description: Optional description
            unit: Optional unit
            tags: Optional tags
            
        Returns:
            GaugeMetric
        """
        full_name = f"{self.namespace}.{name}"
        merged_tags = self._merge_tags(tags)
        
        metric = self._registry.create_gauge(
            name=full_name,
            description=description,
            dimension=MetricDimension.CUSTOM,
            unit=unit,
            tags=merged_tags
        )
        
        self.metrics[name] = metric
        return metric
        
    def histogram(self, name, description=None, unit=None, tags=None, buckets=None):
        """
        Create histogram metric.
        
        Args:
            name: Metric name
            description: Optional description
            unit: Optional unit
            tags: Optional tags
            buckets: Optional histogram buckets
            
        Returns:
            HistogramMetric
        """
        full_name = f"{self.namespace}.{name}"
        merged_tags = self._merge_tags(tags)
        
        metric = self._registry.create_histogram(
            name=full_name,
            description=description,
            dimension=MetricDimension.CUSTOM,
            unit=unit,
            tags=merged_tags,
            buckets=buckets
        )
        
        self.metrics[name] = metric
        return metric
        
    def timer(self, name, description=None, tags=None):
        """
        Create timer metric.
        
        Args:
            name: Metric name
            description: Optional description
            tags: Optional tags
            
        Returns:
            TimerMetric
        """
        full_name = f"{self.namespace}.{name}"
        merged_tags = self._merge_tags(tags)
        
        metric = self._registry.create_timer(
            name=full_name,
            description=description,
            dimension=MetricDimension.CUSTOM,
            tags=merged_tags
        )
        
        self.metrics[name] = metric
        return metric
        
    def get_metric(self, name):
        """
        Get metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric or None
        """
        return self.metrics.get(name)
        
    def _merge_tags(self, tags):
        """
        Merge common tags with specific tags.
        
        Args:
            tags: Specific tags
            
        Returns:
            Merged tags dictionary
        """
        result = self.common_tags.copy()
        if tags:
            result.update(tags)
        return result

class CompositeMetric:
    """Metric composed of multiple other metrics."""
    
    def __init__(self, name, description=None, 
               dimension=MetricDimension.CUSTOM,
               period=MetricPeriod.ALL,
               unit=None,
               tags=None):
        """
        Initialize composite metric.
        
        Args:
            name: Metric name
            description: Optional description
            dimension: Metric dimension
            period: Metric period
            unit: Optional unit of measurement
            tags: Optional tags
        """
        self.name = name
        self.description = description
        self.dimension = dimension
        self.period = period
        self.unit = unit
        self.tags = tags or {}
        self._metrics = {}
        self._registry = MetricsRegistry.get_instance()
        
    def add_metric(self, name, metric):
        """
        Add metric to composite.
        
        Args:
            name: Metric name within composite
            metric: Metric instance
            
        Returns:
            Self for chaining
        """
        self._metrics[name] = metric
        return self
        
    def create_counter(self, name, description=None, unit=None, tags=None):
        """
        Create and add counter metric.
        
        Args:
            name: Metric name
            description: Optional description
            unit: Optional unit
            tags: Optional tags
            
        Returns:
            Created counter metric
        """
        merged_tags = self.tags.copy()
        if tags:
            merged_tags.update(tags)
            
        metric = self._registry.create_counter(
            name=f"{self.name}.{name}",
            description=description,
            dimension=self.dimension,
            unit=unit,
            tags=merged_tags
        )
        
        self._metrics[name] = metric
        return metric
        
    def create_gauge(self, name, description=None, unit=None, tags=None):
        """
        Create and add gauge metric.
        
        Args:
            name: Metric name
            description: Optional description
            unit: Optional unit
            tags: Optional tags
            
        Returns:
            Created gauge metric
        """
        merged_tags = self.tags.copy()
        if tags:
            merged_tags.update(tags)
            
        metric = self._registry.create_gauge(
            name=f"{self.name}.{name}",
            description=description,
            dimension=self.dimension,
            unit=unit,
            tags=merged_tags
        )
        
        self._metrics[name] = metric
        return metric
        
    def create_histogram(self, name, description=None, unit=None, tags=None, buckets=None):
        """
        Create and add histogram metric.
        
        Args:
            name: Metric name
            description: Optional description
            unit: Optional unit
            tags: Optional tags
            buckets: Optional histogram buckets
            
        Returns:
            Created histogram metric
        """
        merged_tags = self.tags.copy()
        if tags:
            merged_tags.update(tags)
            
        metric = self._registry.create_histogram(
            name=f"{self.name}.{name}",
            description=description,
            dimension=self.dimension,
            unit=unit,
            tags=merged_tags,
            buckets=buckets
        )
        
        self._metrics[name] = metric
        return metric
        
    def create_timer(self, name, description=None, tags=None):
        """
        Create and add timer metric.
        
        Args:
            name: Metric name
            description: Optional description
            tags: Optional tags
            
        Returns:
            Created timer metric
        """
        merged_tags = self.tags.copy()
        if tags:
            merged_tags.update(tags)
            
        metric = self._registry.create_timer(
            name=f"{self.name}.{name}",
            description=description,
            dimension=self.dimension,
            tags=merged_tags
        )
        
        self._metrics[name] = metric
        return metric
        
    def get_metric(self, name):
        """
        Get metric by name.
        
        Args:
            name: Metric name within composite
            
        Returns:
            Metric or None
        """
        return self._metrics.get(name)
        
    def get_all_metrics(self):
        """
        Get all metrics in composite.
        
        Returns:
            Dictionary of metrics
        """
        return self._metrics.copy()
        
    def as_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            'name': self.name,
            'description': self.description,
            'dimension': self.dimension.name,
            'period': self.period.name,
            'unit': self.unit,
            'tags': self.tags,
            'metrics': {}
        }
        
        for name, metric in self._metrics.items():
            result['metrics'][name] = metric.as_dict()
            
        return result
        
    def reset(self):
        """Reset all metrics in composite."""
        for metric in self._metrics.values():
            metric.reset()
```

## Implementation Strategy

The metrics framework implementation involves several key components:

### 1. Core Infrastructure

1. **Metrics Registry**: Central repository for all metrics:
   - Maintains references to all metrics in the system
   - Ensures unique metric names
   - Provides lookup by various criteria (name, type, tags)

2. **Base Metric Types**: Foundation classes for different metric types:
   - Counter: For monotonically increasing values
   - Gauge: For values that can go up and down
   - Histogram: For distribution of values
   - Timer: For duration measurements

3. **MetricValue**: Container for metric values with metadata:
   - Stores value, timestamp, and context metadata
   - Enables rich analysis of metrics

### 2. Domain-Specific Metrics

1. **Strategy Metrics**: Specialized metrics for trading strategies:
   - Performance metrics (returns, win rate)
   - Risk metrics (volatility, drawdown)
   - Trading metrics (trade count, profit factor)

2. **System Metrics**: Metrics for system performance:
   - CPU and memory usage
   - I/O operations
   - Network performance

3. **Operational Metrics**: Metrics for system health:
   - Availability and uptime
   - Error rates
   - Data quality and latency

### 3. Extensibility

1. **Custom Metrics Builder**: API for defining custom metrics:
   - Fluent interface for creating metrics
   - Namespace and tag management
   - Type-safe metric creation

2. **Composite Metrics**: Grouping related metrics together:
   - Hierarchical organization
   - Collective operations (reset, summary)
   - Standardized reporting

## Best Practices

### 1. Naming and Organization

- **Consistent Naming Convention**:
  ```python
  # Good naming - hierarchical with clear domain
  metrics_registry.create_gauge(
      name="strategy.macd_crossover.sharpe_ratio",
      description="Sharpe ratio for MACD crossover strategy"
  )
  
  # Avoid flat, generic names
  # Bad: metrics_registry.create_gauge(name="sharpe_ratio")
  ```

- **Use Tags for Dimensions**:
  ```python
  # Add relevant tags for filtering and aggregation
  metrics_registry.create_gauge(
      name="strategy.performance.return",
      tags={
          "strategy_id": "macd_crossover",
          "asset_class": "equity",
          "timeframe": "daily"
      }
  )
  ```

- **Group Related Metrics**:
  ```python
  # Create composite metric for related measures
  risk_metrics = CompositeMetric(
      name="strategy.risk_metrics",
      tags={"strategy_id": strategy_id}
  )
  
  # Add related metrics to composite
  risk_metrics.create_gauge("volatility", "Annualized volatility", unit="percent")
  risk_metrics.create_gauge("max_drawdown", "Maximum drawdown", unit="percent")
  risk_metrics.create_gauge("var_95", "95% Value at Risk", unit="percent")
  ```

### 2. Collection and Sampling

- **Record at Appropriate Frequency**:
  ```python
  # High-frequency metrics - record selectively
  if tick_count % sampling_rate == 0:
      system_metrics.update_cpu_metrics(get_cpu_usage())
  
  # Low-frequency metrics - record every occurrence
  strategy_metrics.record_trade(trade)
  ```

- **Use Timers for Duration Measurements**:
  ```python
  # Use timer context manager for accurate measurements
  with strategy_metrics.start_execution_timer() as timer:
      # Add context to timer
      timer.add_metadata('data_points', len(price_data))
      
      # Execute strategy
      result = strategy.execute(price_data)
  ```

- **Record Detailed Context**:
  ```python
  # Add rich metadata for analysis
  operational_metrics.record_error(
      error_type="DataError",
      source="MarketDataProvider",
      metadata={
          "symbol": "AAPL",
          "timeframe": "1min",
          "message": "Missing data points"
      }
  )
  ```

### 3. Analysis and Reporting

- **Calculate Derived Metrics**:
  ```python
  # Calculate and record derived metrics
  def update_risk_metrics(returns):
      volatility = calculate_volatility(returns)
      sharpe = calculate_sharpe_ratio(returns, volatility)
      sortino = calculate_sortino_ratio(returns)
      
      # Record derived metrics
      strategy_metrics.volatility.set(volatility)
      strategy_metrics.sharpe_ratio.set(sharpe)
      strategy_metrics.sortino_ratio.set(sortino)
  ```

- **Generate Comprehensive Summaries**:
  ```python
  # Get metrics summary for reporting
  def generate_strategy_report(strategy_id):
      metrics = get_strategy_metrics(strategy_id)
      summary = metrics.get_summary()
      
      return {
          "strategy": strategy_id,
          "performance": {
              "total_return": f"{summary['performance']['total_return']:.2f}%",
              "annualized_return": f"{summary['performance']['annualized_return']:.2f}%",
              "win_rate": f"{summary['performance']['win_rate']:.2f}%"
          },
          "risk": {
              "volatility": f"{summary['risk']['volatility']:.2f}%",
              "sharpe_ratio": f"{summary['risk']['sharpe_ratio']:.2f}",
              "max_drawdown": f"{summary['risk']['max_drawdown']:.2f}%"
          }
      }
  ```

- **Periodically Reset Metrics**:
  ```python
  # Reset metrics at appropriate intervals
  def reset_daily_metrics():
      # Get all metrics with daily period
      registry = MetricsRegistry.get_instance()
      daily_metrics = [m for m in registry.get_all_metrics() 
                     if m.period == MetricPeriod.DAY]
      
      # Reset metrics
      for metric in daily_metrics:
          metric.reset()
  ```

## Conclusion

The Metrics Framework provides a comprehensive system for measuring and analyzing various aspects of the ADMF-Trader system. By standardizing metrics collection and analysis, it enables:

1. **Consistent Evaluation**: Trading strategies can be evaluated using standardized metrics

2. **Performance Monitoring**: System performance can be tracked to identify bottlenecks

3. **Operational Visibility**: System health and reliability can be monitored in real-time

4. **Custom Analytics**: Users can define custom metrics for their specific needs

This framework serves as the foundation for data-driven decision making throughout the trading system, ensuring that all components are properly measured and optimized.