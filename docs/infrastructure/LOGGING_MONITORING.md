# Logging and Monitoring Framework

## Overview

This document defines the comprehensive logging and monitoring framework for the ADMF-Trader system. It outlines standardized logging levels, correlation tracking, health checks, and performance metrics that enable effective troubleshooting, performance analysis, and system monitoring.

## Motivation

A robust logging and monitoring framework is critical for:

1. **Troubleshooting**: Identifying and resolving issues quickly
2. **Performance Analysis**: Understanding system behavior and bottlenecks
3. **Auditability**: Tracking system actions for compliance and security
4. **Health Monitoring**: Ensuring system components are operating correctly
5. **Alerting**: Proactively identifying problems before they impact trading

## Logging Architecture

### 1. Logger Hierarchy

The system uses a hierarchical logger structure:

```
RootLogger
├── SystemLogger
│   ├── ConfigurationLogger
│   ├── ContainerLogger
│   └── ExecutionContextLogger
├── ComponentLogger
│   ├── DataHandlerLogger
│   ├── StrategyLogger
│   ├── RiskManagerLogger
│   ├── PortfolioLogger
│   └── BrokerLogger
├── EventLogger
│   ├── MarketDataLogger
│   ├── SignalLogger
│   ├── OrderLogger
│   └── FillLogger
└── PerformanceLogger
    ├── ThroughputLogger
    ├── LatencyLogger
    └── ResourceLogger
```

### 2. Logging Levels

Standard logging levels with clear usage guidelines:

| Level | Numeric Value | Usage Guidelines |
|-------|---------------|------------------|
| TRACE | 5 | Ultra-detailed debugging information, including method entry/exit |
| DEBUG | 10 | Information useful for debugging, not shown in production |
| INFO | 20 | General information about system operation |
| WARNING | 30 | Potential issues that don't affect operation |
| ERROR | 40 | Issues that affect operation but don't require immediate action |
| CRITICAL | 50 | Severe issues requiring immediate attention |

### 3. Structured Logging

The system uses structured logging with standardized fields:

```python
import logging
import json
import uuid
import time
import threading
import socket
from datetime import datetime

class StructuredLogger:
    """
    Structured logger for ADMF-Trader.
    
    Outputs logs in a consistent, structured format.
    """
    
    def __init__(self, name, context=None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            context: Execution context
        """
        self.name = name
        self.context = context
        self.hostname = socket.gethostname()
        
        # Create Python logger
        self.logger = logging.getLogger(name)
        
        # Add trace level
        logging.TRACE = 5
        logging.addLevelName(logging.TRACE, "TRACE")
        
        # Add trace method
        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.TRACE):
                self._log(logging.TRACE, message, args, **kwargs)
                
        logging.Logger.trace = trace
        
        # Initialize correlation ID
        self.correlation_id = None
        
    def set_correlation_id(self, correlation_id=None):
        """
        Set correlation ID for log entries.
        
        Args:
            correlation_id: Correlation ID (generated if None)
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        
    def get_correlation_id(self):
        """
        Get current correlation ID.
        
        Returns:
            Current correlation ID
        """
        return self.correlation_id
        
    def _log(self, level, msg, *args, **kwargs):
        """
        Log message with structured data.
        
        Args:
            level: Log level
            msg: Log message
            *args: Message arguments
            **kwargs: Additional log fields
        """
        # Skip if level not enabled
        if not self.logger.isEnabledFor(level):
            return
            
        # Format message
        if args:
            message = msg % args
        else:
            message = msg
            
        # Create structured log entry
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": logging.getLevelName(level),
            "logger": self.name,
            "message": message,
            "hostname": self.hostname,
            "thread": threading.current_thread().name,
            "correlation_id": self.correlation_id or "unknown"
        }
        
        # Add context information if available
        if self.context:
            entry["execution_mode"] = str(self.context.execution_mode)
            entry["thread_model"] = str(self.context.thread_model)
            
        # Add extra fields
        if "extra" in kwargs:
            for key, value in kwargs["extra"].items():
                entry[key] = value
                
        # Convert to JSON and log
        self.logger.log(level, json.dumps(entry))
        
    def trace(self, msg, *args, **kwargs):
        """Log TRACE message."""
        self._log(logging.TRACE, msg, *args, **kwargs)
        
    def debug(self, msg, *args, **kwargs):
        """Log DEBUG message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        """Log INFO message."""
        self._log(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        """Log WARNING message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """Log ERROR message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        """Log CRITICAL message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
```

### 4. Correlation Tracking

The system uses correlation IDs to track related log entries:

```python
class CorrelationContext:
    """
    Context manager for correlation tracking.
    
    Automatically sets correlation ID for logs within context.
    """
    
    def __init__(self, logger, correlation_id=None):
        """
        Initialize correlation context.
        
        Args:
            logger: Structured logger
            correlation_id: Correlation ID (generated if None)
        """
        self.logger = logger
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.previous_id = None
        
    def __enter__(self):
        """Enter correlation context."""
        # Save previous correlation ID
        self.previous_id = self.logger.get_correlation_id()
        
        # Set new correlation ID
        self.logger.set_correlation_id(self.correlation_id)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit correlation context."""
        # Restore previous correlation ID
        self.logger.set_correlation_id(self.previous_id)
```

### 5. Method Tracing

The system supports automatic method tracing:

```python
def trace_method(logger):
    """
    Decorator for method tracing.
    
    Args:
        logger: Logger to use for tracing
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Start time
            start_time = time.time()
            
            # Log method entry
            arg_str = ", ".join([f"{a}" for a in args[1:]] + [f"{k}={v}" for k, v in kwargs.items()])
            logger.trace(f"ENTER {func.__name__}({arg_str})")
            
            try:
                # Call method
                result = func(*args, **kwargs)
                
                # Log method exit
                elapsed_ms = (time.time() - start_time) * 1000
                logger.trace(f"EXIT {func.__name__} in {elapsed_ms:.2f}ms")
                
                return result
            except Exception as e:
                # Log exception
                elapsed_ms = (time.time() - start_time) * 1000
                logger.trace(f"EXCEPTION in {func.__name__}: {str(e)} after {elapsed_ms:.2f}ms")
                
                # Re-raise
                raise
                
        return wrapper
    return decorator
```

### 6. Log Output Configuration

Multiple log outputs with configurable handlers:

```python
def configure_logging(config):
    """
    Configure logging system.
    
    Args:
        config: Configuration provider
    """
    # Get logging configuration
    log_config = config.get_section("logging")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Configure console output
    if log_config.get("console_output", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_config.get("console_level", log_config.get("level", "INFO"))))
        root_logger.addHandler(console_handler)
        
    # Configure file output
    if log_config.get("file_output", False):
        file_path = log_config.get("file_path", "./logs/admf-trader.log")
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, log_config.get("file_level", log_config.get("level", "INFO"))))
        root_logger.addHandler(file_handler)
        
    # Configure remote output
    if log_config.get("remote_output", False):
        remote_config = log_config.get("remote", {})
        
        if remote_config.get("type") == "syslog":
            # Syslog handler
            from logging.handlers import SysLogHandler
            
            syslog_handler = SysLogHandler(
                address=(remote_config.get("host", "localhost"), remote_config.get("port", 514))
            )
            syslog_handler.setLevel(getattr(logging, remote_config.get("level", log_config.get("level", "INFO"))))
            root_logger.addHandler(syslog_handler)
        elif remote_config.get("type") == "elasticsearch":
            # Elasticsearch handler (would require additional library)
            pass
```

## Monitoring Architecture

### 1. Health Checks

The system implements component health checks:

```python
class HealthCheck:
    """
    Health check for system components.
    
    Provides status information about component health.
    """
    
    def __init__(self, name):
        """
        Initialize health check.
        
        Args:
            name: Health check name
        """
        self.name = name
        self.status = "unknown"
        self.details = {}
        self.last_check = None
        self.check_count = 0
        self.error_count = 0
        
    def check(self):
        """
        Perform health check.
        
        Returns:
            bool: Check passed
        """
        # Must be implemented by subclasses
        raise NotImplementedError
        
    def success(self, details=None):
        """
        Report successful health check.
        
        Args:
            details: Optional details
        """
        self.status = "healthy"
        self.details = details or {}
        self.last_check = datetime.now()
        self.check_count += 1
        
    def warning(self, message, details=None):
        """
        Report warning health check.
        
        Args:
            message: Warning message
            details: Optional details
        """
        self.status = "warning"
        self.details = details or {}
        self.details["message"] = message
        self.last_check = datetime.now()
        self.check_count += 1
        
    def error(self, message, details=None):
        """
        Report error health check.
        
        Args:
            message: Error message
            details: Optional details
        """
        self.status = "unhealthy"
        self.details = details or {}
        self.details["message"] = message
        self.last_check = datetime.now()
        self.check_count += 1
        self.error_count += 1
        
    def to_dict(self):
        """
        Convert health check to dictionary.
        
        Returns:
            Dict representation
        """
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "check_count": self.check_count,
            "error_count": self.error_count
        }
```

### 2. Component Health Checks

Health checks for system components:

```python
class ComponentHealthCheck(HealthCheck):
    """Health check for system component."""
    
    def __init__(self, component):
        """
        Initialize component health check.
        
        Args:
            component: Component to check
        """
        super().__init__(f"component.{component.name}")
        self.component = component
        
    def check(self):
        """
        Check component health.
        
        Returns:
            bool: Check passed
        """
        try:
            # Check if component initialized
            if not hasattr(self.component, "initialized") or not self.component.initialized:
                self.error("Component not initialized")
                return False
                
            # Check if component running (if applicable)
            if hasattr(self.component, "running") and not self.component.running:
                self.warning("Component not running")
                return False
                
            # Component-specific health check if available
            if hasattr(self.component, "health_check"):
                return self.component.health_check(self)
                
            # Default to success
            self.success()
            return True
        except Exception as e:
            self.error(f"Health check failed: {str(e)}")
            return False
```

### 3. System Health Check Manager

Centralized health check management:

```python
class HealthCheckManager:
    """
    System health check manager.
    
    Manages and runs health checks for all components.
    """
    
    def __init__(self):
        """Initialize health check manager."""
        self.health_checks = {}
        self.logger = StructuredLogger("health_check_manager")
        
    def register_check(self, health_check):
        """
        Register health check.
        
        Args:
            health_check: Health check to register
        """
        self.health_checks[health_check.name] = health_check
        
    def register_component(self, component):
        """
        Register component for health checking.
        
        Args:
            component: Component to check
        """
        health_check = ComponentHealthCheck(component)
        self.register_check(health_check)
        
    def run_all_checks(self):
        """
        Run all health checks.
        
        Returns:
            dict: Health check results
        """
        results = {}
        
        for name, check in self.health_checks.items():
            try:
                # Run check
                check.check()
                
                # Add to results
                results[name] = check.to_dict()
            except Exception as e:
                self.logger.error(f"Error running health check {name}: {str(e)}")
                
                # Report check error
                check.error(f"Check execution failed: {str(e)}")
                results[name] = check.to_dict()
                
        return results
        
    def get_system_health(self):
        """
        Get overall system health.
        
        Returns:
            dict: System health status
        """
        # Run all checks
        check_results = self.run_all_checks()
        
        # Determine overall health
        statuses = [result["status"] for result in check_results.values()]
        
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "warning" in statuses:
            overall_status = "warning"
        elif "healthy" in statuses:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
            
        return {
            "status": overall_status,
            "checks": check_results,
            "timestamp": datetime.now().isoformat()
        }
```

### 4. Performance Metrics

The system tracks key performance metrics:

```python
import statistics
from collections import deque

class MetricsCollector:
    """
    Performance metrics collector.
    
    Collects and reports system performance metrics.
    """
    
    def __init__(self, max_samples=1000):
        """
        Initialize metrics collector.
        
        Args:
            max_samples: Maximum samples to store
        """
        self.max_samples = max_samples
        self.metrics = {}
        self.logger = StructuredLogger("metrics_collector")
        
    def record_value(self, name, value, tags=None):
        """
        Record metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        # Create metric if not exists
        if name not in self.metrics:
            self.metrics[name] = {
                "values": deque(maxlen=self.max_samples),
                "count": 0,
                "sum": 0,
                "min": float('inf'),
                "max": float('-inf'),
                "tags": {}
            }
            
        # Add tags if provided
        if tags:
            for tag_name, tag_value in tags.items():
                if tag_name not in self.metrics[name]["tags"]:
                    self.metrics[name]["tags"][tag_name] = set()
                    
                self.metrics[name]["tags"][tag_name].add(tag_value)
                
        # Update statistics
        metric = self.metrics[name]
        metric["values"].append(value)
        metric["count"] += 1
        metric["sum"] += value
        metric["min"] = min(metric["min"], value)
        metric["max"] = max(metric["max"], value)
        
    def record_timing(self, name, duration_ms, tags=None):
        """
        Record timing metric.
        
        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
            tags: Optional tags
        """
        # Add timing-specific tag
        all_tags = tags or {}
        all_tags["metric_type"] = "timing"
        
        # Record value
        self.record_value(name, duration_ms, all_tags)
        
    def record_count(self, name, count=1, tags=None):
        """
        Record count metric.
        
        Args:
            name: Metric name
            count: Count to add
            tags: Optional tags
        """
        # Add count-specific tag
        all_tags = tags or {}
        all_tags["metric_type"] = "count"
        
        # Record value
        self.record_value(name, count, all_tags)
        
    def get_metric_statistics(self, name):
        """
        Get statistics for metric.
        
        Args:
            name: Metric name
            
        Returns:
            dict: Metric statistics
        """
        if name not in self.metrics:
            return None
            
        metric = self.metrics[name]
        values = list(metric["values"])
        
        # Calculate statistics
        stats = {
            "count": metric["count"],
            "min": metric["min"],
            "max": metric["max"],
            "mean": metric["sum"] / metric["count"] if metric["count"] > 0 else 0,
            "tags": metric["tags"]
        }
        
        # Add percentiles if enough values
        if values:
            stats["median"] = statistics.median(values)
            
            # Calculate percentiles
            values.sort()
            n = len(values)
            stats["p90"] = values[int(n * 0.9)]
            stats["p95"] = values[int(n * 0.95)]
            stats["p99"] = values[int(n * 0.99)]
            
        return stats
        
    def get_all_metrics(self):
        """
        Get all metrics.
        
        Returns:
            dict: All metrics
        """
        result = {}
        
        for name in self.metrics:
            result[name] = self.get_metric_statistics(name)
            
        return result
```

### 5. Performance Tracking

Performance tracking for methods and operations:

```python
class PerformanceTracker:
    """
    Performance tracking for methods and operations.
    
    Tracks execution time and reports metrics.
    """
    
    def __init__(self, metrics_collector):
        """
        Initialize performance tracker.
        
        Args:
            metrics_collector: Metrics collector
        """
        self.metrics_collector = metrics_collector
        
    def track_method(self, name=None, tags=None):
        """
        Decorator for method performance tracking.
        
        Args:
            name: Metric name (defaults to method name)
            tags: Optional tags
        """
        def decorator(func):
            metric_name = name or f"method.{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                # Start time
                start_time = time.time()
                
                try:
                    # Call method
                    result = func(*args, **kwargs)
                    
                    # Record timing
                    duration_ms = (time.time() - start_time) * 1000
                    self.metrics_collector.record_timing(metric_name, duration_ms, tags)
                    
                    return result
                except Exception as e:
                    # Record error
                    duration_ms = (time.time() - start_time) * 1000
                    error_tags = dict(tags or {})
                    error_tags["error"] = str(e)
                    self.metrics_collector.record_timing(f"{metric_name}.error", duration_ms, error_tags)
                    
                    # Re-raise
                    raise
                    
            return wrapper
        return decorator
        
    def track_operation(self, name, tags=None):
        """
        Context manager for operation performance tracking.
        
        Args:
            name: Operation name
            tags: Optional tags
            
        Returns:
            Context manager
        """
        return OperationTracker(name, self.metrics_collector, tags)
```

```python
class OperationTracker:
    """Context manager for operation tracking."""
    
    def __init__(self, name, metrics_collector, tags=None):
        """
        Initialize operation tracker.
        
        Args:
            name: Operation name
            metrics_collector: Metrics collector
            tags: Optional tags
        """
        self.name = name
        self.metrics_collector = metrics_collector
        self.tags = tags or {}
        self.start_time = None
        
    def __enter__(self):
        """Enter operation context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit operation context."""
        # Calculate duration
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type:
            # Record error
            error_tags = dict(self.tags)
            error_tags["error"] = str(exc_val)
            self.metrics_collector.record_timing(f"{self.name}.error", duration_ms, error_tags)
        else:
            # Record success
            self.metrics_collector.record_timing(self.name, duration_ms, self.tags)
```

### 6. Resource Monitoring

System resource monitoring:

```python
import psutil
import threading
import time

class ResourceMonitor:
    """
    System resource monitor.
    
    Monitors CPU, memory, and disk usage.
    """
    
    def __init__(self, metrics_collector, interval=10):
        """
        Initialize resource monitor.
        
        Args:
            metrics_collector: Metrics collector
            interval: Collection interval in seconds
        """
        self.metrics_collector = metrics_collector
        self.interval = interval
        self.logger = StructuredLogger("resource_monitor")
        self._running = False
        self._thread = None
        
    def start(self):
        """Start resource monitoring."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
            
    def _monitor_loop(self):
        """Resource monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Wait for next collection
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error collecting resource metrics: {str(e)}")
                time.sleep(self.interval)
                
    def _collect_metrics(self):
        """Collect resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_value("system.cpu.usage", cpu_percent)
        
        # Per-CPU usage
        per_cpu = psutil.cpu_percent(interval=1, percpu=True)
        for i, cpu in enumerate(per_cpu):
            self.metrics_collector.record_value("system.cpu.core.usage", cpu, {"core": i})
            
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.record_value("system.memory.usage", memory.percent)
        self.metrics_collector.record_value("system.memory.available", memory.available)
        
        # Disk usage
        disk = psutil.disk_usage("/")
        self.metrics_collector.record_value("system.disk.usage", disk.percent)
        self.metrics_collector.record_value("system.disk.free", disk.free)
        
        # Process metrics
        process = psutil.Process()
        
        # Process CPU usage
        process_cpu = process.cpu_percent(interval=1)
        self.metrics_collector.record_value("process.cpu.usage", process_cpu)
        
        # Process memory usage
        process_memory = process.memory_info()
        self.metrics_collector.record_value("process.memory.rss", process_memory.rss)
        self.metrics_collector.record_value("process.memory.vms", process_memory.vms)
        
        # Thread count
        thread_count = len(process.threads())
        self.metrics_collector.record_value("process.threads", thread_count)
```

## Integration with Monitoring Systems

### 1. Prometheus Integration

Metrics export for Prometheus:

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram

class PrometheusExporter:
    """
    Prometheus metrics exporter.
    
    Exports system metrics to Prometheus.
    """
    
    def __init__(self, metrics_collector, port=8000):
        """
        Initialize Prometheus exporter.
        
        Args:
            metrics_collector: Metrics collector
            port: HTTP port for metrics
        """
        self.metrics_collector = metrics_collector
        self.port = port
        self.logger = StructuredLogger("prometheus_exporter")
        self.gauges = {}
        self.counters = {}
        self.histograms = {}
        self._running = False
        self._thread = None
        
    def start(self):
        """Start Prometheus exporter."""
        if self._running:
            return
            
        # Start HTTP server
        start_http_server(self.port)
        
        # Start export thread
        self._running = True
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()
        
        self.logger.info(f"Prometheus exporter started on port {self.port}")
        
    def stop(self):
        """Stop Prometheus exporter."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
            
    def _export_loop(self):
        """Metrics export loop."""
        while self._running:
            try:
                # Export metrics
                self._export_metrics()
                
                # Wait before next export
                time.sleep(10)
            except Exception as e:
                self.logger.error(f"Error exporting metrics: {str(e)}")
                time.sleep(10)
                
    def _export_metrics(self):
        """Export metrics to Prometheus."""
        # Get all metrics
        all_metrics = self.metrics_collector.get_all_metrics()
        
        for name, stats in all_metrics.items():
            # Skip empty metrics
            if not stats:
                continue
                
            metric_type = stats.get("tags", {}).get("metric_type", ["gauge"])[0]
            
            if metric_type == "timing":
                # Use histogram for timing metrics
                if name not in self.histograms:
                    self.histograms[name] = Histogram(
                        name.replace(".", "_"),
                        name,
                        ["tag_" + tag for tag in stats.get("tags", {})]
                    )
                    
                # Update histogram with all values
                values = list(self.metrics_collector.metrics[name]["values"])
                for value in values:
                    self.histograms[name].observe(value)
            elif metric_type == "count":
                # Use counter for count metrics
                if name not in self.counters:
                    self.counters[name] = Counter(
                        name.replace(".", "_"),
                        name,
                        ["tag_" + tag for tag in stats.get("tags", {})]
                    )
                    
                # Update counter with latest value
                self.counters[name].inc(stats["mean"])
            else:
                # Use gauge for other metrics
                if name not in self.gauges:
                    self.gauges[name] = Gauge(
                        name.replace(".", "_"),
                        name,
                        ["tag_" + tag for tag in stats.get("tags", {})]
                    )
                    
                # Update gauge with latest value
                self.gauges[name].set(stats["mean"])
```

### 2. ELK Stack Integration

Log export to Elasticsearch:

```python
class ElasticsearchExporter:
    """
    Elasticsearch log exporter.
    
    Exports logs to Elasticsearch.
    """
    
    def __init__(self, host="localhost", port=9200, index_prefix="admf-trader"):
        """
        Initialize Elasticsearch exporter.
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_prefix: Index name prefix
        """
        self.host = host
        self.port = port
        self.index_prefix = index_prefix
        self.logger = StructuredLogger("elasticsearch_exporter")
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 100
        self._running = False
        self._thread = None
        
        # Import Elasticsearch client
        try:
            from elasticsearch import Elasticsearch
            self.client = Elasticsearch([f"{host}:{port}"])
        except ImportError:
            self.logger.error("Elasticsearch client not available")
            self.client = None
        
    def start(self):
        """Start Elasticsearch exporter."""
        if self._running or not self.client:
            return
            
        # Start export thread
        self._running = True
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()
        
        self.logger.info(f"Elasticsearch exporter started, sending to {self.host}:{self.port}")
        
    def stop(self):
        """Stop Elasticsearch exporter."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
            
    def add_log(self, log_entry):
        """
        Add log entry to buffer.
        
        Args:
            log_entry: Log entry dictionary
        """
        with self.buffer_lock:
            self.buffer.append(log_entry)
            
            # Export immediately if buffer full
            if len(self.buffer) >= self.max_buffer_size:
                self._export_logs()
                
    def _export_loop(self):
        """Log export loop."""
        while self._running:
            try:
                # Check if logs to export
                with self.buffer_lock:
                    if not self.buffer:
                        # No logs, wait
                        time.sleep(1)
                        continue
                        
                # Export logs
                self._export_logs()
                
                # Wait before next check
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error exporting logs: {str(e)}")
                time.sleep(5)
                
    def _export_logs(self):
        """Export logs to Elasticsearch."""
        with self.buffer_lock:
            if not self.buffer:
                return
                
            # Get logs to export
            logs = self.buffer
            self.buffer = []
            
        try:
            # Create index name
            index_name = f"{self.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"
            
            # Export logs
            from elasticsearch.helpers import bulk
            
            actions = [
                {
                    "_index": index_name,
                    "_source": log
                }
                for log in logs
            ]
            
            bulk(self.client, actions)
        except Exception as e:
            self.logger.error(f"Failed to export logs: {str(e)}")
            
            # Add logs back to buffer
            with self.buffer_lock:
                self.buffer.extend(logs)
```

### 3. Alerting Integration

Integration with alerting systems:

```python
class AlertManager:
    """
    Alert manager for system monitoring.
    
    Manages alerts and notifications.
    """
    
    def __init__(self, config):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration provider
        """
        self.config = config
        self.logger = StructuredLogger("alert_manager")
        self.alert_handlers = {}
        self._initialize_handlers()
        
    def _initialize_handlers(self):
        """Initialize alert handlers."""
        alert_config = self.config.get_section("monitoring.alerts")
        
        # Email alerts
        if alert_config.get("email.enabled", False):
            self.alert_handlers["email"] = EmailAlertHandler(alert_config.get("email", {}))
            
        # Slack alerts
        if alert_config.get("slack.enabled", False):
            self.alert_handlers["slack"] = SlackAlertHandler(alert_config.get("slack", {}))
            
        # Webhook alerts
        if alert_config.get("webhook.enabled", False):
            self.alert_handlers["webhook"] = WebhookAlertHandler(alert_config.get("webhook", {}))
            
    def alert(self, severity, message, details=None):
        """
        Send alert.
        
        Args:
            severity: Alert severity (info, warning, error, critical)
            message: Alert message
            details: Alert details
        """
        # Create alert
        alert = {
            "severity": severity,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Log alert
        log_method = getattr(self.logger, severity if severity in ["info", "warning", "error", "critical"] else "info")
        log_method(f"ALERT: {message}", extra={"alert": alert})
        
        # Send to handlers
        for name, handler in self.alert_handlers.items():
            try:
                handler.send_alert(alert)
            except Exception as e:
                self.logger.error(f"Error sending alert to {name}: {str(e)}")
```

```python
class EmailAlertHandler:
    """
    Email alert handler.
    
    Sends alerts via email.
    """
    
    def __init__(self, config):
        """
        Initialize email alert handler.
        
        Args:
            config: Email configuration
        """
        self.config = config
        self.logger = StructuredLogger("email_alert_handler")
        
    def send_alert(self, alert):
        """
        Send alert via email.
        
        Args:
            alert: Alert information
        """
        import smtplib
        from email.mime.text import MIMEText
        
        # Create message
        subject = f"[{alert['severity'].upper()}] ADMF-Trader Alert: {alert['message']}"
        body = f"Alert: {alert['message']}\n\nDetails: {alert['details']}\n\nTime: {alert['timestamp']}"
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.config.get("from")
        msg['To'] = ", ".join(self.config.get("recipients", []))
        
        # Send email
        try:
            server = smtplib.SMTP(self.config.get("smtp_server"), self.config.get("smtp_port", 587))
            
            if self.config.get("use_tls", True):
                server.starttls()
                
            if self.config.get("username") and self.config.get("password"):
                server.login(self.config.get("username"), self.config.get("password"))
                
            server.send_message(msg)
            server.quit()
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            raise
```

## Integration with System

### 1. Component Integration

Components integrate with the logging and monitoring framework:

```python
class MonitoredComponent(Component):
    """Component with built-in monitoring."""
    
    def __init__(self, name, config=None):
        """Initialize monitored component."""
        super().__init__(name, config)
        self.logger = StructuredLogger(f"component.{name}")
        self.performance_tracker = None
        
    def initialize(self, context):
        """Initialize component."""
        super().initialize(context)
        
        # Get metrics collector
        if hasattr(context, "metrics_collector"):
            self.metrics_collector = context.metrics_collector
            self.performance_tracker = PerformanceTracker(self.metrics_collector)
            
        # Log initialization
        self.logger.info(f"Component {self.name} initialized")
        
    def start(self):
        """Start component."""
        super().start()
        
        # Log start
        self.logger.info(f"Component {self.name} started")
        
    def stop(self):
        """Stop component."""
        super().stop()
        
        # Log stop
        self.logger.info(f"Component {self.name} stopped")
        
    def health_check(self, check):
        """
        Component health check.
        
        Args:
            check: Health check instance
            
        Returns:
            bool: Check passed
        """
        # Basic health check
        check.success({
            "initialized": self.initialized,
            "running": self.running
        })
        
        return True
```

### 2. System Bootstrap Integration

Integration with system bootstrap:

```python
def bootstrap_system_with_monitoring(config_path=None, env=None):
    """
    Bootstrap system with monitoring.
    
    Args:
        config_path: Path to configuration directory
        env: Environment name
        
    Returns:
        System container
    """
    # Bootstrap base system
    container, context = bootstrap_system(config_path, env)
    
    # Configure logging
    configure_logging(container.get("config"))
    
    # Create system logger
    system_logger = StructuredLogger("system")
    system_logger.info("System bootstrap started")
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    container.register_instance("metrics_collector", metrics_collector)
    context.metrics_collector = metrics_collector
    
    # Create health check manager
    health_check_manager = HealthCheckManager()
    container.register_instance("health_check_manager", health_check_manager)
    
    # Register components for health checks
    for name, component in container.get_components().items():
        health_check_manager.register_component(component)
        
    # Start resource monitor
    resource_monitor = ResourceMonitor(metrics_collector)
    container.register_instance("resource_monitor", resource_monitor)
    resource_monitor.start()
    
    # Create alert manager
    alert_manager = AlertManager(container.get("config"))
    container.register_instance("alert_manager", alert_manager)
    
    # Add metrics exporters if enabled
    monitor_config = container.get("config").get_section("monitoring")
    
    if monitor_config.get("prometheus.enabled", False):
        prometheus_exporter = PrometheusExporter(
            metrics_collector,
            port=monitor_config.get("prometheus.port", 8000)
        )
        container.register_instance("prometheus_exporter", prometheus_exporter)
        prometheus_exporter.start()
        
    if monitor_config.get("elasticsearch.enabled", False):
        elasticsearch_exporter = ElasticsearchExporter(
            host=monitor_config.get("elasticsearch.host", "localhost"),
            port=monitor_config.get("elasticsearch.port", 9200),
            index_prefix=monitor_config.get("elasticsearch.index_prefix", "admf-trader")
        )
        container.register_instance("elasticsearch_exporter", elasticsearch_exporter)
        elasticsearch_exporter.start()
        
    system_logger.info("System bootstrap completed")
    
    return container, context
```

## Configuration Examples

### 1. Development Configuration

```yaml
# development.yaml
logging:
  level: DEBUG
  console_output: true
  console_level: DEBUG
  file_output: true
  file_path: ./logs/admf-trader-dev.log
  file_level: DEBUG
  
monitoring:
  prometheus:
    enabled: true
    port: 8000
  alerts:
    email:
      enabled: false
```

### 2. Testing Configuration

```yaml
# testing.yaml
logging:
  level: INFO
  console_output: true
  console_level: INFO
  file_output: true
  file_path: ./logs/admf-trader-test.log
  file_level: DEBUG
  
monitoring:
  prometheus:
    enabled: true
    port: 8000
  elasticsearch:
    enabled: false
  alerts:
    email:
      enabled: false
```

### 3. Production Configuration

```yaml
# production.yaml
logging:
  level: INFO
  console_output: false
  file_output: true
  file_path: /var/log/admf-trader/trading.log
  file_level: INFO
  remote_output: true
  remote:
    type: elasticsearch
    host: elk-server.example.com
    port: 9200
    level: INFO
  
monitoring:
  prometheus:
    enabled: true
    port: 8000
  elasticsearch:
    enabled: true
    host: elk-server.example.com
    port: 9200
    index_prefix: admf-trader
  alerts:
    email:
      enabled: true
      from: alerts@admf-trader.example.com
      recipients: [admin@example.com, ops@example.com]
      smtp_server: smtp.example.com
      smtp_port: 587
      use_tls: true
      username: alerts@admf-trader.example.com
      password: ${SMTP_PASSWORD}
    slack:
      enabled: true
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#alerts"
```

## Implementation Strategy

### 1. Core Logging Implementation

1. Implement `StructuredLogger` class
2. Create log configuration mechanism
3. Implement correlation tracking

### 2. Monitoring Infrastructure

1. Implement `HealthCheckManager`
2. Create `MetricsCollector`
3. Implement `ResourceMonitor`

### 3. Performance Tracking

1. Implement `PerformanceTracker`
2. Create method and operation tracking decorators
3. Add tracking to critical components

### 4. Integration

1. Update component base classes for logging and monitoring
2. Modify system bootstrap to initialize monitoring
3. Create exporters for external systems

## Best Practices

### 1. Logging Guidelines

- Use appropriate log levels consistently
- Include relevant context in all log messages
- Use structured logging for machine parsing
- Use correlation IDs for tracking multi-component operations
- Include timing information for performance-sensitive operations

### 2. Monitoring Guidelines

- Define clear health check criteria for each component
- Track all critical operations with performance metrics
- Set appropriate thresholds for alerts
- Monitor both system and application-level resources
- Use tags to categorize and filter metrics

### 3. Alerting Guidelines

- Define clear alert severity levels
- Include actionable information in alerts
- Set appropriate thresholds to avoid alert fatigue
- Implement alert deduplication
- Ensure alerts reach the right recipients

## Conclusion

The logging and monitoring framework provides comprehensive visibility into the ADMF-Trader system's operation, performance, and health. By implementing structured logging, correlation tracking, performance metrics, and health checks, the system enables effective troubleshooting, performance analysis, and proactive monitoring.

Key benefits include:

1. **Troubleshooting**: Easily trace issues across components
2. **Performance Analysis**: Identify bottlenecks and optimize critical paths
3. **Health Monitoring**: Detect issues before they impact trading
4. **Alerting**: Respond quickly to critical conditions
5. **Integration**: Connect with industry-standard monitoring tools

With this framework, operators and developers can ensure the system remains reliable, performant, and transparent in its operation.