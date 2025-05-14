# Error Handling Strategy

This document outlines the design for the ADMF-Trader Error Handling Strategy, which provides a comprehensive approach to managing errors, exception handling, error propagation, and recovery mechanisms throughout the system.

## 1. Overview

The Error Handling Strategy defines a structured approach to identify, capture, report, and recover from errors in the ADMF-Trader system. It focuses on four key elements:

1. **Exception Hierarchy**: A comprehensive hierarchy of exception types
2. **Error Propagation**: Consistent patterns for error propagation and handling
3. **Recovery Mechanisms**: Retry and recovery strategies for recoverable errors
4. **Error Testing**: Error injection capabilities for testing error handling

## 2. Exception Hierarchy

### 2.1 Exception Base Class

All exceptions in the system inherit from a common base class:

```python
class ADMFException(Exception):
    """Base exception for all ADMF-Trader exceptions."""
    
    def __init__(self, message, code=None, details=None, recoverable=False, context=None):
        """
        Initialize exception with detailed information.
        
        Args:
            message (str): Human-readable error message
            code (str, optional): Error code for programmatic handling
            details (dict, optional): Additional error details
            recoverable (bool): Whether the error is potentially recoverable
            context (dict, optional): Execution context when error occurred
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.recoverable = recoverable
        self.context = context or {}
        self.timestamp = datetime.datetime.now()
        
        # Construct standard error message
        full_message = f"[{self.code}] {self.message}" if self.code else self.message
        super().__init__(full_message)
    
    def to_dict(self):
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    @classmethod
    def from_exception(cls, exception, message=None, code=None, details=None, recoverable=False):
        """Create an ADMF exception from another exception."""
        new_message = message or str(exception)
        new_details = details or {}
        new_details["original_exception"] = {
            "type": exception.__class__.__name__,
            "message": str(exception)
        }
        
        return cls(new_message, code, new_details, recoverable)
```

### 2.2 Exception Class Hierarchy

The system defines a structured hierarchy of exception types:

```
ADMFException (Base)
├── ConfigurationError
│   ├── MissingConfigurationError
│   ├── InvalidConfigurationError
│   └── ConfigurationValidationError
├── DataError
│   ├── DataSourceError
│   ├── DataIntegrityError
│   ├── DataFormatError
│   └── DataSplitError
├── EventError
│   ├── EventPublishError
│   ├── EventSubscriptionError
│   ├── InvalidEventTypeError
│   └── EventContextViolationError
├── ComponentError
│   ├── ComponentInitializationError
│   ├── ComponentStateError
│   ├── ComponentLifecycleError
│   └── ComponentDependencyError
├── StrategyError
│   ├── SignalGenerationError
│   ├── StrategyParameterError
│   └── IndicatorCalculationError
├── RiskError
│   ├── RiskLimitViolationError
│   ├── PositionSizingError
│   └── PositionTrackingError
├── ExecutionError
│   ├── OrderExecutionError
│   ├── FillProcessingError
│   └── BrokerCommunicationError
├── SystemError
│   ├── ThreadSafetyViolationError
│   ├── MemoryError
│   ├── PerformanceError
│   └── ConcurrencyError
├── ValidationError
│   ├── StateValidationError
│   ├── DataValidationError
│   └── RuleValidationError
└── OptimizationError
    ├── ParameterSpaceError
    ├── ObjectiveFunctionError
    └── OptimizationRunError
```

Each exception class includes standard documentation:

```python
class DataValidationError(ValidationError):
    """
    Exception raised when data fails validation checks.
    
    Attributes:
        message (str): Explanation of the validation failure
        code (str): Error code for programmatic handling
        details (dict): Additional error details including validation failures
        recoverable (bool): Whether recovery is possible (default: True)
        
    Common causes:
        - Missing required fields in data
        - Data values outside acceptable ranges
        - Inconsistent data relationships
        
    Recovery strategies:
        - Retry with corrected data
        - Use default values for missing/invalid fields
        - Skip the invalid data item
    """
```

## 3. Error Propagation Patterns

### 3.1 Error Boundaries

Error boundaries provide a way to contain and handle errors within logical components:

```python
class ErrorBoundary:
    """
    A context manager to create an error boundary, which captures and handles
    exceptions in a controlled manner.
    """
    
    def __init__(self, component_name, handler=None, logger=None, reraise=True, 
                 transform=True, publish_event=True, event_bus=None):
        """
        Initialize error boundary.
        
        Args:
            component_name (str): Name of the component creating the boundary
            handler (callable, optional): Function to handle errors
            logger (Logger, optional): Logger to use for error logging
            reraise (bool): Whether to reraise exceptions after handling
            transform (bool): Whether to transform exceptions to ADMFException
            publish_event (bool): Whether to publish error events
            event_bus (EventBus, optional): Event bus for error events
        """
        self.component_name = component_name
        self.handler = handler
        self.logger = logger
        self.reraise = reraise
        self.transform = transform
        self.publish_event = publish_event
        self.event_bus = event_bus
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
            
        # Transform exception if needed
        exception = exc_val
        if self.transform and not isinstance(exception, ADMFException):
            exception = ADMFException.from_exception(
                exception, 
                message=f"Error in component {self.component_name}: {str(exception)}",
                details={"component": self.component_name}
            )
            
        # Log the error
        if self.logger:
            self.logger.error(
                f"Error in component {self.component_name}: {exception}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            
        # Publish error event
        if self.publish_event and self.event_bus:
            self.event_bus.publish(Event(
                EventType.ERROR,
                {
                    "component": self.component_name,
                    "error": exception.to_dict() if hasattr(exception, "to_dict") else str(exception)
                }
            ))
            
        # Call custom handler
        if self.handler:
            self.handler(exception)
            
        # Indicate whether to suppress the exception
        return not self.reraise
```

### 3.2 Component Error Handling

Components standardize error handling through base class functionality:

```python
class Component:
    """Base class for all components with error handling."""
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.error_handlers = {}
        
    def handle_error(self, error_type, handler):
        """Register an error handler for a specific error type."""
        self.error_handlers[error_type] = handler
        
    def execute_with_error_handling(self, func, *args, **kwargs):
        """Execute a function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Determine error type and find appropriate handler
            error_type = type(e)
            handler = None
            
            # Look for exact type match first, then base classes
            if error_type in self.error_handlers:
                handler = self.error_handlers[error_type]
            else:
                # Find handler for parent classes
                for registered_type, registered_handler in self.error_handlers.items():
                    if issubclass(error_type, registered_type):
                        handler = registered_handler
                        break
            
            # Execute handler if found
            if handler:
                return handler(e, *args, **kwargs)
            
            # No handler found, re-raise as ADMFException
            if not isinstance(e, ADMFException):
                raise ADMFException.from_exception(
                    e,
                    message=f"Error in component {self.name}: {str(e)}",
                    details={"component": self.name}
                )
            raise
```

### 3.3 Error Events

Errors are published as events to enable system-wide monitoring:

```python
class ErrorEventPublisher:
    """Utility for publishing error events to the event bus."""
    
    def __init__(self, event_bus, component_name):
        self.event_bus = event_bus
        self.component_name = component_name
        
    def publish_error(self, error, context=None):
        """Publish an error event to the event bus."""
        error_data = {
            "component": self.component_name,
            "error": error.to_dict() if hasattr(error, "to_dict") else str(error),
            "context": context or {}
        }
        
        self.event_bus.publish(Event(EventType.ERROR, error_data))
```

## 4. Retry Mechanisms

### 4.1 Retry Decorator

For simple retry functionality:

```python
def retry(max_attempts=3, backoff_factor=2, max_delay=60, 
          retry_exceptions=(Exception,), logger=None):
    """
    Decorator that retries a function in case of specified exceptions.
    
    Args:
        max_attempts (int): Maximum number of attempts
        backoff_factor (float): Factor to multiply delay after each attempt
        max_delay (float): Maximum delay between retries in seconds
        retry_exceptions (tuple): Exceptions to catch and retry
        logger (Logger, optional): Logger for retry attempts
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = 1
            last_exception = None
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    attempts += 1
                    last_exception = e
                    
                    # Check if we should retry
                    if attempts >= max_attempts:
                        break
                        
                    # Calculate delay with exponential backoff
                    delay = min(delay * backoff_factor, max_delay)
                    
                    # Log retry attempt
                    if logger:
                        logger.warning(
                            f"Retry attempt {attempts}/{max_attempts} for {func.__name__} "
                            f"after {delay}s: {str(e)}"
                        )
                        
                    # Sleep before retry
                    time.sleep(delay)
            
            # Retries exhausted, convert to ADMFException if needed
            if not isinstance(last_exception, ADMFException):
                raise ADMFException.from_exception(
                    last_exception,
                    message=f"Failed after {max_attempts} attempts: {str(last_exception)}",
                    details={"attempts": max_attempts}
                )
            raise last_exception
            
        return wrapper
    return decorator
```

### 4.2 Retry Context Manager

For more complex retry scenarios:

```python
class RetryContext:
    """
    Context manager for retry operations with customizable policies.
    """
    
    def __init__(self, max_attempts=3, backoff_factor=2, max_delay=60,
                 retry_exceptions=(Exception,), logger=None,
                 on_retry=None, on_give_up=None):
        """
        Initialize retry context.
        
        Args:
            max_attempts (int): Maximum number of attempts
            backoff_factor (float): Factor to multiply delay after each attempt
            max_delay (float): Maximum delay between retries in seconds
            retry_exceptions (tuple): Exceptions to catch and retry
            logger (Logger, optional): Logger for retry attempts
            on_retry (callable, optional): Called before each retry
            on_give_up (callable, optional): Called when retries are exhausted
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_exceptions = retry_exceptions
        self.logger = logger
        self.on_retry = on_retry
        self.on_give_up = on_give_up
        
        # State
        self.attempts = 0
        self.last_exception = None
        self.delay = 1
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
            
        # Check if this exception should be retried
        retry_matched = False
        for retry_exception in self.retry_exceptions:
            if issubclass(exc_type, retry_exception):
                retry_matched = True
                break
                
        if not retry_matched:
            return False
            
        self.attempts += 1
        self.last_exception = exc_val
        
        # Check if we should retry
        if self.attempts >= self.max_attempts:
            if self.on_give_up:
                self.on_give_up(exc_val, self.attempts)
            return False
            
        # Calculate delay with exponential backoff
        self.delay = min(self.delay * self.backoff_factor, self.max_delay)
        
        # Log retry attempt
        if self.logger:
            self.logger.warning(
                f"Retry attempt {self.attempts}/{self.max_attempts} "
                f"after {self.delay}s: {str(exc_val)}"
            )
            
        # Call on_retry callback
        if self.on_retry:
            self.on_retry(exc_val, self.attempts, self.delay)
            
        # Sleep before retry
        time.sleep(self.delay)
        
        # Indicate that we're handling the exception
        return True
```

### 4.3 RetryableOperation Class

For complex operations with state management:

```python
class RetryableOperation:
    """
    A class for operations that require complex retry logic with state management.
    """
    
    def __init__(self, name, retry_policy, logger=None, event_bus=None):
        """
        Initialize a retryable operation.
        
        Args:
            name (str): Operation name for tracking and logging
            retry_policy (dict): Policy parameters (max_attempts, backoff, etc.)
            logger (Logger, optional): Logger for retry attempts
            event_bus (EventBus, optional): Event bus for publishing retry events
        """
        self.name = name
        self.logger = logger
        self.event_bus = event_bus
        
        # Retry policy
        self.max_attempts = retry_policy.get("max_attempts", 3)
        self.backoff_factor = retry_policy.get("backoff_factor", 2)
        self.max_delay = retry_policy.get("max_delay", 60)
        self.retry_exceptions = retry_policy.get("retry_exceptions", (Exception,))
        
        # State
        self.attempts = 0
        self.last_attempt_time = None
        self.last_exception = None
        self.successes = 0
        self.failures = 0
        
    def execute(self, operation, *args, **kwargs):
        """
        Execute an operation with retry logic.
        
        Args:
            operation (callable): The operation to execute
            *args, **kwargs: Arguments to pass to the operation
        """
        self.attempts = 0
        delay = 1
        
        while self.attempts < self.max_attempts:
            try:
                self.last_attempt_time = datetime.datetime.now()
                result = operation(*args, **kwargs)
                self.successes += 1
                return result
            except self.retry_exceptions as e:
                self.attempts += 1
                self.last_exception = e
                
                # Publish event
                if self.event_bus:
                    self.event_bus.publish(Event(
                        EventType.RETRY,
                        {
                            "operation": self.name,
                            "attempt": self.attempts,
                            "max_attempts": self.max_attempts,
                            "error": str(e)
                        }
                    ))
                
                # Check if we should retry
                if self.attempts >= self.max_attempts:
                    self.failures += 1
                    
                    # Publish failure event
                    if self.event_bus:
                        self.event_bus.publish(Event(
                            EventType.RETRY_EXHAUSTED,
                            {
                                "operation": self.name,
                                "attempts": self.attempts,
                                "error": str(e)
                            }
                        ))
                    
                    # Raise appropriate exception
                    if not isinstance(e, ADMFException):
                        raise ADMFException.from_exception(
                            e,
                            message=f"Operation {self.name} failed after {self.attempts} attempts",
                            details={"operation": self.name, "attempts": self.attempts}
                        )
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(delay * self.backoff_factor, self.max_delay)
                
                # Log retry
                if self.logger:
                    self.logger.warning(
                        f"Retry attempt {self.attempts}/{self.max_attempts} "
                        f"for operation {self.name} after {delay}s: {str(e)}"
                    )
                
                # Sleep before retry
                time.sleep(delay)
```

## 5. Error Injection Testing Framework

### 5.1 Error Injector

The Error Injector enables controlled error injection for testing:

```python
class ErrorInjector:
    """
    Framework for injecting errors at specific points for testing error handling.
    """
    
    def __init__(self):
        self.injection_points = {}
        self.active = False
        
    def register_injection_point(self, point_id, probability=1.0, error_factory=None):
        """
        Register an error injection point.
        
        Args:
            point_id (str): Identifier for the injection point
            probability (float): Probability of error injection (0.0-1.0)
            error_factory (callable): Function that returns an exception to raise
        """
        self.injection_points[point_id] = {
            "probability": probability,
            "error_factory": error_factory or (lambda: Exception(f"Injected error at {point_id}")),
            "count": 0,
            "active": True
        }
        
    def enable(self):
        """Enable error injection globally."""
        self.active = True
        
    def disable(self):
        """Disable error injection globally."""
        self.active = False
        
    def activate_point(self, point_id):
        """Activate a specific injection point."""
        if point_id in self.injection_points:
            self.injection_points[point_id]["active"] = True
            
    def deactivate_point(self, point_id):
        """Deactivate a specific injection point."""
        if point_id in self.injection_points:
            self.injection_points[point_id]["active"] = False
            
    def set_probability(self, point_id, probability):
        """Set error injection probability for a point."""
        if point_id in self.injection_points:
            self.injection_points[point_id]["probability"] = max(0.0, min(1.0, probability))
            
    def inject_error(self, point_id):
        """
        Check and potentially inject an error at the specified point.
        
        Args:
            point_id (str): Identifier for the injection point
            
        Raises:
            Exception: If error should be injected
        """
        if not self.active:
            return
            
        if point_id not in self.injection_points:
            return
            
        injection_point = self.injection_points[point_id]
        if not injection_point["active"]:
            return
            
        # Check probability
        if random.random() < injection_point["probability"]:
            injection_point["count"] += 1
            error = injection_point["error_factory"]()
            raise error
```

### 5.2 Fault Injection Decorator

For easy error injection in methods:

```python
def inject_fault(point_id, injector):
    """
    Decorator to add fault injection to a method.
    
    Args:
        point_id (str): Identifier for the injection point
        injector (ErrorInjector): Error injector instance
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to inject an error first
            try:
                injector.inject_error(point_id)
            except Exception as e:
                # Log the injected error
                logging.debug(f"Injected error at {point_id}: {str(e)}")
                raise
                
            # If no error was injected, execute the function normally
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### 5.3 Error Injection Test Case

Base class for tests that use error injection:

```python
class ErrorInjectionTestCase(unittest.TestCase):
    """Base class for tests that use error injection."""
    
    def setUp(self):
        self.error_injector = ErrorInjector()
        self.error_injector.enable()
        
    def tearDown(self):
        self.error_injector.disable()
        
    def register_error_point(self, point_id, exception_class=Exception, probability=1.0):
        """Register an error injection point for the test."""
        self.error_injector.register_injection_point(
            point_id,
            probability=probability,
            error_factory=lambda: exception_class(f"Test-injected error at {point_id}")
        )
        
    def test_with_error_injection(self, point_id, test_function, expected_exception=None):
        """
        Run a test function with error injection.
        
        Args:
            point_id (str): Injection point ID
            test_function (callable): Function to test
            expected_exception (class): Expected exception class
        """
        self.register_error_point(point_id)
        
        if expected_exception:
            with self.assertRaises(expected_exception):
                test_function()
        else:
            test_function()
```

## 6. Error Monitoring and Logging

### 6.1 Structured Error Logging

```python
class ErrorLogger:
    """Specialized logger for error handling with structured logging."""
    
    def __init__(self, name, event_bus=None):
        self.logger = logging.getLogger(name)
        self.event_bus = event_bus
        
    def log_error(self, error, context=None, exc_info=True):
        """
        Log an error with structured information.
        
        Args:
            error: The error/exception that occurred
            context (dict, optional): Additional context information
            exc_info (bool): Whether to include exception info in log
        """
        context = context or {}
        
        # Extract error details
        error_details = {
            "type": error.__class__.__name__,
            "message": str(error)
        }
        
        # Add additional fields if it's an ADMFException
        if isinstance(error, ADMFException):
            error_details.update({
                "code": error.code,
                "details": error.details,
                "recoverable": error.recoverable,
                "context": error.context
            })
        
        # Combine with provided context
        log_context = {**context, "error": error_details}
        
        # Format structured log message
        log_message = f"Error: {error}"
        
        # Log the error
        self.logger.error(log_message, exc_info=exc_info, extra={"context": log_context})
        
        # Publish error event if event bus is available
        if self.event_bus:
            self.event_bus.publish(Event(
                EventType.ERROR,
                {
                    "error": error_details,
                    "context": context
                }
            ))
```

### 6.2 Error Monitoring

```python
class ErrorMonitor:
    """
    Central error monitoring and aggregation system.
    """
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.error_counts = defaultdict(int)
        self.error_records = defaultdict(list)
        self.max_records_per_type = 100
        
        # Subscribe to error events
        self.event_bus.subscribe(EventType.ERROR, self.on_error)
        
    def on_error(self, event):
        """Handle error events for monitoring."""
        error_data = event.get_data().get("error", {})
        error_type = error_data.get("type", "UnknownError")
        
        # Update error counts
        self.error_counts[error_type] += 1
        
        # Store error record
        error_record = {
            "timestamp": datetime.datetime.now(),
            "error": error_data,
            "context": event.get_data().get("context", {})
        }
        
        # Limit number of records per type
        self.error_records[error_type].append(error_record)
        if len(self.error_records[error_type]) > self.max_records_per_type:
            self.error_records[error_type] = self.error_records[error_type][-self.max_records_per_type:]
            
        # Check for error thresholds
        self._check_thresholds(error_type)
        
    def _check_thresholds(self, error_type):
        """Check if error counts exceed thresholds and take action."""
        count = self.error_counts[error_type]
        
        # Example threshold checks
        if count >= 100:
            self.event_bus.publish(Event(
                EventType.ALERT,
                {
                    "type": "error_threshold",
                    "error_type": error_type,
                    "count": count,
                    "severity": "critical"
                }
            ))
        elif count >= 50:
            self.event_bus.publish(Event(
                EventType.ALERT,
                {
                    "type": "error_threshold",
                    "error_type": error_type,
                    "count": count,
                    "severity": "high"
                }
            ))
```

## 7. Component Integration

### 7.1 Enhanced Component Base Class

```python
class Component:
    """Extended Component base class with error handling integration."""
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.error_logger = ErrorLogger(f"{self.__class__.__name__}.{name}")
        self.error_handlers = {}
        self.retry_policies = {}
        
    def initialize(self, context):
        """Initialize component with enhanced error handling."""
        try:
            # Register with error monitoring
            if context.has("error_monitor"):
                self.error_monitor = context.get("error_monitor")
                
            # Set up error event publisher
            if context.has("event_bus"):
                self.event_bus = context.get("event_bus")
                self.error_publisher = ErrorEventPublisher(self.event_bus, self.name)
                
            # Initialize component-specific error handling
            self._initialize_error_handling()
                
            # Continue with normal initialization
            self._do_initialize(context)
        except Exception as e:
            # Log and convert the error
            self.error_logger.log_error(
                e, context={"stage": "initialization", "component": self.name}
            )
            
            # Convert to appropriate exception type
            if not isinstance(e, ComponentInitializationError):
                raise ComponentInitializationError(
                    f"Failed to initialize component {self.name}",
                    details={"component": self.name, "original_error": str(e)},
                    recoverable=False
                ) from e
            raise
            
    def handle_error(self, error_type, handler):
        """Register an error handler for a specific error type."""
        self.error_handlers[error_type] = handler
        
    def define_retry_policy(self, operation_name, policy):
        """Define a retry policy for a specific operation."""
        self.retry_policies[operation_name] = policy
        
    def with_error_boundary(self, operation_name=None):
        """Create an error boundary for a specific operation."""
        return ErrorBoundary(
            component_name=f"{self.name}.{operation_name}" if operation_name else self.name,
            logger=self.logger,
            event_bus=getattr(self, "event_bus", None)
        )
        
    def with_retry(self, operation_name):
        """Create a retry context for a specific operation."""
        policy = self.retry_policies.get(operation_name, {})
        return RetryContext(
            max_attempts=policy.get("max_attempts", 3),
            backoff_factor=policy.get("backoff_factor", 2),
            max_delay=policy.get("max_delay", 60),
            retry_exceptions=policy.get("retry_exceptions", (Exception,)),
            logger=self.logger
        )
```

### 7.2 Async Error Handling for Live Trading

```python
class AsyncErrorHandler:
    """Error handling utility for asynchronous operations in live trading."""
    
    def __init__(self, logger, event_bus=None):
        self.logger = logger
        self.event_bus = event_bus
        
    async def handle_async_error(self, coro, error_context=None):
        """
        Execute an async coroutine with error handling.
        
        Args:
            coro: The coroutine to execute
            error_context (dict, optional): Context information for errors
        """
        try:
            return await coro
        except Exception as e:
            # Log the error
            self.logger.error(
                f"Async operation error: {str(e)}",
                exc_info=True,
                extra={"context": error_context}
            )
            
            # Publish error event
            if self.event_bus:
                error_data = {
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "context": error_context or {}
                }
                
                if isinstance(e, ADMFException):
                    error_data.update({
                        "code": e.code,
                        "details": e.details,
                        "recoverable": e.recoverable
                    })
                    
                await self.event_bus.publish_async(Event(
                    EventType.ERROR,
                    {"error": error_data}
                ))
                
            # Re-raise appropriate exception
            if not isinstance(e, ADMFException):
                raise ADMFException.from_exception(
                    e,
                    message=f"Async operation failed: {str(e)}",
                    details=error_context
                ) from e
            raise
```

## 8. Usage Examples

### 8.1 Basic Error Handling Pattern

```python
def process_data(self, data):
    with self.with_error_boundary("process_data"):
        # Validate inputs
        if not data:
            raise DataError("Empty data received", recoverable=True)
        
        try:
            # Process the data
            result = self._do_process_data(data)
            return result
        except Exception as e:
            # Convert to domain-specific exception
            raise DataProcessingError(
                f"Error processing data: {str(e)}",
                details={"data_size": len(data)},
                recoverable=False
            ) from e
```

### 8.2 Retry Pattern

```python
def fetch_market_data(self, symbol):
    with self.with_retry("fetch_market_data"):
        response = self.data_client.get_data(symbol)
        if response.status_code != 200:
            raise DataSourceError(
                f"Failed to fetch data for {symbol}: {response.status_code}",
                recoverable=True
            )
        return response.json()
```

### 8.3 Error Event Pattern

```python
def on_order_rejected(self, order, reason):
    # Create detailed error information
    error_details = {
        "order_id": order.id,
        "symbol": order.symbol,
        "quantity": order.quantity,
        "rejection_reason": reason
    }
    
    # Log structured error
    self.error_logger.log_error(
        OrderRejectedError(
            f"Order {order.id} rejected: {reason}",
            details=error_details,
            recoverable=True
        ),
        context={"component": "execution", "operation": "order_submission"}
    )
```

### 8.4 Error Injection Testing

```python
class RetryMechanismTest(ErrorInjectionTestCase):
    def setUp(self):
        super().setUp()
        self.data_handler = DataHandler("test_data_handler")
        
    def test_data_fetch_retry(self):
        # Register injection point at data fetch operation
        self.register_error_point(
            "data_handler.fetch",
            exception_class=DataSourceError,
            probability=1.0  # Always inject on first try
        )
        
        # Configure decreasing probability to allow eventual success
        def decrease_probability(attempt):
            if attempt > 1:
                self.error_injector.set_probability("data_handler.fetch", 0.0)
                
        # Test with retry mechanism
        with RetryContext(
            max_attempts=3,
            retry_exceptions=(DataSourceError,),
            on_retry=lambda e, attempt, delay: decrease_probability(attempt)
        ):
            result = self.data_handler.fetch_data("AAPL")
            
        # Should succeed after retry
        self.assertIsNotNone(result)
```

## 9. Implementation Plan

### 9.1 Phase 1: Exception Hierarchy

1. Implement the `ADMFException` base class
2. Develop domain-specific exception classes
3. Document exception types and handling guidelines

### 9.2 Phase 2: Error Boundaries and Propagation

1. Implement the `ErrorBoundary` class
2. Enhance `Component` base class with error handling
3. Create error event types and publishing utilities
4. Integrate with the event system

### 9.3 Phase 3: Retry Mechanisms

1. Implement retry decorators and context managers
2. Create standard retry policies for common operations
3. Integrate retry mechanisms with components

### 9.4 Phase 4: Testing and Monitoring

1. Implement error injection framework
2. Create error monitoring and logging utilities
3. Develop test utilities for error handling

## 10. Benefits

1. **Clarity**: Clear exception hierarchy improves code readability
2. **Consistency**: Standardized error handling patterns across components
3. **Recoverability**: Retry mechanisms for handling transient failures
4. **Visibility**: Structured error logging and monitoring for troubleshooting
5. **Testability**: Error injection capabilities for testing error handling

## 11. Best Practices

1. **Categorize Errors**: Use appropriate exception types for different error categories
2. **Provide Context**: Include relevant context in exceptions for debugging
3. **Specify Recoverability**: Mark exceptions as recoverable when retry is possible
4. **Use Error Boundaries**: Contain errors within logical component boundaries
5. **Log Structured Data**: Include structured data in error logs for analysis
6. **Document Recovery Paths**: Document standard recovery strategies for errors
7. **Test Error Handling**: Use error injection testing to verify error handling logic
8. **Monitor Error Rates**: Track error frequencies to identify systemic issues