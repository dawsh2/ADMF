# ADMF Core: Infrastructure Services

## 1. Overview

The ADMF-Trader Core module provides a suite of essential infrastructure services designed to ensure system reliability, observability, robustness, and developer productivity. These services include comprehensive frameworks for logging, monitoring, error handling, debugging, testing, and validation. This document outlines the idealized design for these critical infrastructure components.

## 2. Logging Framework

A robust logging framework is essential for diagnostics, auditing, and understanding system behavior.

### 2.1. Architectural Principles

* **Hierarchical Loggers**: Utilize a standard logger hierarchy (e.g., `SystemLogger.ConfigurationLogger`, `ComponentLogger.StrategyLogger`).
* **Standardized Logging Levels**: Consistent use of defined log levels.
* **Structured Logging**: Output logs in a machine-parsable format (e.g., JSON) with consistent fields for easier analysis and integration with log management systems.
* **Correlation Tracking**: Use correlation IDs to trace operations across multiple components and services.
* **Performance Considerations**: Logging should be efficient to minimize impact on application performance.

### 2.2. Logging Levels and Usage

| Level    | Usage Guidelines                                                                 |
| :------- | :------------------------------------------------------------------------------- |
| TRACE    | Ultra-detailed for method entry/exit, fine-grained state changes. (Custom Level) |
| DEBUG    | Information useful for debugging specific component behavior.                   |
| INFO     | General information about system operation, significant lifecycle events.        |
| WARNING  | Potential issues or unexpected situations that don't halt operation.            |
| ERROR    | Errors that affect current operation but might be recoverable or isolated.       |
| CRITICAL | Severe errors requiring immediate attention, potentially system-wide impact.     |

*Your `src/core/logging_setup.py` provides initial configuration for standard Python logging levels.*

### 2.3. Idealized `StructuredLogger`

* **Purpose**: A wrapper around the standard logging library to enforce structured output and add common contextual information.
* **Key Fields**:
    * `timestamp`: ISO 8601 UTC.
    * `level`: (e.g., INFO, ERROR).
    * `logger`: Name of the logger (e.g., `component.MyStrategy`).
    * `message`: The log message.
    * `hostname`: Machine hostname.
    * `thread`: Thread name/ID.
    * `correlation_id`: For tracing related operations.
    * `execution_mode`: Current `RunMode`.
    * `component_config_key`, `instance_name`: From `ComponentBase`.
    * `extra`: Custom key-value pairs specific to the log event.
* **Output**: JSON format is preferred for easy parsing by log aggregation tools.

### 2.4. Idealized `CorrelationContext`

A context manager to automatically manage and propagate `correlation_id`s for logs generated within its scope. This helps trace a single operation or event flow across multiple component interactions.

```python
# Conceptual Usage
# logger = StructuredLogger("MyComponentLogger")
# with CorrelationContext(logger, correlation_id="trace-abc-123"):
#     logger.info("Starting operation X")
#     # ... other operations ...
#     logger.info("Finished operation X") # Will have same correlation_id
```

### 2.5. Idealized @trace_method Decorator

A decorator to automatically log method entry, arguments, exit, return values (optional), and execution time at the TRACE level.

### 2.6. Log Output Configuration

Support for multiple handlers:
* Console output (configurable level).
* File output (configurable path, level, rotation policies from LOGGING_OPTIMIZATION.MD).
* Remote output (e.g., Syslog, Elasticsearch/ELK stack).

Configuration should allow setting log levels per module or component.

## 3. Monitoring Framework

The monitoring framework provides tools for observing system health, performance, and resource utilization.

### 3.1. Health Checks (Idealized Design)

* **HealthCheck Base Class**: Defines an interface for health check logic (check(), success(), warning(), error(), to_dict()).
* **ComponentHealthCheck**: A specific HealthCheck that verifies the status of a ComponentBase instance (e.g., initialized, running).
* **HealthCheckManager**:
    * Registers all health checks in the system.
    * Periodically runs all checks.
    * Aggregates results to provide an overall system health status.
    * Can expose health status via an HTTP endpoint or publish health events.
### 3.2. Performance Metrics Collection (Idealized Design)

* **MetricsCollector**:
    * `record_value(name, value, tags=None)`: Records a generic metric value.
    * `record_timing(name, duration_ms, tags=None)`: Specialized for duration/latency.
    * `record_count(name, count=1, tags=None)`: Specialized for counters.
    * Stores metrics (values, counts, sums, min/max) with optional tags for dimensionality.
    * Calculates statistics (mean, median, percentiles like p90, p95, p99).
* **PerformanceTracker**:
    * `@track_method(name=None, tags=None)` decorator: Automatically records execution time and success/failure of decorated methods.
    * `track_operation(name, tags=None)` context manager: For tracking custom code blocks.
    * Uses MetricsCollector to store the collected performance data.
### 3.3. ResourceMonitor (Idealized Design)

Monitors system-level and process-level resources using psutil:
* CPU usage (total and per-core).
* Memory usage (system total, available, process RSS/VMS).
* Disk usage (total, free).
* Process thread count.

Records these metrics periodically using the MetricsCollector.
### 3.4. Integration with External Monitoring Systems (Idealized Design)

* **Prometheus Exporter**: Exposes metrics collected by MetricsCollector in Prometheus format via an HTTP endpoint.
* **Elasticsearch (ELK) Exporter**: Sends structured logs to an Elasticsearch instance for centralized logging and analysis.
### 3.5. Alerting (AlertManager - Idealized Design)

* Centralized system for defining alert conditions based on metrics or health checks.
* Triggers notifications (e.g., email, Slack, webhook) when alert conditions are met.
## 4. Error Handling Framework

A comprehensive error handling strategy ensures robustness and aids in diagnostics.

### 4.1. ADMFException Base Class (Idealized Design)

All custom exceptions in ADMF-Trader should inherit from ADMFException.

**Attributes**:
* `message`: Human-readable error message.
* `code`: Optional unique error code for programmatic handling.
* `details`: Dictionary for additional contextual information.
* `recoverable`: Boolean indicating if the error is potentially recoverable.
* `context`: Execution context when the error occurred.
* `timestamp`.

**Methods**:
* `to_dict()`: For serialization (e.g., in logs or API responses).
* `from_exception(cls, exception, ...)`: Class method to wrap external exceptions.

Your `src/core/exceptions.py` defines `ADMFTraderError` and some specific errors, which is a good start towards this.
### 4.2. Exception Hierarchy (Idealized Design)

A structured hierarchy of specific exception types (e.g., ConfigurationError, DataError, ExecutionError, ComponentError, RiskError) inheriting from ADMFException. Each specialized exception should document common causes and potential recovery strategies.

### 4.3. ErrorBoundary Context Manager (Idealized Design)

A context manager (`with ErrorBoundary(...):`) to wrap logical blocks of code or component operations.

**Responsibilities**:
* Catch exceptions within its scope.
* Optionally transform caught exceptions into a standard ADMFException type.
* Log the error with context.
* Optionally publish an ERROR event to the EventBus.
* Optionally call a custom error handler.
* Control whether the exception is re-raised or suppressed.
### 4.4. Retry Mechanisms (Idealized Design)

For handling transient, recoverable errors:

* **@retry Decorator**: Simple decorator for retrying a function call with configurable attempts, backoff strategy (e.g., exponential), max delay, and specific retryable exceptions.
* **RetryContext Context Manager**: For more complex retry scenarios within a block of code, offering similar configurability to the decorator plus callbacks (on_retry, on_give_up).
* **RetryableOperation Class**: For operations requiring more complex state management across retries or more sophisticated retry policies.
### 4.5. Error Event Publishing

Standardized ERROR events should be published to the EventBus when significant errors occur, allowing system-wide error monitoring or reaction. These events should contain structured error information (e.g., from ADMFException.to_dict()).

### 4.6. Error Injection Framework for Testing (Idealized Design)

* **ErrorInjector**: A component to register "injection points" in the code.
* **@inject_fault(point_id, injector)**: A decorator to mark methods where errors can be conditionally injected during tests.
* Allows testing the robustness of error handling paths by deliberately triggering exceptions.
## 5. Debugging Framework [Idealized Design]

Provides tools for inspecting, tracing, and troubleshooting the system.

### 5.1. Core Components of the Debugging Framework

* **DebugManager**: Central coordinator for debugging activities. Activates/deactivates debug mode and other debugging components. Can manage debugging sessions and breakpoints.
* **ExecutionTracer**: Tracks the flow of key events (especially signal -> order -> fill) and associated state snapshots within a "trace."
* **StateInspector**: Captures and compares snapshots of component states at different times. Supports watchpoints on specific attributes.
* **EventRecorder**: Records a stream of system events (with filtering) that can be saved and replayed for reproducing scenarios or debugging.
* **DebugVisualizer**: Generates visual representations of event flows or component state diagrams (e.g., HTML, JSON for external tools).
### 5.2. Integration and Usage

* **Lifecycle Integration**: Debugging components are managed by the main Bootstrap system and are typically only active when debug mode is enabled in the configuration.
* **EventBus Hooks**: Debugging components (like ExecutionTracer, EventRecorder) hook into the EventBus to observe events.
* **Use Cases**: Tracing specific signal paths, inspecting state changes around an issue, replaying event sequences that lead to errors, setting conditional breakpoints.

Your `src/core/debug_logger.py` provides a very basic utility for temporary log level changes, but the full debugging framework is a significant idealized feature.

## 6. Testing Strategy Framework [Conceptual]

This outlines the approach to testing core components, rather than being a runtime framework itself.

### 6.1. Test Types and Structure

* **Unit Tests**: For isolated components, mocking dependencies.
* **Integration Tests**: For component interactions and event flows.
* **Property-Based Tests**: For verifying system properties with generated inputs (e.g., using hypothesis).
* **Performance Tests**: Benchmarks for critical paths, memory usage validation, concurrency stress tests.
### 6.2. Key Focus Areas for Core Testing

* **Component Isolation & Lifecycle**: Verifying reset() clears state, correct lifecycle transitions.
* **Event Propagation & Handling**: Ensuring events flow as expected and handlers are correctly invoked.
* **State Boundary Verification**: No state leakage between runs.
* **Critical Path Performance**: For EventBus, DataHandler, etc.
* **Concurrency**: Thread safety of shared core components.
### 6.3. Idealized Test Utilities and Fixtures

* **ComponentTestCase**: A base test class providing common setup (mock context, event bus).
* **TrackingEventBus**: An EventBus subclass that records published events and handler call sequences for assertion.
* **Test Data Generators**: For creating synthetic market data, events, etc.
## 7. Validation Framework [Idealized Design]

Provides tools for ensuring data integrity, system consistency, and correct behavior.

### 7.1. Core Components of the Validation Framework

* **ValidationManager**: Coordinates all validation operations, registers validators, and aggregates results.
* **ValidatorBase**: Abstract base class for all specific validators, defining a validate() method.
* **ValidationResult**: Stores the outcome (pass/fail, messages, details) of a single validator's checks.
* **ValidationSummary**: Aggregates multiple ValidationResult objects into a system-wide summary.
### 7.2. Categories of Validators (Conceptual Examples for Core)

* **SystemConsistencyValidator**:
    * **EventSystemValidator**: Checks EventBus configuration, handler registrations, critical event paths.
    * **ComponentStateValidator**: Verifies component initialization, reset() effectiveness.
    * **DependencyValidator**: Uses DependencyGraph to check for cycles, missing dependencies (as covered in 3_CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md).
* **ConfigurationValidator**: Validates global and component-specific configurations against schemas or rules.
* **DataIntegrityValidator** (if core data structures are validated here, otherwise in data module): Checks integrity of core data representations if any (e.g., internal event queues, context objects).
### 7.3. Integration

* **Lifecycle Hooks**: Validation can be triggered at specific lifecycle points (e.g., post-initialization, pre-run, post-run).
* **Configuration Loading**: ConfigValidator can be integrated into the configuration loading process.

This comprehensive suite of infrastructure services, once fully implemented, will provide a very strong foundation for developing, operating, and maintaining the ADMF-Trader system.


---
This draft for `7_CORE_INFRASTRUCTURE_SERVICES.md` is quite large, reflecting the number of important frameworks it covers. It attempts to synthesize the key design aspects from `LOGGING_MONITORING.MD`, `ERROR_HANDLING.md`, `DEBUGGING_FRAMEWORK.md`, `TESTING_STRATEGY.md`, and `VALIDATION_FRAMEWORK.md`.

Please review this extensive document. Given its size, you might want to focus on one major section at a time (e.g., Logging, then Monitoring, etc.) for detailed feedback. Let me know your thoughts. If you're happy with this, we've covered the primary consolidation targets for `docs/core/` that we planned!
