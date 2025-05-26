# ADMF Core: Analytics and Reporting Frameworks

## 1. Overview

The ADMF-Trader Core module provides foundational frameworks for system-wide analytics and reporting. These frameworks aim to:
* Standardize the collection, calculation, and analysis of key performance indicators (KPIs) and operational metrics.
* Offer a flexible system for generating, scheduling, and distributing reports to transform raw data into actionable insights.

This document outlines the idealized design for the Core Metrics Framework and the Core Reporting System.

## 2. Core Metrics Framework

The Metrics Framework provides a comprehensive system for defining, collecting, and managing various types of metrics across ADMF-Trader.

### 2.1. Motivation & Principles

* **Standardization**: Ensure consistent metric definitions and calculations.
* **Comprehensiveness**: Support diverse metric types for strategy performance, system operation, and risk.
* **Accessibility**: Allow easy registration, retrieval, and aggregation of metrics.
* **Extensibility**: Enable definition of custom metrics.
* **Analysis Support**: Facilitate both real-time monitoring and historical analysis.

### 2.2. Core Metric Concepts

* **`MetricType` (Enum)**: Defines the nature of the metric.
    * `COUNTER`: Monotonically increasing value (e.g., number of trades).
    * `GAUGE`: Value that can arbitrarily go up or down (e.g., current CPU usage, portfolio value).
    * `HISTOGRAM`: Tracks the statistical distribution of a set of values (e.g., trade P&L distribution).
    * `TIMER`: Measures durations of operations, often using a Histogram internally.
    * `RATIO`: Represents a ratio between two other metrics.
    * `COMPOSITE`: A metric composed of multiple other metrics.
* **`MetricDimension` (Enum)**: Categorizes metrics for organization and filtering.
    * `PERFORMANCE`: Strategy or portfolio performance.
    * `RISK`: Risk-related measures.
    * `SYSTEM`: System health and performance (CPU, memory, queues).
    * `OPERATIONAL`: Operational aspects (error rates, uptime).
    * `CUSTOM`: User-defined or application-specific dimensions.
* **`MetricPeriod` (Enum)**: Specifies the time aggregation period for a metric.
    * `TICK`, `MINUTE`, `HOUR`, `DAY`, `WEEK`, `MONTH`, `YEAR`, `ALL` (all-time), `CUSTOM`.
* **`MetricValue` (Class)**: A container for a recorded metric observation.
    * Attributes: `value` (the actual metric data), `timestamp`, `metadata` (optional key-value pairs for context).

### 2.3. `Metric` Base Class (Idealized Design)

An abstract base class for all specific metric types.
* **Attributes**: `name` (unique identifier), `description`, `metric_type` (from `MetricType`), `dimension`, `period`, `unit` (e.g., "percent", "seconds", "count"), `tags` (dictionary for further categorization), `values` (list of `MetricValue` objects).
* **Methods**:
    * `record(value, timestamp=None, metadata=None)`: Adds a new `MetricValue`.
    * `get_latest()`: Returns the most recent `MetricValue`.
    * `get_values(start_time=None, end_time=None)`: Returns a list of `MetricValue` objects within a time range.
    * `reset()`: Clears the history of recorded values.
    * `as_series()`: Converts recorded values to a pandas Series (time-indexed).
    * `as_dict()`: Serializes the metric's current state and definition.

### 2.4. Specific Metric Implementations (Idealized Design)

These classes extend `Metric` and implement logic specific to their type:

* **`CounterMetric(Metric)`**:
    * `increment(amount=1, ...)`: Increases the counter.
    * `get_count()`: Returns the current count.
* **`GaugeMetric(Metric)`**:
    * `set(value, ...)`: Sets the current value of the gauge.
    * `get_average()`, `get_min()`, `get_max()`: Calculate statistics over recorded values.
* **`HistogramMetric(Metric)`**:
    * `observe(value, ...)`: Records a value and updates histogram buckets.
    * Configuration: `buckets` (list defining bucket boundaries).
    * `get_count()`, `get_sum()`, `get_average()`, `get_bucket_counts()`, `get_percentile(p)`.
* **`TimerMetric(Metric)`**:
    * Often uses a `HistogramMetric` internally to store durations.
    * `start()`: Returns a `TimerContext` manager.
    * `record_duration(duration, ...)`: Manually records a duration.
    * Methods for count, average duration, percentiles (e.g., p50, p90, p99).
* **`TimerContext` (Context Manager)**:
    * `__enter__()`: Records start time.
    * `__exit__()`: Calculates duration, records it in the parent `TimerMetric`, and can add metadata like exception info.

### 2.5. `MetricsRegistry` (Singleton - Idealized Design)

A central, singleton registry for all metrics defined and used within the system.
* **Purpose**:
    * Ensures unique metric names.
    * Provides factory methods for creating standard metric types (e.g., `create_counter()`, `create_gauge()`). This ensures metrics are consistently defined and registered.
    * Allows lookup of metrics by name, dimension, or tags.
* **Methods**:
    * `create_counter(name, description, ...)`: Creates and registers a `CounterMetric`.
    * `create_gauge(name, description, ...)`: Creates and registers a `GaugeMetric`.
    * (Similar methods for `HistogramMetric`, `TimerMetric`).
    * `get_metric(name)`: Retrieves a registered metric.
    * `get_metrics_by_dimension(dimension)`: Retrieves all metrics of a given dimension.
    * `get_metrics_by_tags(tags_dict)`: Retrieves metrics matching specific tags.
    * `get_all_metrics()`: Returns a list of all registered metrics.
    * `reset_all_metrics()`: Calls `reset()` on all registered metrics.

### 2.6. Domain-Specific Metric Groups (Conceptual)

While the Core module provides the *framework*, specific modules or the application layer would use this framework to define logical groups of metrics. Examples from the design:
* **`StrategyMetrics`**: Encapsulates metrics specific to evaluating a trading strategy (e.g., total return, annualized return, volatility, Sharpe ratio, max drawdown, trade count, win rate, profit factor, average trade return, strategy execution time). Instantiated per strategy.
* **`SystemMetrics`**: Tracks overall system performance (e.g., CPU usage, memory usage, disk I/O, network I/O, queue sizes, queue latencies).
* **`OperationalMetrics`**: Monitors operational health (e.g., uptime, availability, error counts/rates, request counts/durations, component health, data latency, data quality).

These classes would use the `MetricsRegistry` to create and manage their respective metrics, often prefixing metric names (e.g., `strategy_id.total_return`) and using common tags.

### 2.7. `CustomMetricsBuilder` & `CompositeMetric` (Idealized Design)

* **`CustomMetricsBuilder`**: A utility to help create sets of custom metrics under a specific namespace with common tags, using the `MetricsRegistry`.
* **`CompositeMetric`**: A way to define a metric that is itself composed of several other registered metrics, potentially with a calculation function to derive its value.

## 3. Core Reporting System

The Reporting System provides a framework for generating, scheduling, and distributing reports, transforming raw data and metrics into structured, human-readable insights.

### 3.1. Motivation & Principles

* **Data Synthesis**: Convert trading data, metrics, and events into meaningful reports.
* **Audience-Specificity**: Allow different report structures for different user roles.
* **Flexibility**: Support various time horizons, output formats, and delivery methods.
* **Efficiency**: Handle potentially large data volumes.

### 3.2. Core Reporting Concepts

* **`ReportFormat` (Enum)**: Defines output formats.
    * `HTML`, `PDF`, `JSON`, `CSV`, `EXCEL`, `TEXT`.
* **`ReportPeriod` (Enum)**: Specifies the time period a report covers.
    * `INTRADAY`, `DAILY`, `WEEKLY`, `MONTHLY`, `QUARTERLY`, `YEARLY`, `CUSTOM`.
* **`ReportType` (Enum)**: Categorizes the type or purpose of the report.
    * `STRATEGY_PERFORMANCE`, `SYSTEM_STATUS`, `OPERATIONAL`, `RISK`, `CUSTOM`.

### 3.3. `ReportTemplate` Class (Idealized Design)

* **Purpose**: Base class for report templates, abstracting the template engine (e.g., Jinja2).
* **Attributes**: `name`, `description`, `template_path` or `template_string`.
* **Methods**: `render(data_dict)`: Populates the template with provided data and returns the rendered content (typically HTML).

### 3.4. `ReportDefinition` Class (Idealized Design)

* **Purpose**: Defines the structure, data sources, processing steps, and output format for a specific type of report.
* **Attributes**: `name`, `description`, `report_type` (enum), `template` (a `ReportTemplate` instance), `format` (enum), `period` (enum).
* **Data Pipeline**:
    * `data_providers`: A list of functions or objects that supply data for the report.
    * `post_processors`: A list of functions or objects that transform or enrich the collected data (e.g., chart generators).
* **Methods**:
    * `add_data_provider(provider)`, `add_post_processor(processor)`.
    * `collect_data(parameters)`: Invokes all data providers.
    * `process_data(data, parameters)`: Applies all post-processors.
    * `generate_report(parameters)`: Orchestrates data collection, processing, and rendering through the template.
    * Internal methods for format conversion (e.g., `_html_to_pdf`, `_data_to_csv`).

### 3.5. `ReportGenerator` Class (Idealized Design)

* **Purpose**: A central manager for all `ReportDefinition`s and `ReportTemplate`s in the system.
* **Methods**:
    * `register_template(template_instance)`.
    * `register_report(report_definition_instance)`.
    * `create_report(report_name, parameters)`: Finds the `ReportDefinition` by name and calls its `generate_report` method.
    * `get_report_definitions()`, `get_templates()`.

### 3.6. Report Scheduling System (Idealized Design)

An automated system for generating and delivering reports based on schedules.
* **`ReportSchedule` Class**:
    * Attributes: `report_name` (to be generated), `schedule_type` (e.g., 'daily' at '08:00', 'weekly' on 'Monday at 09:00', 'interval' every '1h'), `parameters` (for report generation), `delivery_method` (enum), `delivery_parameters` (e.g., email recipients, file path), `enabled`.
* **`ReportDeliveryMethod` (Enum)**: `EMAIL`, `FILE` (save to disk), `HTTP` (POST to an endpoint), `CONSOLE`, `CUSTOM`.
* **`ReportScheduler` Class**:
    * Manages a list of `ReportSchedule` instances.
    * Uses a scheduling library (e.g., Python's `schedule`) to trigger report generation jobs.
    * `_run_report(schedule_id)`: Called by the scheduler; uses `ReportGenerator` to create the report and then `_deliver_report`.
    * `_deliver_report(report_content, schedule_obj)`: Implements various delivery methods (e.g., `_deliver_email` using `smtplib`, `_deliver_file`).
    * Methods to `add_schedule`, `enable_schedule`, `disable_schedule`, `start/stop` the scheduler thread.

### 3.7. Domain-Specific Report Components (Conceptual Examples)

As with metrics, the Core module provides the reporting *framework*. Specific data providers, post-processors (like chart generators), and report definitions would be implemented in relevant modules or the application layer. Examples cited in the design:
* `StrategyPerformanceDataProvider`
* `StrategyPerformanceChartGenerator`
* `SystemStatusDataProvider`

### 3.8. Dynamic HTML Templates (Examples)

The design includes examples of HTML templates (using Jinja2-like syntax) for `StrategyPerformanceReport` and `SystemStatusReport`, demonstrating how data (summary metrics, charts as base64 images) can be presented.

## 4. Implementation Status & Gaps

The Core Metrics Framework and Reporting System, as detailed in `docs/core/analytics/`, represent significant idealized features.
* **Current `src/core/`**: These comprehensive frameworks (MetricsRegistry, specific Metric types, ReportGenerator, ReportScheduler, etc.) are **not currently implemented** within the `src/core/` Python files provided.
* **Relation to `Analytics` submodule in `docs/ARCH.MD`**: `docs/ARCH.MD` mentions "Core: Foundation services including ... and the Analytics submodule for performance measurement and reporting." The designs in `docs/core/analytics/` fulfill this vision for an "Analytics submodule" within Core.

The `Analytics Component Framework` document (`docs/strategy/ANALYTICS_COMPONENT_FRAMEWORK.md`) also describes `AnalyticsComponent`, `Classifier`, and `MetaLabeler` which seem to be intended to live in `src/strategy/analytics/`. These components would be *users* or *producers* of data for the Core Metrics Framework and might be sources for the Reporting System. The Core frameworks discussed here provide the underlying tools for those higher-level analytical components.

This consolidated document provides a target design for building robust analytics and reporting capabilities into the ADMF-Trader's core.
