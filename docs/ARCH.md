# ADMF-Trader Detailed Architecture

This document provides an in-depth technical overview of the ADMF-Trader system architecture, describing each module's design patterns, data flows, and implementation details.

## Table of Contents
1. [Overview and Design Philosophy](#overview-and-design-philosophy)
2. [Main Entry Point](#main-entry-point)
3. [Core Module](#core-module)
4. [Data Module](#data-module)
5. [Strategy Module](#strategy-module)
6. [Risk Module](#risk-module)
7. [Execution Module](#execution-module)
8. [Event System](#event-system)
9. [Optimization Framework](#optimization-framework)
10. [Isolation Principles](#isolation-principles)

---

## Overview and Design Philosophy

ADMF-Trader is an algorithmic trading framework designed with a modular, event-driven architecture for systematic strategy development, testing, optimization, and deployment. The system is organized into distinct modules with well-defined responsibilities:

- **Core**: Foundation services including event bus, component lifecycle, configuration, dependency injection, and the Analytics submodule for performance measurement and reporting
- **Data**: Market data processing, train/test splitting, and data source management
- **Strategy**: Signal generation, optimization frameworks, and trading logic
- **Risk**: Position sizing, risk limits, and portfolio management
- **Execution**: Order management, broker simulation, backtest coordination and trade execution

### Key Architectural Principles

1. **Event-Driven Communication**: Components interact through an event system, enabling loose coupling and extensibility
2. **Component-Based Design**: All system elements follow a consistent lifecycle pattern with explicit state transitions
3. **Dependency Injection**: Components receive dependencies via DI container, promoting testability and modularity
4. **Clear Interface Boundaries**: Modules expose well-defined interfaces to prevent implementation leakage
5. **State Isolation**: Careful management of state for reliable optimization and backtesting
6. **Compositional Architecture**: Complex components are built by combining simpler ones
7. **Configuration Driven**: The system is managed through a central configuration, rather than direct code modification

### System Flow

The typical flow of data through the system:

1. **Data Flow**: `DataHandler` loads market data and emits `BAR` events
2. **Signal Generation**: `Strategy` consumes `BAR` events and emits `SIGNAL` events
3. **Risk Management**: `RiskManager` consumes `SIGNAL` events and emits `ORDER` events
4. **Order Execution**: `Broker` consumes `ORDER` events and emits `FILL` events
5. **Portfolio Tracking**: `Portfolio` consumes `FILL` events and updates positions/equity

---

## Main Entry Point

The main.py file serves as the central entry point for the ADMF-Trader system, providing a command-line interface that handles configuration loading and system initialization.

### Command-Line Interface

```python
# Handle command-line arguments
parser = argparse.ArgumentParser(description='ADMF-Trader CLI')
parser.add_argument('--config', required=True, help='Path to configuration file')
parser.add_argument('--bars', type=int, help='Limit processing to specified number of bars')
# ... additional arguments
```

Arguments can be added to override config settings, such as --bars BARS, which allows the user to specify that only the first BARS bars of data should be used. This enables shortened cycles while developing and debugging.


### Bootstrap Process

```python
# Set up the bootstrap system
bootstrap = Bootstrap(
    config_files=[args.config],
    debug=args.debug,
    log_level=logging.WARNING if not args.verbose and not args.debug else logging.INFO,
    log_file=args.log_file or "trading.log"
)

# Initialize system
container, config = bootstrap.setup()
```

The bootstrap system initializes the entire application:
1. Loads configuration from specified YAML file
2. Sets up the dependency injection container
3. Configures logging based on command-line arguments
4. Registers and initializes all system components


### Event System Initialization

Before any modules are loaded, main.py initializes and standardizes the event system to ensure consistent event handling throughout the application:

```python
# Initialize standardized event system before any modules load
try:
    from src.core.events import initialize_standardized_events
    standardized_count = initialize_standardized_events()
except ImportError:
    print("⚠️ Failed to initialize standardized event system, using default behavior")
```

This ensures that all components use the same event system implementation, preventing inconsistencies in event handling.

### Error Handling and Logging

The main.py file implements comprehensive error handling and logging:

```python
try:
    # Configure logging based on arguments
    configure_logging(debug=args.debug, log_file=args.log_file)
    
    # Run the appropriate mode
    # ...
    
except Exception as e:
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return 1
```

This ensures that exceptions are properly logged with stack traces, helping with debugging and error resolution.

---

## Core Module

The Core module provides foundational infrastructure services that all other modules depend on, implementing critical patterns for component lifecycle, configuration, dependency injection, and event management.

### Component Lifecycle System

```
Component (Abstract)
  └── initialize(context)
  └── start()
  └── stop()
  └── reset()
  └── teardown()
  └── initialize_event_subscriptions()
```

All system components inherit from the `Component` base class, which enforces a consistent lifecycle pattern with clearly defined states:

1. **CREATED**: Initial state after instantiation
2. **INITIALIZED**: After dependencies are injected and component is ready
3. **RUNNING**: During active operation
4. **STOPPED**: After operation completes
5. **DISPOSED**: After resources are released

Each lifecycle method has specific responsibilities:
- `initialize(context)`: Set up the component with dependencies from context
- `start()`: Begin component operation
- `stop()`: End component operation
- `reset()`: Clear internal state while preserving configuration
- `teardown()`: Release resources for garbage collection

Implementing this consistent lifecycle ensures state can be properly reset between test runs, a critical requirement for optimization where multiple backtests run sequentially.

### Dependency Injection Container

```python
class Container:
    def register(self, name, component, singleton=True)
    def register_instance(self, name, instance)
    def register_factory(self, name, factory)
    def get(self, name)
    def has(self, name)
    def reset()
```

The Container implements a service locator pattern for dependency management:

- **Instance Registration**: Centralized registry of singleton components
- **Lazy Loading**: Components are instantiated only when needed
- **Factory Support**: Dynamic component creation through factory functions
- **Circular Dependency Detection**: Prevents infinite recursion
- **Resource Management**: Proper disposal of managed resources

By controlling component creation and lifecycle through the DI container, the system ensures proper isolation between optimization runs.

### Configuration System

```python
class Config:
    def load(self, config_file)
    def get(self, key, default=None)
    def set(self, key, value)
    def as_dict()
    def update(self, other_config)
    def load_env(prefix)
```

The configuration system provides hierarchical access to settings:

- **YAML Loading**: Parses configuration from YAML files
- **Environment Variables**: Overrides settings with environment variables
- **Schema Validation**: Validates configuration against defined schemas
- **Dot Notation**: Supports hierarchical access (`config.get('data.sources.0.symbol')`)
- **Type Conversion**: Automatic conversion of values to appropriate types
- **Default Values**: Graceful handling of missing configuration

### Event System

```
EventBus
  └── publish(event)
  └── subscribe(event_type, handler)
  └── unsubscribe(event_type, handler)
  └── reset()
```

The event system enables loosely coupled communication between components:

- **Event Types**: Standard event categories with defined schemas
- **Publish/Subscribe**: Components can publish and subscribe to events
- **Handler Registration**: Multiple handlers can be registered per event type
- **Event Context**: Events can be confined to specific contexts
- **Deduplication**: Prevents duplicate event processing

### Bootstrap System

```python
class SystemBootstrap:
    def setup()
    def register_hook(hook_point, callback)
    def teardown()
    def _setup_components(container, config)
```

The bootstrap system orchestrates the initialization of the entire application:

- **Component Discovery**: Automatically finds and registers components
- **Initialization Order**: Ensures proper dependency resolution
- **Configuration Loading**: Handles configuration files and environment variables
- **Extension Points**: Hook system for customizing the bootstrap process
- **Resource Management**: Proper cleanup of resources during shutdown

### Analytics Submodule

```python
class PerformanceAnalytics(Component):
    def calculate_returns(equity_curve)
    def calculate_sharpe_ratio(returns)
    def calculate_drawdown(equity_curve)
    def calculate_trade_statistics(trades)
    def generate_report(equity_curve, trades)
```

The Analytics submodule, part of the Core module, provides performance measurement and reporting:

- **Returns**: Absolute, percentage, and annualized returns
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, etc.
- **Drawdown Analysis**: Maximum drawdown, average drawdown, recovery time
- **Trade Statistics**: Win rate, profit factor, average win/loss
- **Report Generation**: Creates performance reports with charts and statistics

```python
class ReportGenerator:
    def generate_text_report(results)
    def generate_html_report(results)
    def generate_equity_chart(equity_curve)
    def generate_drawdown_chart(drawdowns)
```

The report generation system creates performance reports:

- **Text Reports**: Plain text summaries of performance
- **HTML Reports**: Interactive reports with charts and tables
- **Charts**: Equity curves, drawdowns, trade distributions
- **Trade Analysis**: Detailed breakdowns of individual trades
- **Parameter Analysis**: Results of parameter optimization

---

## Data Module

The Data module is responsible for providing market data to the system and maintaining clean separation between training and testing datasets, a critical requirement for preventing look-ahead bias in optimization.

### Data Handler

```
DataHandler (Abstract)
  └── initialize(context)
  └── update_bars()
  └── get_latest_bar(symbol)
  └── get_latest_bars(symbol, N=1)
  └── reset()
  └── set_active_split(split_name)
```

The DataHandler is responsible for loading and providing market data:

- **Data Loading**: Loads data from CSV files or other sources
- **Data Preprocessing**: Normalizes and cleans market data
- **Bar Publication**: Publishes bar events to the event system
- **Data Access**: Provides methods to access latest bars
- **Train/Test Splitting**: Manages separate datasets for training and testing

### Data Models

```python
class Bar:
    def __init__(self, timestamp, symbol, open, high, low, close, volume, timeframe)
    def to_dict()
    @classmethod
    def from_dict(cls, data_dict)
```

The Bar class represents OHLCV (Open, High, Low, Close, Volume) market data with standardized fields and conversion methods.

### Historical Data Handler

```python
class HistoricalDataHandler(DataHandler):
    def _load_data()
    def _split_data(df, symbol)
    def update_bars()
    def setup_train_test_split(method, train_ratio, split_date)
```

The HistoricalDataHandler implements the DataHandler interface for backtesting:

- **CSV Loading**: Reads data from CSV files
- **Train/Test Splitting**: Implements various splitting methods
  - Ratio-based: Split at a percentage point (e.g., 70/30)
  - Date-based: Split at a specific date
  - Fixed-period: Split after a specified number of bars
- **Bar Iteration**: Advances through the dataset one bar at a time
- **Memory Isolation**: Creates deep copies of data for train/test isolation

### Time Series Splitter

```python
class TimeSeriesSplitter:
    def split(df)
    def _split_by_ratio(df, ratio)
    def _split_by_date(df, date)
    def _split_by_periods(df, periods)
```

The TimeSeriesSplitter handles the creation of training and testing datasets:

- **Split Verification**: Ensures no data leakage between datasets
- **Isolation**: Creates completely separate DataFrame copies
- **Timestamps Validation**: Checks for overlapping timestamps
- **Memory Tracking**: Logs DataFrame memory addresses for debugging

---

## Strategy Module

The Strategy module defines the trading logic that analyzes market data and generates trading signals.

### Strategy Base Class

```
Strategy (Abstract)
  └── initialize(context)
  └── on_bar(event)
  └── calculate_signals(bar)
  └── emit_signal(symbol, direction, price, quantity)
  └── reset()
```

The Strategy base class provides the framework for trading strategies:

- **State Management**: Maintains internal state for indicators and signals
- **Event Handling**: Processes market data events
- **Signal Generation**: Creates standardized signal events
- **Parameter Management**: Handles strategy parameters and optimization
- **Lifecycle Management**: Resets state between runs

### Component Architecture

```
Component (Base)
  └── Strategy (Abstract)
      └── CompositeStrategy
      └── MultipleTimeframeStrategy
      └── ConcreteStrategies (e.g., SimpleMACrossoverStrategy)
```

Strategies can be composed hierarchically:

- **Strategy Components**: Reusable elements like indicators and filters
- **Composite Pattern**: Strategies that contain other strategies
- **Timeframe Handling**: Support for multiple timeframes with aggregation

### Strategy Factory

```python
class StrategyFactory:
    def create_strategy(name, **kwargs)
    def register_strategy(name, strategy_class)
    def discover_strategies(directory)
```

The StrategyFactory manages strategy discovery and instantiation:

- **Dynamic Loading**: Discovers strategy implementations at runtime
- **Instantiation**: Creates strategy instances with proper configuration
- **Parameter Mapping**: Maps configuration parameters to strategy properties
- **Dependency Injection**: Injects required dependencies

### Strategy Implementation Example

```python
class SimpleMACrossoverStrategy(Strategy):
    def __init__(self, name, config=None):
        super().__init__(name, config)
        # Get parameters with defaults
        self.fast_window = self.get_parameter('fast_window', 10)
        self.slow_window = self.get_parameter('slow_window', 30)
        self.position_size = self.get_parameter('position_size', 100)
        
        # Initialize state
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
        
    def on_bar(self, event):
        bar = event.data
        symbol = bar.symbol
        
        # Update price history
        if symbol not in self.prices:
            self.prices[symbol] = []
        self.prices[symbol].append(bar.close)
        
        # Wait until we have enough data
        if len(self.prices[symbol]) < self.slow_window:
            return
            
        # Calculate indicators
        self.fast_ma[symbol] = np.mean(self.prices[symbol][-self.fast_window:])
        self.slow_ma[symbol] = np.mean(self.prices[symbol][-self.slow_window:])
        
        # Check for crossover
        if len(self.prices[symbol]) <= self.slow_window:
            return
            
        # Get current position
        current_position = self.current_position.get(symbol, 0)
        
        # Check for buy signal - fast MA crosses above slow MA
        if self.fast_ma[symbol] > self.slow_ma[symbol] and current_position <= 0:
            self.emit_signal(
                symbol=symbol,
                direction="BUY",
                quantity=self.position_size,
                price=bar.close,
                metadata={"reason": "fast_ma_above_slow_ma"}
            )
            self.current_position[symbol] = self.position_size
            
        # Check for sell signal - fast MA crosses below slow MA
        elif self.fast_ma[symbol] < self.slow_ma[symbol] and current_position >= 0:
            self.emit_signal(
                symbol=symbol,
                direction="SELL",
                quantity=self.position_size,
                price=bar.close,
                metadata={"reason": "fast_ma_below_slow_ma"}
            )
            self.current_position[symbol] = -self.position_size
            
    def reset(self):
        """Reset strategy state between optimization runs."""
        super().reset()
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
```

---

## Risk Module

The Risk module manages position sizing, risk control, and portfolio tracking.

### Risk Manager

```
RiskManagerBase (Abstract)
  └── on_signal(event)
  └── size_position(signal)
  └── validate_order(order)
  └── emit_order(order_data)
```

The Risk Manager converts signals to orders with appropriate risk controls:

- **Signal Processing**: Extracts information from signal events
- **Position Sizing**: Calculates appropriate position sizes
- **Risk Validation**: Applies risk limits to proposed orders
- **Order Generation**: Creates order events for valid trades
- **Statistics Tracking**: Tracks risk metrics and order generation

### Position Sizing Strategies

```
PositionSizer (Abstract)
  └── FixedSizer
  └── PercentEquitySizer
  └── PercentRiskSizer
  └── KellySizer
  └── VolatilitySizer
```

Position Sizers calculate appropriate trade sizes:

- **Fixed Size**: Uses a constant number of shares/contracts
- **Percent of Equity**: Sizes based on portfolio equity percentage
- **Percent Risk**: Sizes based on stop distance and risk percentage
- **Kelly Criterion**: Optimal size based on win rate and win/loss ratio
- **Volatility Sizing**: Adjusts size based on market volatility

### Risk Limits

```
RiskLimit (Abstract)
  └── MaxPositionSizeLimit
  └── MaxExposureLimit
  └── MaxDrawdownLimit
  └── MaxLossLimit
  └── MaxPositionsLimit
```

Risk Limits enforce trading constraints:

- **Position Size**: Caps size of individual positions
- **Total Exposure**: Limits overall market exposure
- **Drawdown Control**: Reduces trading as drawdown increases
- **Loss Limits**: Suspends trading at maximum loss thresholds
- **Position Count**: Limits number of concurrent positions

### Portfolio Management

```
PortfolioManager
  └── on_fill(event)
  └── on_bar(event)
  └── update_positions(bar)
  └── calculate_equity()
  └── get_position(symbol)
```

The Portfolio Manager tracks positions and equity:

- **Position Tracking**: Maintains current positions and their details
- **Equity Calculation**: Updates portfolio equity based on prices
- **Cash Management**: Tracks available capital
- **Trade Tracking**: Records trade history and statistics
- **Performance Metrics**: Calculates returns, drawdowns, etc.

### Position Tracking

```python
class Position:
    def update(direction, quantity, price)
    def calculate_pnl()
    def mark_to_market(price)
    def close(price)
```

The Position class represents individual security positions:

- **Position Updates**: Handles additions and reductions
- **P&L Calculation**: Computes realized and unrealized P&L
- **Cost Basis Tracking**: Maintains accurate average price
- **Market Value**: Updates based on current prices
- **Position Direction**: Handles long, short, and flat positions


### Note:
The Risk module provides 'sane defaults' that the configuration will default to if not specified. The module can also act as a passthrough for debugging strategy performance without interference from risk management.

### Passthrough Configuration

The Risk module supports a "passthrough" mode for strategy testing:

```python
# Configure Risk module as passthrough
risk_config = {
    'class': 'PassthroughRiskManager',
    'parameters': {
        'enabled': True
    }
}
```

When configured as a passthrough, the Risk module will:
- Forward signals to orders without applying risk limits
- Generate orders with exact requested quantities
- Skip position sizing calculations
- Maintain portfolio tracking for analysis

---

## Execution Module

The Execution module handles order processing, market simulation and backtest coordination.

### Broker

```
BrokerBase (Abstract)
  └── on_order(event)
  └── process_order(order)
  └── check_fill_conditions(order, bar)
  └── execute_order(order, price)
```

The Broker processes orders and generates fills:

- **Order Reception**: Receives order events from the event bus
- **Fill Conditions**: Determines if orders can be filled based on prices
- **Execution Price**: Applies slippage to determine fill price
- **Commission Calculation**: Computes trading costs
- **Fill Generation**: Creates and publishes fill events

### Passthrough Broker

The Execution module supports a "passthrough" broker implementation:

```python
# Configure Execution module with passthrough broker
execution_config = {
    'broker': {
        'class': 'PassthroughBroker',
        'parameters': {
            'enabled': True
        }
    }
}
```

The PassthroughBroker provides:
- Immediate fills for all orders without delay
- No slippage impact on execution prices
- Zero commission costs
- Simplified execution for strategy testing

### Order Manager

```
OrderManager
  └── on_signal(event)
  └── on_fill(event)
  └── create_order(signal)
  └── update_order(order_id, status)
```

The Order Manager creates and tracks orders:

- **Order Creation**: Converts signals to standardized orders
- **Order Tracking**: Maintains registry of active orders
- **Order Status**: Updates order status based on fills
- **Duplication Prevention**: Ensures no duplicate orders for the same signal
- **Rule Tracking**: Maps orders back to originating rules/signals

### Simulated Broker

```python
class SimulatedBroker(BrokerBase):
    def on_bar(event)
    def check_pending_orders(bar)
    def calculate_execution_price(order, bar)
    def apply_slippage(price, direction, quantity)
    def calculate_commission(price, quantity)
```

The SimulatedBroker implements realistic market simulation:

- **Market Data**: Updates prices from bar events
- **Order Processing**: Checks fill conditions for pending orders
- **Slippage Models**: Simulates price impact with configurable models
- **Commission Models**: Calculates trading costs with various fee structures
- **Fill Generation**: Creates fill events for executed orders

### Backtest Coordinator

```
BacktestCoordinator
  └── initialize(context)
  └── setup()
  └── run()
  └── process_bar()
  └── close_positions()
  └── calculate_statistics()
```

The Backtest Coordinator orchestrates the entire backtesting process:

- **Component Management**: Initializes and configures all components
- **Data Processing**: Drives the system with market data
- **Event Flow**: Ensures proper event sequencing
- **State Management**: Tracks system state throughout the backtest
- **Results Collection**: Gathers and processes backtest results
- **Statistics Calculation**: Computes performance metrics

---

## Event System

The event system is the communication backbone of the entire application, enabling loose coupling between components.

### Event Types

```python
class EventType(Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO = "PORTFOLIO"
    BACKTEST_START = "BACKTEST_START"
    BACKTEST_END = "BACKTEST_END"
```

Standard event types ensure consistent communication patterns.

### Event Structure

```python
class Event:
    def __init__(self, event_type, data=None)
    def get_type()
    def get_data()
```

Events carry type and data information:

- **Event Type**: Categorizes the event for routing
- **Event Data**: Contains event-specific information
- **Metadata**: Timestamps, source component, etc.

### Event Publication

```python
def publish_event(event_bus, event_type, data):
    event = Event(event_type, data)
    return event_bus.publish(event)
```

Standardized publishers ensure consistent event creation:

- **Type-Specific Publishers**: Specialized functions for each event type
- **Data Validation**: Ensures events contain required fields
- **Deduplication Keys**: Adds unique identifiers for deduplication

### Event Subscription

```python
event_bus.subscribe(EventType.BAR, strategy.on_bar)
event_bus.subscribe(EventType.SIGNAL, risk_manager.on_signal)
event_bus.subscribe(EventType.ORDER, broker.on_order)
event_bus.subscribe(EventType.FILL, portfolio.on_fill)
```

Components register for events they need to process.

### Event Flow

The standard event flow in the system:

1. DataHandler emits BAR events with market data
2. Strategy consumes BAR events and emits SIGNAL events
3. RiskManager consumes SIGNAL events and emits ORDER events
4. Broker consumes ORDER events and emits FILL events
5. Portfolio consumes FILL events and emits PORTFOLIO events

### Context Management

```python
with EventContext("train"):
    # Events published here are confined to the training context
    run_backtest(params)

with EventContext("test"):
    # Events published here are confined to the testing context
    run_backtest(params)
```

Events can be confined to specific contexts for proper isolation.

---

## Optimization Framework

The optimization framework enables systematic parameter tuning for trading strategies, with a focus on preventing overfitting through proper train/test validation.

### Parameter Space

```python
class ParameterSpace:
    def add_parameter(name, min_value, max_value, step=1, type="integer")
    def get_combinations()
    def from_dict(config)
```

The ParameterSpace defines the search space for optimization:

- **Parameter Definitions**: Ranges, steps, and types
- **Parameter Generation**: Creates all combinations for grid search
- **Configuration Loading**: Loads parameter space from configuration
- **Validation**: Ensures parameters are within valid ranges

### Optimizer

```python
class Optimizer:
    def optimize()
    def _run_backtest_with_params(params, split)
    def _evaluate_results(train_results, test_results)
```

The Optimizer conducts parameter searches:

- **Parameter Iteration**: Tests different parameter combinations
- **Backtest Execution**: Runs backtests with specific parameters
- **Result Evaluation**: Computes objective function for each result
- **Train/Test Validation**: Ensures generalization by testing on unseen data
- **Result Collection**: Gathers and sorts results by performance

### Optimization Methods

```
OptimizationMethod (Abstract)
  └── GridSearch
  └── RandomSearch
  └── WalkForwardOptimization
```

Various optimization methods can be used:

- **Grid Search**: Exhaustive evaluation of all parameter combinations
- **Random Search**: Random sampling of the parameter space
- **Walk-Forward**: Optimization on rolling time windows

### Objective Functions

```python
class ObjectiveFunction:
    def __call__(backtest_result)
```

Objective functions evaluate backtest performance:

- **Sharpe Ratio**: Default objective function
- **Return**: Absolute or percentage return
- **Custom Metrics**: User-defined performance metrics
- **Multi-Objective**: Weighted combinations of metrics

### Enhanced Optimization Architecture

The enhanced optimization module provides a modular, extensible framework:

```
optimization/
├── interfaces/
│   ├── optimization_method.py
│   ├── optimization_metric.py
│   └── optimization_target.py
├── methods/
│   ├── grid_search.py
│   ├── random_search.py
│   └── walk_forward.py
├── metrics/
│   ├── return_metrics.py
│   ├── risk_metrics.py
│   └── combined_metrics.py
├── targets/
│   ├── strategy_target.py
│   └── portfolio_target.py
└── optimizer.py
```

This architecture enables:

- **Pluggable Components**: Methods, metrics, and targets can be swapped
- **Customization**: Easy extension for new optimization approaches
- **Composability**: Components can be combined in various ways
- **Clean Separation**: Responsibilities are clearly defined
- **Dependency Injection**: Components receive their dependencies explicitly

### Regime-Specific Optimization

The framework supports optimizing for different market regimes:

```python
# Run regime-specific optimization
result = optimization_manager.run_optimization(
    sequence_name="regime_specific",
    targets=["rule_weights"],
    methods={"rule_weights": "genetic"},
    metrics={"rule_weights": "sharpe_ratio"},
    regime_detector_target="regime_detector"
)
```

This enables strategies that adapt to different market conditions:

- **Regime Detection**: Identify different market states
- **Regime-Specific Parameters**: Optimize parameters for each regime
- **Adaptive Strategies**: Switch between parameter sets based on regime
- **Regime Transition Management**: Handle transitions between regimes

---

## Isolation Principles

Proper state isolation is critical for reliable optimization results. The ADMF-Trader system implements several key principles to ensure clean isolation.

### Consistent Component Lifecycle

All components follow the same lifecycle pattern:

```python
def initialize(context):
    """Set up with dependencies."""
    pass
    
def reset():
    """Clear state between runs."""
    pass
    
def teardown():
    """Release resources."""
    pass
```

This ensures consistent state management across components.

### Event Context Isolation

Events are confined to specific contexts:

```python
with EventContext("train"):
    # Train context events
    run_backtest(train_data)

with EventContext("test"):
    # Test context events
    run_backtest(test_data)
```

This prevents event leakage between optimization runs.

### Deep Data Copying

Train/test data sets are completely isolated:

```python
def split_data(self):
    """Create isolated datasets."""
    train_data = self.data.copy(deep=True)
    test_data = self.data.copy(deep=True)
    return train_data, test_data
```

This prevents data leakage between training and testing phases.

### Factory-Based Component Creation

Fresh component instances are created for each optimization run:

```python
def create_component_for_run(component_type, config):
    """Create a fresh component instance."""
    component = component_factory.create(component_type, config)
    component.initialize(context)
    return component
```

This ensures each optimization run starts with clean components.

### Explicit Context Boundaries

Optimization runs have explicit boundaries:

```python
with OptimizationContext() as context:
    # Set up run-specific context
    context.register_components()
    
    # Run optimization within this context
    result = optimizer.optimize(context)
```

This makes run boundaries clear and ensures proper cleanup.

### Clean Dependency Injection

Dependencies are explicitly provided and isolated:

```python
def create_isolated_container():
    """Create a fresh DI container for an optimization run."""
    container = Container()
    
    # Register components with isolated state
    container.register('event_bus', EventBus)
    container.register('data_handler', HistoricalDataHandler)
    container.register('strategy', StrategyFactory.create)
    
    return container
```

This prevents shared state between runs.

By following these principles, the ADMF-Trader system ensures reliable, reproducible results from optimization runs, preventing overfitting and data leakage that could lead to misleading performance expectations.

## Module Interfaces

Each module in the ADMF-Trader system exposes well-defined interfaces to ensure proper integration and encapsulation. The following describes the key interfaces for each module and their interaction patterns.

### Core Module Interfaces

**Component**
```python
class Component:
    def __init__(self, name, parameters=None)
    def initialize(context)
    def initialize_event_subscriptions()
    def start()
    def stop()
    def reset()
    def teardown()
```

**EventBus**
```python
class EventBus:
    def publish(event)
    def subscribe(event_type, handler)
    def unsubscribe(event_type, handler)
    def reset()
```

**Container**
```python
class Container:
    def register(name, component_class, singleton=True)
    def register_instance(name, instance)
    def get(name)
    def has(name)
    def reset()
```

### Data Module Interfaces

**DataHandlerBase**
```python
class DataHandlerBase(Component):
    def load_data(symbols)
    def update_bars()
    def get_latest_bar(symbol)
    def get_latest_bars(symbol, N=1)
    def setup_train_test_split(method, train_ratio, split_date)
    def set_active_split(split_name)
```

### Strategy Module Interfaces

**StrategyBase**
```python
class StrategyBase(Component):
    def initialize(context)
    def initialize_event_subscriptions()
    def on_bar(event)
    def calculate_signals(bar)
    def emit_signal(symbol, direction, price, quantity)
    def reset()
```

### Risk Module Interfaces

**RiskManagerBase**
```python
class RiskManagerBase(Component):
    def initialize(context)
    def initialize_event_subscriptions()
    def on_signal(event)
    def size_position(signal)
    def validate_order(order)
    def emit_order(order_data)
    def reset()
```

**PortfolioBase**
```python
class PortfolioBase(Component):
    def initialize(context)
    def initialize_event_subscriptions()
    def on_fill(event)
    def on_bar(event)
    def update_positions(bar)
    def calculate_equity()
    def get_position(symbol)
    def get_portfolio_value()
    def reset()
```

### Execution Module Interfaces

**BrokerBase**
```python
class BrokerBase(Component):
    def initialize(context)
    def initialize_event_subscriptions()
    def on_order(order_event)
    def execute_order(order_data, price=None)
    def cancel_order(order_id)
    def get_order_status(order_id)
```

### Module Interaction Patterns

The following interactions occur between modules:

1. **Data → Strategy**: The DataHandler emits BAR events which the Strategy consumes to generate signals.
   ```python
   # In DataHandler
   self.event_bus.publish(Event(EventType.BAR, bar_data))
   
   # In Strategy
   def on_bar(self, event):
       bar_data = event.get_data()
       # Process market data and generate signals
   ```

2. **Strategy → Risk**: The Strategy emits SIGNAL events which the RiskManager consumes to create orders.
   ```python
   # In Strategy
   self.event_bus.publish(Event(EventType.SIGNAL, signal_data))
   
   # In RiskManager
   def on_signal(self, event):
       signal_data = event.get_data()
       # Apply position sizing and risk limits
   ```

3. **Risk → Execution**: The RiskManager emits ORDER events which the Broker consumes to execute trades.
   ```python
   # In RiskManager
   self.event_bus.publish(Event(EventType.ORDER, order_data))
   
   # In Broker
   def on_order(self, event):
       order_data = event.get_data()
       # Execute order and generate fills
   ```

4. **Execution → Risk**: The Broker emits FILL events which the Portfolio consumes to update positions.
   ```python
   # In Broker
   self.event_bus.publish(Event(EventType.FILL, fill_data))
   
   # In Portfolio
   def on_fill(self, event):
       fill_data = event.get_data()
       # Update positions and equity
   ```