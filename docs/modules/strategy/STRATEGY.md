# ADMF-Trader Strategy Module (STRATEGY.MD)

## I. Overview & Core Philosophy

The Strategy module is central to the ADMF-Trader system, responsible for analyzing market data and generating trading signals. It's designed to be modular, extensible, and maintainable, forming the primary decision-making core. Strategies analyze market conditions and produce actionable signals.

The module supports a component-based architecture where strategies can be built from smaller, reusable parts.

## II. Strategy Component Architecture

### A. Core Design Principle: Built-in Optimizability

**All strategy components in ADMF-Trader are designed to be optimizable by default.** This is a fundamental architectural decision that ensures seamless integration with the optimization framework. Rather than using separate mixins or interfaces, the optimization methods are built directly into the base classes of all strategy components.

**Important: While all components CAN be optimized, not all components HAVE parameters.** The base classes provide sensible default implementations of the optimization methods that assume no parameters. Components with parameters override these defaults.

### B. Base Strategy (`StrategyBase` or `Strategy`)

* **Purpose**: Provides a foundational class for all trading strategies, defining a common interface and functionality for event handling, parameter management, signal generation, and optimization.
* **Inheritance**: Extends `ComponentBase` (from `core`).
* **Key Responsibilities & Methods**:
    * Event handling: Subscribes to `BAR` events via an `on_bar` method.
    * Signal calculation and emission: Implements `calculate_signals` to determine trading signals based on processed bar data and `emit_signal` to publish these signals.
    * Lifecycle methods: Manages its lifecycle through `initialize`, `reset`, and `teardown` methods.
    * Indicator management: Handles the update of internal indicators via `_update_indicators`.
    * Component Composition: Supports building complex strategies by adding sub-components (Indicators, Features, Rules, or other Strategies) using an `add_component` method.
* **Built-in Optimization Interface**:
    * `get_parameter_space()`: Returns a dictionary defining optimizable parameters and their possible values. Default: returns {} (no parameters)
    * `set_parameters(params)`: Applies a set of parameters to the strategy. Default: no-op
    * `get_parameters()`: Returns current parameter values as a dictionary. Default: returns {}
    * `validate_parameters(params)`: Validates that a parameter set is valid, returns (bool, error_message). Default: returns (True, "")
    * Components with parameters should override these methods as needed
    * This design ensures every strategy can work with the optimization framework, even if it has no parameters to optimize

### C. Strategy Building Blocks

Strategies are constructed using several types of composable components, each with built-in optimization support:

1.  **Indicators (`IndicatorBase`)**:
    * **Purpose**: Perform technical calculations on market data (e.g., Moving Averages, RSI, ATR).
    * **Core Interface**: 
        * `calculate(value, timestamp)`: Initial calculation
        * `update(value, timestamp)`: Incremental updates
        * Properties: `value` (current reading), `ready` (has enough data)
    * **Built-in Optimization Interface** (with default implementations):
        * `get_parameter_space()`: Returns optimizable parameters (e.g., {'period': [10, 20, 30]}). Default: {}
        * `set_parameters(params)`: Updates indicator parameters. Default: no-op
        * `get_parameters()`: Returns current parameters. Default: {}
        * `validate_parameters(params)`: Ensures valid parameter combinations. Default: (True, "")

2.  **Features (`FeatureBase`)**:
    * **Purpose**: Represent higher-level market characteristics derived from one or more indicators or raw market data (e.g., Trend Strength, Momentum, identified Regime).
    * **Interface**: Similar to Indicators, with both core calculation methods and built-in optimization methods.

3.  **Rules (`RuleBase`)**:
    * **Purpose**: Encapsulate specific trading decision logic based on indicator and feature values.
    * **Core Interface**:
        * `evaluate(data)`: Returns (is_triggered, signal_strength) tuple
        * `weight`: Property for use in ensemble strategies
    * **Built-in Optimization Interface** (with default implementations):
        * `get_parameter_space()`: Defines rule parameters (thresholds, windows, etc.). Default: {}
        * `set_parameters(params)`: Updates rule parameters. Default: no-op
        * `get_parameters()`: Returns current parameters. Default: {}
        * `validate_parameters(params)`: Validates rule-specific constraints. Default: (True, "")

### D. Example Implementation

```python
# Example 1: Rule with parameters (overrides optimization methods)
class MACrossoverRule(RuleBase):
    """Example rule with parameters showing optimization support"""
    
    def __init__(self, fast_window=10, slow_window=30):
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window
        
    def get_parameter_space(self):
        """Define what can be optimized"""
        return {
            'fast_window': [5, 10, 15, 20],
            'slow_window': [20, 30, 40, 50]
        }
    
    def validate_parameters(self, params):
        """Ensure fast < slow"""
        fast = params.get('fast_window', self.fast_window)
        slow = params.get('slow_window', self.slow_window)
        if fast >= slow:
            return False, "Fast window must be less than slow window"
        return True, ""
    
    def evaluate(self, data):
        """Trading logic implementation"""
        # ... implementation
        pass

# Example 2: Rule without parameters (uses default optimization methods)
class BuyOnMondayRule(RuleBase):
    """Example rule with no parameters - uses default optimization methods"""
    
    def evaluate(self, data):
        """Buy signal on Mondays only"""
        is_monday = data['timestamp'].weekday() == 0
        return is_monday, 1.0 if is_monday else 0.0
    
    # No need to override optimization methods - defaults handle it:
    # get_parameter_space() returns {}
    # set_parameters() does nothing
    # get_parameters() returns {}
    # validate_parameters() returns (True, "")
```

This design ensures that:
1. **Every component is optimizable** - Even parameter-less components work with the optimization framework
2. **Consistent interface** - All components use the same optimization methods
3. **Minimal boilerplate** - Components without parameters don't need to implement empty methods
4. **Type safety** - The base classes provide the optimization interface
5. **Seamless integration** - The optimization framework handles components with or without parameters gracefully

### E. Composite and Ensemble Strategies

* Strategies can be formed by combining multiple simpler strategy components or sub-strategies.
* These composite structures, like an `EnsembleStrategy`, can use various methods for signal aggregation, such as majority voting or weighted voting, based on the weights assigned to individual rules or sub-strategies.
* **Aggregation Methods**:
    * **Weighted Average**: Combines signal strengths using component weights, producing a continuous signal strength
    * **Majority Vote**: Binary decision based on the majority direction of component signals
    * **Threshold-based**: Requires minimum agreement ratio before generating signals

### F. Parameter Management

* A crucial aspect of strategy design is managing parameters. This includes standardized access, validation, handling of default values, and defining a parameter space for optimization.
* The `ParameterSet` class is envisioned for this, providing a structured way to manage parameter schemas, values, validation, and type conversions.

### G. Strategy Factory

* A `StrategyFactory` is intended to manage the discovery, instantiation, and configuration of strategy instances, simplifying their creation and setup.

### H. Signal Processing Pipeline

The complete signal flow through the system:
```
Market Data (BAR events)
    ↓
Strategy Components (Indicators → Features → Rules)
    ↓
Raw Signals (with strength and metadata)
    ↓
MetaLabelers (quality scoring)
    ↓
Risk Manager (filtering/sizing)
    ↓
Final Orders
```

## III. MetaComponent Framework (Formerly AnalyticsComponent)

This framework introduces components that analyze market conditions and trading signals without directly making trading decisions. They provide essential context to both Strategy and Risk modules.

### A. `MetaComponent` Base Class

* **Purpose**: A base class for components that analyze data without making trading decisions. MetaComponents maintain state, provide context, and offer insights.
* **Inheritance**: Extends `BaseComponent`.
* **Key Methods**: Includes an abstract `analyze` method, along with `get_current_analysis` and `get_history`.

### B. `Classifier`

* **Purpose**: A specialized `MetaComponent` that analyzes market data to produce categorical labels (e.g., market regimes, volatility states). It does not generate trading signals directly.
* **Functionality**:
    * Implements an abstract `classify(data)` method for specific classification logic.
    * Provides `get_current_classification()` and `get_classification_history()`.
    * Typically subscribes to `BAR` events to receive market data.
    * Publishes `CLASSIFICATION` events, especially when the identified category changes.
* **`RegimeDetector`**:
    * A concrete implementation of a `Classifier` designed to identify market regimes.
    * Utilizes configured indicators (like ATR for volatility, MAs for trend) and thresholds to make classifications.
    * Incorporates stabilization logic (e.g., `min_regime_duration`) to prevent overly frequent changes in the detected regime.
    * Its classifications can be used by adaptive strategies to switch parameters or by the Risk module to filter signals.

### C. `MetaLabeler` (Conceptual)

* **Purpose**: A type of `MetaComponent` that provides a secondary layer of analysis, specifically focused on evaluating the quality of trading signals generated by strategies.
* **Inputs**: Consumes `SIGNAL`, `FILL`, and `TRADE_COMPLETE` events.
* **Outputs**: Produces assessments of signal quality, such as confidence scores or win probabilities, and publishes these as `META_LABEL` events.

## IV. Regime-Adaptive Strategies

* **Concept**: Strategies that can alter their parameters or logic based on the currently detected market regime.
* **Interaction**: A `RegimeAdaptiveStrategy` subscribes to `CLASSIFICATION` events from a `RegimeDetector`. When a regime change is detected, it loads and applies a pre-optimized set of parameters suited for that new regime.
* **Parameter Source**: Optimized parameters for each regime are typically stored in a configuration file (e.g., JSON) generated by an `EnhancedOptimizer`.
* **Parameter Switching Modes**:
    * **Immediate**: Switch parameters as soon as regime changes
    * **Delayed**: Wait for current positions to close before switching
    * **Gradual**: Blend parameters over multiple bars
    * **Trade-locked**: Maintain entry parameters for trade lifetime

## V. Handling of Boundary Trades and Strategy-Regime Conflicts

* **Boundary Trades**: Trades initiated in one regime and concluding in another present challenges due to parameter mismatches. These are tracked, and their performance is analyzed separately.
* **Strategy-Regime Conflict**: The initial approach is often "Strategy Overrides Regime," where regimes primarily influence parameter selection for optimization, but the strategy's core logic generates signals. Filtering or overriding these signals based on regime compatibility can be a subsequent step, potentially handled by the Risk module.
* **Boundary Defense Strategies (`src/strategy/BOUNDARY_DEFENSE.MD`)**: This document explores critical issues arising from parameter switching during regime transitions and suggests solutions such as:
    * Closing all positions upon regime change.
    * Incorporating boundary trade performance into the optimization fitness function.
    * Implementing trade-locked parameters, where a trade uses the parameters active at its entry for its entire lifecycle.
    * Gradual parameter transitions over several bars.
    * Optimizing for cross-regime robustness to ensure parameters don't catastrophically fail in adjacent or misclassified regimes.

## VI. Best Practices for Strategy Implementation

* **State Management**: Initialize all state within the constructor and ensure a thorough `reset` method to clear state for backtesting or re-optimization.
* **Parameter Definition**: Use clear schemas for parameters, including types, defaults, and validation rules (min/max, allowed values).
* **Signal Structure**: Emit signals in a standardized dictionary format, including symbol, direction, quantity, price, timestamp, and reason.
* **Indicator Efficiency**: Utilize rolling windows and consider vectorized calculations (e.g., with numpy) for performance. Leveraging established libraries like TA-Lib for complex indicators can also be beneficial.
* **Modularity**: Break down complex logic into individual, reusable rules and features.

## VII. Event Flow and Lifecycle

### A. Standard Event Subscriptions
* Strategies typically subscribe to: `BAR`, `FILL`, `PORTFOLIO_UPDATE`
* MetaComponents subscribe to: `BAR` (Classifiers), `SIGNAL` (MetaLabelers)
* Strategies emit: `SIGNAL` events
* Classifiers emit: `CLASSIFICATION` events

### B. Component Lifecycle
1. **Construction**: Initialize state containers and parameters
2. **Initialization**: Subscribe to events, load initial parameters
3. **Operation**: Process events, update state, emit signals
4. **Reset**: Clear all state for new backtest/session
5. **Teardown**: Unsubscribe from events, cleanup resources