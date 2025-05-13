# ADMF-Trader Documentation

This document provides an overview of the ADMF-Trader system, an algorithmic trading framework with modular design and event-driven architecture.

## System Architecture

ADMF-Trader is designed with a modular, event-driven architecture for systematic strategy development, testing, optimization, and deployment. The system is organized into five primary modules:

1. **Core Module**: Foundation services including event bus, component lifecycle, configuration, dependency injection, and the Analytics submodule for performance measurement and reporting
2. **Data Module**: Market data processing, train/test splitting, and data source management
3. **Strategy Module**: Signal generation, optimization frameworks, and trading logic
4. **Risk Module**: Position sizing, risk limits, and portfolio management
5. **Execution Module**: Order management, broker simulation, backtest coordination and trade execution

## System Flow

The typical flow of data through the system:

1. **Data Flow**: `DataHandler` loads market data and emits `BAR` events
2. **Signal Generation**: `Strategy` consumes `BAR` events and emits `SIGNAL` events 
3. **Risk Management**: `RiskManager` consumes `SIGNAL` events and emits `ORDER` events
4. **Order Execution**: `Broker` consumes `ORDER` events and emits `FILL` events
5. **Portfolio Tracking**: `Portfolio` consumes `FILL` events and updates positions/equity

## Key Architectural Principles

1. **Event-Driven Communication**: Components interact through an event system, enabling loose coupling and extensibility
2. **Component-Based Design**: All system elements follow a consistent lifecycle pattern with explicit state transitions
3. **Dependency Injection**: Components receive dependencies via DI container, promoting testability and modularity
4. **State Isolation**: Careful management of state for reliable optimization and backtesting
5. **Single Source of Truth**: Portfolio in Risk module is the authority for positions
6. **Thread Safety**: Thread-safe collections and proper locking for all shared state
7. **Memory Management**: Bounded collections with pruning and efficient data views

## Thread Safety Strategy

Thread safety is critical in the ADMF-Trader system as many components can operate concurrently. The following principles should be followed:

### When to Use Locks

- Use locks when modifying shared state that can be accessed by multiple threads
- Lock at the method level rather than for individual operations
- Keep critical sections as small as possible for performance

### Thread-Safe Collections

The system provides several thread-safe collection classes:

```python
# Thread-safe dictionary example
from core.utils.collections import ThreadSafeDict

class Component:
    def __init__(self):
        self.shared_state = ThreadSafeDict()
        
    def update_state(self, key, value):
        # Thread-safe operation - no additional locks needed
        self.shared_state[key] = value
```

### Avoiding Race Conditions

- Use atomic operations when possible
- Acquire locks in a consistent order to prevent deadlocks
- Prefer thread-safe collections over manual locking when possible
- Document thread safety assumptions in component interfaces

### Exception Handling in Multi-threaded Context

- Catch exceptions within thread operations to prevent silent failures
- Log all exceptions in worker threads
- Use exception handling to maintain system stability
- Implement appropriate recovery mechanisms

## Error Handling Strategy

ADMF-Trader implements a consistent approach to error handling for robustness and maintainability.

### Exception Handling Guidelines

1. **When to Catch vs. Propagate**:
   - Catch exceptions only when you can properly handle them
   - Propagate exceptions when they represent system-level issues
   - Wrap lower-level exceptions in domain-specific ones

2. **Logging Standards**:
   - Use ERROR level for conditions that require immediate attention
   - Use WARNING level for unexpected situations that don't affect core functionality
   - Use INFO level for significant state transitions
   - Include contextual information in log messages

3. **Recovery Strategies**:
   - Implement graceful degradation where possible
   - Use retry mechanisms with exponential backoff for transient failures
   - Provide clear failure messages with recovery suggestions
   - Enable restart from the last known good state

4. **Exception Hierarchy**:
   ```
   ADMFException (Base)
     └── ConfigurationException
     └── DataException
     └── EventException
     └── ValidationException
     └── ExecutionException
     └── StrategyException
     └── RiskException
   ```

## Project Goals

Our primary goal is to implement enough of the ADMF system to run a rudimentary backtest with the following features:

1. **Train/Test Splitting**: Proper data isolation between training and testing periods
2. **Grid Search Optimization**: Parameter optimization for a simple moving average crossover strategy
3. **Default Components**: Standard risk and position management components
4. **Data Source**: Using the `data/SPY_1min.csv` data file as our primary test dataset

## Implementation Guides

Detailed implementation guides for each module can be found in their respective directories:

- [Core Module Implementation](/docs/core/IMPLEMENTATION.md)
- [Data Module Implementation](/docs/data/IMPLEMENTATION.md)
- [Strategy Module Implementation](/docs/strategy/IMPLEMENTATION.md)
- [Risk Module Implementation](/docs/risk/IMPLEMENTATION.md)
- [Execution Module Implementation](/docs/execution/IMPLEMENTATION.md)

## Next Phase: Ensemble Strategy

After achieving the primary goals, our next step will be to implement an ensemble strategy system with the following capabilities:

1. **Multiple Rule-Based Strategies**: Framework for combining multiple rule-based strategies
2. **Regime Identification**: Regime filters to identify different market conditions
3. **Regime-Specific Optimization**: Optimize strategy components under distinct market regimes
4. **Genetic Algorithm Weighting**: Determine optimal weights for strategy components
5. **Composite Strategy Framework**: Dynamically adjust weights based on detected market regimes