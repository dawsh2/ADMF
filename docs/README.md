# ADMF-Trader Framework

An Algorithmic Trading Framework with modular, event-driven architecture for systematic strategy development, testing, optimization, and deployment. This framework is designed to support regime-specific optimization for adaptive trading strategies.

## System Architecture

ADMF-Trader is organized into five primary modules:

1. **Core Module**: Foundation services including event bus, component lifecycle, configuration, dependency injection, and analytics
2. **Data Module**: Market data processing, train/test splitting, and data source management
3. **Strategy Module**: Signal generation, optimization frameworks, and trading logic
4. **Risk Module**: Position sizing, risk limits, and portfolio management
5. **Execution Module**: Order management, broker simulation, backtest coordination, and trade execution

## Project Structure

```
/ADMF/
├── config/                   # Configuration files
│   ├── config.yaml           # Main configuration
│   └── phase3_config_examples.yaml
│
├── data/                     # Market data files
│   └── SPY_1min.csv          # Sample data for testing
│
├── docs/                     # Documentation
│   ├── ARCH.md               # System architecture
│   ├── GOALS.md              # Project goals
│   ├── IMPROVEMENTS.md       # Suggested improvements
│   │
│   ├── core/                 # Core module documentation
│   │   ├── IMPLEMENTATION.MD
│   │   ├── analytics/        # Analytics system docs
│   │   ├── architecture/     # Architecture docs
│   │   ├── communication/    # Event system docs
│   │   ├── concurrency/      # Thread safety docs
│   │   ├── foundation/       # Component lifecycle docs
│   │   └── infrastructure/   # Error handling, logging docs
│   │
│   ├── data/                 # Data module documentation
│   ├── execution/            # Execution module documentation
│   ├── risk/                 # Risk module documentation
│   └── strategy/             # Strategy module documentation
│       └── optimization/     # Optimization framework docs
│
├── src/                      # Source code
│   ├── core/                 # Core module implementation
│   │   ├── __init__.py
│   │   ├── component.py      # Base component class
│   │   ├── config.py         # Configuration system
│   │   ├── container.py      # Dependency injection container
│   │   ├── event.py          # Event definitions
│   │   ├── event_bus.py      # Event bus implementation
│   │   ├── exceptions.py     # Exception hierarchy
│   │   └── logging_setup.py  # Logging configuration
│   │
│   ├── data/                 # Data module implementation
│   │   └── csv_data_handler.py # CSV data source handler
│   │
│   ├── execution/            # Execution module implementation
│   │   └── simulated_execution_handler.py # Simulated broker
│   │
│   ├── risk/                 # Risk module implementation
│   │   ├── basic_portfolio.py # Portfolio management
│   │   └── basic_risk_manager.py # Risk management
│   │
│   └── strategy/             # Strategy module implementation
│       ├── classifier.py     # Strategy classification
│       ├── ma_strategy.py    # Moving average strategy
│       ├── regime_detector.py # Market regime detection
│       ├── regime_adaptive_strategy.py # Regime-adaptive strategy
│       ├── components/       # Strategy components
│       │   ├── indicators/   # Technical indicators
│       │   └── rules/        # Trading rules
│       ├── implementations/  # Strategy implementations
│       └── optimization/     # Optimization framework
│           ├── basic_optimizer.py # Basic grid search
│           └── enhanced_optimizer.py # Regime-specific optimizer
│
├── main.py                   # Entry point
└── README.md                 # This file
```

## Module Overview

### 1. Core Module

The Core module provides fundamental services used throughout the system:

- **Component System**: Base component class with lifecycle methods (initialize, setup, run, teardown)
- **Event System**: Thread-safe event bus with publish/subscribe capabilities
- **Configuration**: YAML-based configuration loading with parameter validation
- **Dependency Injection**: Container for component registration and resolution
- **Logging**: Configurable logging throughout the system
- **Exception Handling**: Hierarchical exception classes for better error management

### 2. Data Module

The Data module handles market data management and preprocessing:

- **Data Handlers**: CSV data loading with bar aggregation capabilities
- **Data Splitting**: Train/test data splitting for strategy evaluation
- **Market Data**: Processing of OHLCV (Open, High, Low, Close, Volume) data
- **Event Generation**: Emits BAR events to the event bus

### 3. Strategy Module

The Strategy module contains trading logic and optimization:

- **Base Strategies**: Simple implementations like Moving Average Crossover
- **Regime Detection**: Market regime identification (trending, volatile, ranging)
- **Regime-Adaptive Strategies**: Parameter switching based on detected regimes
- **Technical Indicators**: Trend, oscillator, and volatility indicators
- **Trading Rules**: Rule-based signal generation
- **Optimization**: Parameter optimization frameworks

### 4. Risk Module

The Risk module manages portfolio and risk:

- **Portfolio Management**: Position and equity tracking
- **Risk Management**: Position sizing and risk limits
- **Regime-Specific Performance**: Performance metrics by market regime
- **Boundary Trade Handling**: Management of trades that span multiple regimes

### 5. Execution Module

The Execution module handles order execution:

- **Simulated Broker**: Backtest-oriented order execution
- **Order Management**: Processing of orders and generation of fills
- **Slippage Models**: Realistic order execution simulation
- **Commission Models**: Transaction cost modeling

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

## Implementation Philosophy

The development follows an **incremental, working-first approach** based on these key principles:

1. **Start with minimal working code** that demonstrates complete functionality
2. **Maintain event-driven architecture** from the beginning to prevent lookahead bias
3. **Refactor incrementally** from simple to complex architecture
4. **Test each step** to maintain a working system throughout development
5. **Use passthrough modes** to simplify complex components when needed

This philosophy guides our phased implementation approach, allowing us to:
- Have a working system at every stage of development
- Add architectural complexity only as needed
- Prevent lookahead bias by maintaining proper event flow
- Implement advanced features in a structured way

### Using Detailed Design Documents

The extensive design documents serve as:
1. **Long-Term Architectural Vision**: Defining the target state of the framework
2. **Library of Pre-Designed Features**: Reference for implementing new features
3. **Guide for Refactoring**: Direction for incremental improvements
4. **Understanding Component Dependencies**: Clarity on inter-component interactions

During early phases, these documents help establish proper interfaces and architectural patterns, while in later phases they guide the implementation of more advanced features.

## Implementation Status

The framework is implemented in phases:

### Phase 1: Minimal Working System (Completed)
- Created single-file, event-driven backtest system that works end-to-end
- Implemented event-driven architecture (BAR → SIGNAL → ORDER → FILL)
- Added moving average crossover strategy calculation
- Implemented portfolio tracking with position updates
- Produced equity curve output and basic performance metrics

### Phase 2: Core Module Extraction (Completed)
- Extracted event system to `src/core/event_bus.py` and `event.py`
- Extracted component base class to `src/core/component.py` with lifecycle methods
- Extracted configuration system to `src/core/config.py`
- Extracted dependency injection container to `src/core/container.py`
- Integrated extracted components into working system
- Added proper error handling and logging

### Phase 3: Module Implementation and Regime-Specific Optimization (Current)
- Implemented Data Module with CSV data loading and train/test splitting
- Implemented Strategy Module with simple strategies, indicators, and regime detection
- Implemented Risk Module with basic position sizing and regime-specific performance tracking
- Implemented Execution Module with simulated broker for backtesting
- Added Enhanced Optimization with regime-specific parameter optimization
- Fixed portfolio reset issue for accurate optimization comparison
- Improved regime detection sensitivity

### Phase 4: Advanced Features (Planned)
- Advanced Strategies (composite and ensemble strategies)
- Optimization Framework enhancements (walk-forward testing, cross-validation)
- Live Trading capabilities with broker integration
- Performance Tuning for large datasets
- Machine learning integration for regime detection

## Regime-Specific Optimization

The system supports optimizing strategy parameters for different market regimes:

1. **Regime Detection**: The RegimeDetector component identifies market conditions (trending, volatile, ranging, etc.)
2. **Regime-Specific Parameters**: The EnhancedOptimizer optimizes parameters for each detected regime
3. **Parameter Switching**: The RegimeAdaptiveStrategy dynamically switches parameters based on detected regime
4. **Boundary Trade Handling**: The Portfolio component properly tracks trades that span multiple regimes

### RegimeDetector Implementation

The regime detector uses a combination of technical indicators to classify market conditions:

- **Trend Indicators**: Moving average directional analysis
- **Volatility Indicators**: ATR (Average True Range) for volatility measurement
- **Momentum Indicators**: RSI for overbought/oversold detection

The current regime classification includes:
- `trending_up_volatile`: Strong uptrend with high volatility
- `trending_up_low_vol`: Steady uptrend with low volatility
- `ranging_low_vol`: Sideways market with low volatility
- `trending_down`: Downtrend (with various volatility profiles)
- `default`: When no specific regime is detected

### Boundary Trade Handling

Special care is taken for trades that open in one regime and close in another:

1. Each trade is tagged with the regime in which it was entered
2. When a trade exits in a different regime, it's tagged as a "boundary trade"
3. Boundary trades are tracked separately for performance analysis
4. This allows for proper attribution of performance across regime transitions

## Current Phase 3 Status and Recommendations

### Portfolio Reset Fix

A critical issue was identified with the optimization process where the portfolio state was not being reset between optimization runs. This caused:

1. **State Persistence**: Each parameter combination was starting with the portfolio state from the previous run, creating unfair comparisons
2. **Incorrect Initial Values**: In our test case, the portfolio was starting at 109,961.25 instead of 100,000.00
3. **No Trades in Test Phase**: The test phase wasn't executing any trades due to incorrect initial state

The comprehensive fix implemented:

1. **Added reset() method to BasicPortfolio**:
   ```python
   def reset(self):
       """Reset the portfolio to its initial state for a fresh backtest run."""
       self.logger.info(f"Resetting portfolio '{self.name}' to initial state")
       
       # Close any open positions
       if self.open_positions:
           now = datetime.datetime.now(datetime.timezone.utc)
           self.close_all_positions(now)
           
       # Reset cash and positions
       self.current_cash = self.initial_cash
       self.open_positions = {}
       self._trade_log = []
       
       # Reset performance metrics
       self.realized_pnl = 0.0
       self.unrealized_pnl = 0.0
       self.current_holdings_value = 0.0
       self.current_total_value = self.initial_cash
       
       # Reset market data
       self._last_bar_prices = {}
       
       # Reset history
       self._portfolio_value_history = []
       
       # Reset market regime to default
       self._current_market_regime = "default"
       
       self.logger.info(f"Portfolio '{self.name}' reset successfully. Cash: {self.current_cash:.2f}, Total Value: {self.current_total_value:.2f}")
   ```

2. **Modified optimizers to reset portfolio before each run**:
   ```python
   # Reset portfolio state to ensure a clean start
   try:
       self.logger.info(f"Optimizer: Resetting portfolio state before {dataset_type} run with params: {params_to_test}")
       if hasattr(portfolio_manager, 'reset') and callable(portfolio_manager.reset):
           portfolio_manager.reset()
       else:
           self.logger.warning("Portfolio does not have a reset method. State may persist between runs.")
   except Exception as e:
       self.logger.error(f"Error resetting portfolio before backtest run: {e}", exc_info=True)
   ```

3. **Created test verification script** (`test_portfolio_reset.py`)**:
   - Runs two identical backtests with a reset in between
   - Verifies that both runs produce the same final portfolio value
   - Confirms proper reset of cash to initial value (100,000.00)
   - Ensures all positions and trade history are cleared

The files modified during this fix implementation:
1. `/Users/daws/ADMF/src/risk/basic_portfolio.py` - Added reset() method
2. `/Users/daws/ADMF/src/strategy/optimization/basic_optimizer.py` - Added portfolio reset
3. `/Users/daws/ADMF/src/strategy/optimization/enhanced_optimizer.py` - Added portfolio reset 
4. `/Users/daws/ADMF/test_portfolio_reset.py` - Created verification test

### Regime Detection Settings

Current test results with 1000 bars show:
- Only the "default" regime is being detected
- No regime transitions are being identified
- The Sharpe ratio for the default regime is negative (-0.15)
- We need modifications to make the regime detector more sensitive

Based on analysis of our indicator distributions, we've made these adjustments:

1. **Reduced minimum regime duration** from 3 to 2 bars to detect shorter-lived regimes

2. **Made indicators more responsive with shorter periods**:
   - ATR period reduced from 20 to 10
   - Moving average periods changed from 10/30 to 5/20

3. **Adjusted threshold values based on observed data distributions**:
   ```yaml
   regime_thresholds:
     trending_up_volatile:
       trend_10_30: {"min": 0.02}   # Reduced from 0.3 to 0.02
       atr_20: {"min": 0.15}      # Increased from 0.01 to 0.15
     
     trending_up_low_vol:
       trend_10_30: {"min": 0.02}   # Reduced from 0.3 to 0.02
       atr_20: {"max": 0.15}      # Increased from 0.01 to 0.15
       
     ranging_low_vol:
       trend_10_30: {"min": -0.01, "max": 0.01} # Tightened from -0.15/0.15 to -0.01/0.01
       atr_20: {"max": 0.12}     # Increased from 0.008 to 0.12
       
     trending_down:
       trend_10_30: {"max": -0.01}  # Changed from -0.3 to -0.01
   ```

4. **Added "trending_down" regime** classification to better capture downtrends

Our indicator analysis revealed:
- Current ATR values consistently above 0.1, suggesting previous thresholds (0.01, 0.008) were too low
- Trend_10_30 values mostly between -0.01 and 0.03, suggesting previous thresholds (0.3, -0.3) were too extreme

If these adjustments don't yield sufficient regime detection, we'll consider:
1. Adding more sophisticated change-point detection algorithms
2. Implementing statistical methods for regime identification
3. Using different indicators more suitable for our dataset

### Next Steps

1. Run optimization with more data (5000+ bars) to increase regime variety:
   ```bash
   python main.py --config config/config.yaml --bars 5000 --optimize
   ```

2. Verify regime transitions are being detected:
   ```bash
   python test_regime_detection.py --bars 5000
   ```

3. Evaluate boundary trade handling between regimes
4. Test the complete Phase 3 implementation with real-world data
5. Consider alternative indicators or statistical methods if regime detection remains challenging

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Required packages (install via `pip`):
  - numpy
  - pandas
  - matplotlib
  - pyyaml
  - ta-lib (optional, for advanced indicators)

### Basic Usage

To run a basic backtest:

```bash
python main.py --config config/config.yaml
```

For optimization:

```bash
python main.py --config config/config.yaml --optimize
```

To limit the number of bars processed:

```bash
python main.py --config config/config.yaml --bars 1000
```

To test regime detection specifically:

```bash
python test_regime_detection.py --bars 1000
```

To verify the portfolio reset functionality:

```bash
python test_portfolio_reset.py
```

### Configuration

The system is configured through YAML files in the `config/` directory. The main configuration file (`config.yaml`) contains sections for:

- **Data**: Data source settings and processing options
- **Strategy**: Strategy type and parameters
- **Risk**: Risk management settings and position sizing
- **Execution**: Execution settings, slippage, and commission models
- **Regime Detection**: Settings for market regime classification

Example configuration:

```yaml
data:
  source: "csv"
  file_path: "data/SPY_1min.csv"
  train_test_split: 0.7

strategy:
  type: "regime_adaptive"
  base_strategy: "ma_crossover"
  parameters:
    fast_period: 10
    slow_period: 30
    
risk:
  position_size: "fixed"
  size: 100
  
regime_detector:
  min_regime_duration: 2
  indicators:
    - name: "trend_10_30"
      type: "trend"
      parameters:
        fast_period: 10
        slow_period: 30
    - name: "atr_20"
      type: "volatility"
      parameters:
        period: 20
```

## Documentation and Resources

### Module Documentation

Detailed implementation guides for each module can be found in their respective directories:

- [Core Module](/docs/core/IMPLEMENTATION.MD) - Base component system, event bus, configuration
- [Data Module](/docs/data/DATA.md) - Data handling, sources, and train/test splitting
- [Strategy Module](/docs/strategy/STRATEGY_IMPLEMENTATION.MD) - Signal generation and optimization
- [Risk Module](/docs/risk/RISK_IMPLEMENTATION.MD) - Portfolio and risk management
- [Execution Module](/docs/execution/EXECUTION.md) - Order execution and brokerage

Note: 
- The Data Module documentation has been consolidated into a single comprehensive file.
- The Execution Module documentation has been consolidated into EXECUTION.md as the primary reference, but the original files are retained as they contain some additional implementation details and specialized guidance.

**Note**: This consolidated README provides a high-level overview, while the module-specific documentation contains detailed implementation patterns, code examples, and best practices for each component. Refer to the original documentation for implementation specifics.

### Architecture Documentation

- [Detailed Architecture](/docs/ARCH.md) - Comprehensive system architecture design
- [Project Goals](/docs/GOALS.md) - Project objectives and success criteria
- [Event Architecture](/docs/core/communication/EVENT_ARCHITECTURE.md) - Event system design
- [Component Architecture](/docs/core/architecture/COMPONENT_ARCHITECTURE.md) - Component design patterns
- [Thread Safety](/docs/core/concurrency/THREAD_SAFETY.md) - Concurrency management
- [Error Handling](/docs/core/infrastructure/ERROR_HANDLING.md) - Exception hierarchy and handling
- [Logging & Monitoring](/docs/core/infrastructure/LOGGING_MONITORING.md) - Diagnostic infrastructure

### Advanced Implementation Documentation

- [Strategy Implementation](/docs/strategy/STRATEGY_IMPLEMENTATION.MD) - Strategy design
- [Regime Optimization](/docs/strategy/optimization/REGIME_OPTIMIZATION_IMPLEMENTATION.md) - Regime-specific optimization
- [Analytics Framework](/docs/strategy/ANALYTICS_COMPONENT_FRAMEWORK.md) - Performance metrics
- [Dependency Management](/docs/core/foundation/DEPENDENCY_MANAGEMENT.md) - DI container implementation
- [Resource Optimization](/docs/core/performance/RESOURCE_OPTIMIZATION.md) - Performance tuning
- [Strategic Caching](/docs/core/performance/STRATEGIC_CACHING.md) - Cache implementation patterns

### Thread Safety Implementation

Thread safety is implemented throughout the system using:
- Thread-safe collections (ConcurrentDictionary, BlockingCollection)
- Explicit locking with ReaderWriterLockSlim for high-contention resources
- Immutable objects for shared state
- Message passing through the event system for component communication
- Thread-local storage for context-specific data

See [Thread Safety](/docs/core/concurrency/THREAD_SAFETY.md) for detailed implementation patterns.

### Error Handling Framework

The system implements a comprehensive error handling approach:

#### Exception Hierarchy
```
ADMFException (Base)
├── ConfigurationException
│   ├── InvalidConfigurationException
│   └── MissingConfigurationException
├── ComponentException
│   ├── ComponentInitializationException
│   ├── ComponentLifecycleException
│   └── DependencyResolutionException
├── DataException
│   ├── DataLoadException
│   ├── DataValidationException
│   └── TrainTestSplitException
├── StrategyException
│   ├── SignalGenerationException
│   ├── OptimizationException
│   └── RegimeDetectionException
└── ExecutionException
    ├── OrderValidationException
    ├── FillProcessingException
    └── BrokerException
```

#### Error Recovery
- Component-level reset mechanisms for recovering from failed states
- Automatic retry with backoff for transient failures
- Graceful degradation with fallback strategies
- Comprehensive logging with contextual information
- Error event propagation for system-wide error handling

See [Error Handling](/docs/core/infrastructure/ERROR_HANDLING.md) for detailed implementation.

## Ongoing Development Tasks

### Priority Tasks
- Implement risk config for signal interpretation (reversal, long only, etc.)
- Run optimization with more data (5000+ bars) for better regime detection
- Test portfolio reset implementation with complex strategies
- Verify regime detection with the updated threshold values
- Adjust indicator parameters based on results

### Future Enhancements
- Enhance boundary trade handling for trades spanning multiple regimes
- Implement cross-validation for regime optimization
- Add attribution analysis for boundary trades (entry/exit regime combinations)
- Consider more sophisticated regime detection algorithms (HMM, statistical methods)
- Improve performance metrics and reporting with regime-specific statistics
- Implement strategy-specific regime definitions rather than one-size-fits-all

## Contributing

When contributing to this project, please:

1. Maintain the event-driven architecture
2. Follow the component lifecycle pattern
3. Add tests for new functionality
4. Update documentation to reflect changes
5. Respect the phased implementation approach

## Documentation Notes

This README.md serves as a consolidated reference for the ADMF-Trader framework, providing:

1. **High-level architectural overview** of the system and its components
2. **Implementation philosophy** guiding the development process
3. **Current status** of the project across implementation phases
4. **Key features** like regime-specific optimization and portfolio management
5. **Getting started guide** for basic usage and configuration

While this document provides a comprehensive overview, the original detailed documentation files contain implementation-specific information that may be valuable during development. Refer to the module-specific documentation linked above for detailed implementation patterns, code examples, and best practices.

The consolidation was performed as part of an effort to streamline documentation and provide a central entry point for understanding the ADMF-Trader framework.