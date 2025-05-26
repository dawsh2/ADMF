## Execution Module Documentation

### Overview

The Execution module is a core part of the ADMF-Trader system, tasked with order processing, market simulation, and the coordination of backtests. It takes orders from the Risk module, simulates how they would be executed in the market, including realistic slippage and commission, and creates fill events when orders are executed. A key architectural point is that the Execution module exclusively processes ORDER events from the Risk module and does not directly handle SIGNAL events, which are the Risk module's responsibility.

### Problem Statement

The ADMF-Trader system needs to function in various execution contexts, each with distinct threading demands: backtesting (historical simulation with flexible thread needs), optimization (running multiple backtests simultaneously), and live trading (handling real-time market data and orders). Without well-defined execution modes, issues such as inconsistent thread safety (leading to race conditions or performance hits), unclear concurrency assumptions, inconsistent thread management, performance bottlenecks due to excessive synchronization, and difficulty in understanding system behavior in different modes can arise.

### Key Components

The Execution module comprises the following main components:

* **ExecutionModule**
    * **OrderManager**: Manages the lifecycle and tracking of orders.
    * **Broker (Interface)**: An abstract interface for broker interactions.
        * **SimulatedBroker**: A broker for backtesting that simulates market conditions.
        * **PassthroughBroker**: A broker for testing and development that bypasses complex simulations.
    * **SlippageModels**: Simulate the price impact of trades.
        * **FixedSlippage**: Applies a fixed percentage for slippage.
        * **PercentageSlippage**: Calculates slippage based on order size.
        * **VolumeSlippage**: Slippage relative to trading volume.
    * **CommissionModels**: Calculate trading costs.
        * **FixedCommission**: A fixed charge per trade.
        * **PercentCommission**: Commission as a percentage of trade value.
        * **TieredCommission**: Commission rates that vary with trade size.
    * **BacktestCoordinator**: Manages and orchestrates backtest executions.

#### 1. Order Manager

The Order Manager receives ORDER events from the Risk module, validates these orders, and passes them to the correct broker. It is responsible for tracking the complete lifecycle of an order and maintaining the system's order state. Key duties include processing and validating incoming orders, tracking order status, forwarding valid orders, updating status based on fills, and keeping a history of orders. It utilizes `ThreadSafeDict` for active and completed orders and has a configurable limit for the number of completed orders to store. It subscribes to ORDER and FILL events via a `SubscriptionManager`.

#### 2. Broker

The Broker interface defines how orders from the Order Manager are processed and how fill events are generated upon execution. It's designed to implement slippage and commission models for realistic simulation. Concrete implementations must define methods for handling order events, executing orders, canceling orders, and retrieving order status.

#### 3. Simulated Broker

The `SimulatedBroker` is an implementation of the Broker interface designed for backtesting with realistic market simulation. It manages pending orders and the latest prices using `ThreadSafeDict` and incorporates configured slippage and commission models. It can create various slippage models (Fixed, Percentage, Volume-based) and commission models (Fixed, Percentage, Tiered) based on configuration, defaulting to fixed models with zero impact/cost if not specified. The `SimulatedBroker` processes order events by validating them and then handling different order types like MARKET and LIMIT orders based on available price data and order conditions. Orders that cannot be immediately processed are stored as pending.

#### 4. Slippage Models

Slippage models simulate the price impact of orders. The base `SlippageModel` class requires an `apply_slippage` method to be implemented. Specific models include `FixedSlippageModel` (adds a fixed percentage), `PercentageSlippageModel` (slippage based on order size), and `VolumeBasedSlippageModel` (slippage relative to order size and volume).

#### 5. Commission Models

Commission models are used to calculate trading costs. The base `CommissionModel` requires a `calculate_commission` method. Implementations include `FixedCommissionModel` (fixed amount per trade), `PercentageCommissionModel` (percentage of trade value), and `TieredCommissionModel` (rates based on trade size).

#### 6. Backtest Coordinator

The `BacktestCoordinator` orchestrates the backtesting process. Its responsibilities include managing component lifecycles, orchestrating data flow, ensuring positions are closed at the end of a backtest, and collecting, analyzing, and calculating performance statistics.

### Execution Modes

The ADMF-Trader system supports several execution modes to cater to different threading needs:

* `BACKTEST_SINGLE`: Single-threaded backtesting (fast, no thread safety needed).
* `BACKTEST_PARALLEL`: Multi-threaded backtest components (thread safety required).
* `OPTIMIZATION`: Parallel optimization (multiple backtest instances).
* `LIVE_TRADING`: Real-time market trading (multi-threaded, thread safety required).
* `PAPER_TRADING`: Simulated live trading (multi-threaded).
* `REPLAY`: Event replay mode (configurable threading model).

### Thread Models

Each execution mode is associated with a thread model:

* `SINGLE_THREADED`: All operations occur in a single thread.
* `MULTI_THREADED`: Operations can occur across multiple threads.
* `PROCESS_PARALLEL`: Parallel processes with internal thread management.
* `ASYNC_SINGLE`: Single event loop, asynchronous processing.
* `ASYNC_MULTI`: Multiple event loops, asynchronous processing.
* `MIXED`: Mixed model with custom thread management.

The default mapping between execution modes and thread models is defined, for example, `BACKTEST_SINGLE` maps to `SINGLE_THREADED`, and `LIVE_TRADING` maps to `ASYNC_MULTI`.

### Execution Context

The `ExecutionContext` class encapsulates the execution environment, including the execution mode and thread model. It uses thread-local storage (`threading.local()`) to manage the current context for each thread, with class methods to get and set the current context.

### Implementation Structure

The Execution module's code is organized into subdirectories: `interfaces`, `order`, `broker`, `models`, and `backtest`.

### Order Processing

#### 1. Order Lifecycle

Orders progress through a defined set of states: `RECEIVED`, `VALIDATED`, `ROUTED`, `PARTIAL`, `FILLED`, `CANCELLED`, `REJECTED`, and `EXPIRED`.

#### 2. Order Validation

The `OrderValidator` provides static methods to ensure orders have all necessary fields (e.g., `order_id`, `symbol`, `quantity`, `direction`, `order_type`) and that values are valid (e.g., non-zero quantity, valid direction, and order type).

#### 3. Order Execution

Different order types have specific execution rules:

* **Market Orders**: Execute immediately at the current price plus slippage.
* **Limit Orders**: Execute when the market price reaches or surpasses the limit price.
* **Stop Orders**: Execute when the price touches or crosses the stop price.
* **Stop-Limit Orders**: Become limit orders when the stop price is reached.

### Realistic Market Simulation

#### 1. OHLC Bar Execution Model

For realistic backtesting, order execution prices are determined based on OHLC (Open, High, Low, Close) bar data. For market buy orders, the high price of the bar might be used, and for market sell orders, the low price, representing conservative execution. For limit buy orders, if the bar's low is at or below the limit price, the limit price is used for execution; for limit sell orders, if the bar's high is at or above the limit price, the limit price is used.

#### 2. Volume Constraints

Order sizes can be limited by available volume within a bar, typically as a percentage of the bar's total volume (e.g., `max_volume_pct`).

#### 3. Realistic Fill Prices

Fill prices can incorporate models like VWAP (Volume Weighted Average Price), which can be approximated as (Open+High+Low+Close)/4 if volume data is available, otherwise defaulting to the close price.

### Passthrough Execution

The `PassthroughBroker` allows for strategy testing without the effects of slippage or commission. It generates an immediate fill event at the requested price (or limit price) with zero commission.

### Thread Management and Concurrency

#### 1. Thread Pool Management

The `ThreadPoolManager` creates and manages thread pools based on the execution mode and thread model defined in the `ExecutionContext`. For instance, in `SINGLE_THREADED` mode, no pools are needed. In `MULTI_THREADED` mode for `BACKTEST_PARALLEL`, limited worker thread pools for 'data' and 'compute' are created. For live trading (`is_live`), separate pools for 'market_data', 'order_processing', and 'strategy' are established. In `PROCESS_PARALLEL` mode (e.g., for `OPTIMIZATION`), a `ProcessPoolExecutor` is used. The manager provides a method to get an executor for a named pool and a shutdown method.

##### 1.1 Thread Affinity Management

Thread affinity can be set to bind specific threads to CPU cores for performance optimization, using OS-level functions like `os.sched_setaffinity`. Recommendations include dedicated cores for market data processing and order processing, multiple cores for strategy computation, and shared cores for background tasks.

#### 2. Thread Isolation Guidelines

The `ThreadIsolationGuidelines` class provides recommendations for the isolation level of components based on the execution context and component type (e.g., 'data_handler', 'strategy'). Isolation levels can be 'shared', 'isolated', 'process_isolated', or 'thread_isolated'. It also determines if locks should be used; for example, `BACKTEST_SINGLE` mode doesn't require locks, while live trading always does.

#### 3. Thread Synchronization Guidelines

The `ThreadSynchronizationGuidelines` class recommends synchronization primitives (e.g., 'none', 'lock', 'thread_local', 'event', 'queue') and locking strategies (e.g., 'none', 'reader_writer_lock', 'fine_grained_lock', 'reentrant_lock') based on the execution context and component type. Single-threaded mode requires no synchronization.

### Configuration Examples

YAML configuration examples are provided for:

* **Single-Threaded Backtest**: Specifies `BACKTEST_SINGLE` mode, `SINGLE_THREADED` model, and disables thread safety for components for performance.
* **Live Trading**: Specifies `LIVE_TRADING` mode, `MULTI_THREADED` model, configures thread pools for market data, order processing, and strategy, and enables thread safety for components.

### Best Practices

* **Order Management**: Track order lifecycles, use unique IDs, maintain thread-safe collections, validate orders, and handle partial fills.
* **Slippage Modeling**: Use varied models for different conditions, implement volume-based slippage, consider volatility, and adjust based on order size relative to volume.
* **Backtest Coordination**: Close positions at backtest end, calculate comprehensive metrics, track trades with attribution, manage component lifecycles, and ensure clean state resets.
* **Thread Safety**: Use thread-safe collections, appropriate locking, atomic operations where possible, document guarantees, and validate based on execution mode.

### Usage Examples

Python code snippets demonstrate:

* **Basic Backtesting**: Creating an `ExecutionContext` for `BACKTEST_SINGLE`, initializing components within the context, running a backtest engine, and printing performance statistics like total return and Sharpe ratio.
* **Live Trading**: Creating an `ExecutionContext` for `LIVE_TRADING`, initializing components, starting a trading system, and handling graceful shutdown.

### Advanced Performance Analysis

#### 1. Trade Analysis

A function `analyze_trades` is outlined to calculate detailed trade metrics such as win rate, profit factor, average win/loss, average holding time, max win/loss, and average trade size.

#### 2. Equity Curve Analysis

An `analyze_equity_curve` function is described to compute advanced metrics from an equity curve, including total return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, profit-to-drawdown ratio, and winning days percentage.

#### 3. Advanced Slippage Models

A `VolatilityAwareSlippageModel` is introduced as an example of an advanced model that adjusts slippage based on recent price volatility, using parameters like base slippage, volatility multiplier, and default volatility.

### Implementation Strategy Roadmap

The implementation is planned in steps:

1.  **Core Implementation**: `ExecutionMode`, `ThreadModel`, `ExecutionContext`, and system bootstrap updates.
2.  **Thread Management**: `ThreadPoolManager`, thread isolation guidelines, and thread affinity management.
3.  **Mode-Specific Components**: Coordinators and engines for single-threaded backtests, multi-threaded backtests, optimization, and live trading.
4.  **Testing**: Mode-specific test suites, thread safety validation, and performance benchmarks.

### Asynchronous Architecture Integration

For details on asynchronous implementation, a reference to `ASYNCHRONOUS_ARCHITECTURE.md` is provided, covering async component interfaces, event loop management, async-specific thread safety, and implementation patterns.

#### Hybrid Execution Model

The system supports a hybrid execution model, choosing synchronous or asynchronous paradigms based on the execution context (e.g., synchronous for `BACKTEST_SINGLE`, asynchronous for `LIVE_TRADING`) for optimal simplicity, performance, and resource utilization.

### Error Handling Framework

A comprehensive error handling approach includes:

1.  **Thread-specific error handling**: Managing thread interruption, thread-safe error reporting, and using timeouts.
2.  **Graceful recovery strategies**: Retry mechanisms with backoff, circuit breakers, and fallback options.
3.  **Shutdown coordination**: Graceful shutdown of thread pools, resource cleanup, and handling shutdown signals.

### Conclusion

The Execution module is vital for ADMF-Trader, offering realistic market simulation, order lifecycle management, flexible execution modes with appropriate thread management, performance optimization, advanced analysis tools, and support for hybrid execution models. Following the document's guidelines and best practices enables the development of robust trading strategies with realistic execution dynamics.
