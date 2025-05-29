## Execution Module Documentation

### Overview

The Execution module is a core part of the ADMF-Trader system, tasked with order processing, market simulation, and the coordination of backtests. It takes orders from the Risk module, simulates how they would be executed in the market, including realistic slippage and commission, and creates fill events when orders are executed. A key architectural point is that the Execution module exclusively processes ORDER events from the Risk module and does not directly handle SIGNAL events, which are the Risk module's responsibility.

**Critical Architecture Update**: The Execution module implements the **Scoped Container Pattern** for complete state isolation between execution contexts. This ensures that parallel backtests, optimization trials, and live trading strategies operate in completely isolated environments with no possibility of state leakage.

### Problem Statement

The ADMF-Trader system needs to function in various execution contexts, each with distinct threading demands and isolation requirements: backtesting (historical simulation with flexible thread needs and complete state isolation), optimization (running multiple backtests simultaneously with guaranteed independence), and live trading (handling real-time market data and orders with strategy isolation). Without well-defined execution modes AND proper container-based isolation, issues such as:
- State leakage between parallel executions
- Inconsistent results in optimization
- Race conditions from shared mutable state
- Inability to guarantee reproducible results
- Difficulty debugging issues in specific execution contexts

### Architectural Foundation: Scoped Containers

#### The Container-Based Isolation Model

The Execution module uses **Scoped Containers** as its primary isolation mechanism. Each execution context (backtest, optimization trial, live strategy) runs within its own container with:

1. **Isolated Component Instances**: Each container has its own instances of:
   - Portfolio (tracks positions and P&L independently)
   - Risk Manager (applies rules without interference)
   - Broker (simulates execution in isolation)
   - Event Bus (events don't leak between contexts)
   - Strategy (maintains its own state)

2. **Shared Read-Only Services**: Containers can access shared, immutable services:
   - Historical market data
   - Configuration settings
   - System-wide parameters

3. **Complete Lifecycle Management**: Each container manages the full lifecycle of its components from initialization through teardown.

### Key Components

The Execution module comprises the following main components:

* **ExecutionModule**
    * **OrderManager**: Manages the lifecycle and tracking of orders (created per container)
    * **Broker (Interface)**: An abstract interface for broker interactions
        * **SimulatedBroker**: A broker for backtesting that simulates market conditions (one per container)
        * **PassthroughBroker**: A broker for testing and development that bypasses complex simulations
    * **SlippageModels**: Simulate the price impact of trades (can be shared or per-container)
    * **CommissionModels**: Calculate trading costs (can be shared or per-container)
    * **BacktestCoordinator**: Creates and manages scoped containers for backtest executions
    * **OptimizationExecutor**: Orchestrates multiple containers for parallel optimization
    * **LiveTradingManager**: Manages isolated containers for live trading strategies

### The BacktestCoordinator: Container Orchestration

The `BacktestCoordinator` is the central component responsible for creating and managing scoped containers. It ensures complete isolation between backtest runs.

```python
class BacktestCoordinator:
    """Orchestrates backtesting using scoped containers for isolation"""
    
    def __init__(self, shared_services: Dict[str, Any]):
        self.shared_services = shared_services
        self.container_factory = ContainerFactory(shared_services)
        self.active_containers: Dict[str, ScopedContainer] = {}
        self._lock = threading.RLock()
    
    def run_backtest(self, backtest_config: Dict[str, Any]) -> BacktestResults:
        """Run a single backtest in an isolated container"""
        
        # 1. Create a fresh container for this backtest
        container = self._create_backtest_container(backtest_config)
        container_id = container.container_id
        
        try:
            # 2. Initialize all components in the container
            container.initialize_scope()
            
            # 3. Get core components from container
            data_handler = container.resolve("data_handler")
            strategy = container.resolve("strategy")
            portfolio = container.resolve("portfolio")
            risk_manager = container.resolve("risk_manager")
            broker = container.resolve("broker")
            event_bus = container.resolve("event_bus")
            
            # 4. Run the backtest loop
            results = self._execute_backtest_loop(
                container, 
                data_handler, 
                event_bus,
                backtest_config
            )
            
            # 5. Close all positions at end
            self._close_all_positions(container, portfolio, broker)
            
            # 6. Collect final results
            results.update(self._collect_results(container))
            
            return results
            
        finally:
            # 7. Always teardown container to ensure cleanup
            container.teardown_scope()
            self._remove_container(container_id)
    
    def _create_backtest_container(self, config: Dict[str, Any]) -> ScopedContainer:
        """Create a properly configured backtest container"""
        
        container = ScopedContainer(f"backtest_{uuid.uuid4().hex[:8]}")
        
        # Register shared read-only services
        container.register_shared_service("market_data", self.shared_services["market_data"])
        container.register_shared_service("config", self.shared_services["config"])
        
        # Create container-specific components
        
        # Data Handler (with isolated state for train/test splits)
        container.create_component({
            'name': 'data_handler',
            'class': 'HistoricalDataHandler',
            'params': {
                'symbols': config['symbols'],
                'start_date': config['start_date'],
                'end_date': config['end_date']
            }
        })
        
        # Strategy (with isolated state)
        container.create_component({
            'name': 'strategy',
            'class': config['strategy_class'],
            'params': config['strategy_params']
        })
        
        # Portfolio (completely isolated P&L tracking)
        container.create_component({
            'name': 'portfolio',
            'class': 'Portfolio',
            'params': {
                'initial_cash': config.get('initial_cash', 100000),
                'currency': config.get('currency', 'USD')
            }
        })
        
        # Risk Manager (isolated risk state)
        container.create_component({
            'name': 'risk_manager',
            'class': 'RiskManager',
            'params': config.get('risk_params', {})
        })
        
        # Broker (isolated execution simulation)
        container.create_component({
            'name': 'broker',
            'class': 'SimulatedBroker',
            'params': {
                'slippage_model': config.get('slippage_model'),
                'commission_model': config.get('commission_model')
            }
        })
        
        # Event Bus is automatically created by container
        
        return container
    
    def _execute_backtest_loop(self, container: ScopedContainer, 
                               data_handler: DataHandler,
                               event_bus: EventBus,
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main backtest event loop"""
        
        results = {
            'events_processed': 0,
            'bars_processed': 0
        }
        
        # Main backtest loop
        while data_handler.continue_backtest:
            try:
                # Get next event from queue
                event = event_bus.get_next_event()
                
                if event is None:
                    # No events, update bars
                    if data_handler.update_bars():
                        results['bars_processed'] += 1
                    else:
                        # No more data
                        break
                else:
                    # Event is processed by components via their subscriptions
                    # The isolated event bus ensures events don't leak
                    results['events_processed'] += 1
                    
            except Exception as e:
                container.logger.error(f"Error in backtest loop: {e}")
                if config.get('stop_on_error', True):
                    raise
                    
        return results
```

### Container Lifecycle in Backtesting

```
┌─────────────────────────────────────────────────────────────┐
│                 Backtest Execution Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Create Container       2. Initialize Components         │
│  ┌─────────────────┐      ┌─────────────────────┐         │
│  │ ScopedContainer │      │ • Data Handler      │         │
│  │   "backtest_1"  │ ---> │ • Strategy          │         │
│  │                 │      │ • Portfolio         │         │
│  └─────────────────┘      │ • Risk Manager      │         │
│                           │ • Broker            │         │
│                           │ • Event Bus         │         │
│                           └─────────────────────┘         │
│                                    │                        │
│  3. Run Backtest Loop              ▼                       │
│  ┌─────────────────────────────────────────────┐          │
│  │  while data available:                      │          │
│  │    - Update bars → BAR events               │          │
│  │    - Strategy → SIGNAL events               │          │
│  │    - Risk Manager → ORDER events            │          │
│  │    - Broker → FILL events                   │          │
│  │    - Portfolio updates                      │          │
│  └─────────────────────────────────────────────┘          │
│                                    │                        │
│  4. Finalize                       ▼                       │
│  ┌─────────────────────────────────────────────┐          │
│  │ • Close all positions                       │          │
│  │ • Calculate final metrics                   │          │
│  │ • Collect results                           │          │
│  └─────────────────────────────────────────────┘          │
│                                    │                        │
│  5. Teardown                       ▼                       │
│  ┌─────────────────────────────────────────────┐          │
│  │ • Stop all components                       │          │
│  │ • Unsubscribe events                        │          │
│  │ • Release resources                         │          │
│  │ • Destroy container                         │          │
│  └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Parallel Optimization with Container Isolation

The `OptimizationExecutor` leverages scoped containers to run multiple backtests in parallel with complete isolation:

```python
class OptimizationExecutor:
    """Executes optimization using parallel isolated containers"""
    
    def __init__(self, shared_services: Dict[str, Any], max_workers: int = None):
        self.backtest_coordinator = BacktestCoordinator(shared_services)
        self.max_workers = max_workers or cpu_count()
        self.results_collector = OptimizationResultsCollector()
    
    def run_optimization(self, base_config: Dict[str, Any], 
                        parameter_space: List[Dict[str, Any]]) -> OptimizationResults:
        """Run parallel optimization with isolated containers"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all parameter combinations
            futures = []
            for params in parameter_space:
                # Merge base config with specific parameters
                backtest_config = {**base_config, 'strategy_params': params}
                
                # Each future runs in its own container
                future = executor.submit(
                    self.backtest_coordinator.run_backtest,
                    backtest_config
                )
                futures.append((params, future))
            
            # Collect results
            for params, future in futures:
                try:
                    result = future.result()
                    self.results_collector.add_result(params, result)
                except Exception as e:
                    self.results_collector.add_error(params, e)
        
        return self.results_collector.get_optimization_results()
```

### Live Trading with Container Isolation

Each live trading strategy runs in its own container to ensure complete isolation:

```python
class LiveTradingManager:
    """Manages live trading strategies in isolated containers"""
    
    def __init__(self, shared_services: Dict[str, Any]):
        self.shared_services = shared_services
        self.container_factory = ContainerFactory(shared_services)
        self.active_strategies: Dict[str, LiveStrategyContainer] = {}
        self._lock = threading.RLock()
    
    def add_strategy(self, strategy_id: str, strategy_config: Dict[str, Any]) -> None:
        """Add a new live trading strategy in its own container"""
        
        with self._lock:
            if strategy_id in self.active_strategies:
                raise ValueError(f"Strategy {strategy_id} already exists")
            
            # Create isolated container for this strategy
            container = self._create_live_container(strategy_id, strategy_config)
            
            # Initialize components
            container.initialize_scope()
            
            # Create management wrapper
            strategy_container = LiveStrategyContainer(
                container_id=container.container_id,
                container=container,
                config=strategy_config
            )
            
            # Start strategy execution thread
            strategy_container.start()
            
            self.active_strategies[strategy_id] = strategy_container
    
    def _create_live_container(self, strategy_id: str, 
                              config: Dict[str, Any]) -> ScopedContainer:
        """Create container for live trading strategy"""
        
        container = ScopedContainer(f"live_{strategy_id}")
        
        # Shared services
        container.register_shared_service("market_data_feed", 
                                        self.shared_services["market_data_feed"])
        container.register_shared_service("order_router", 
                                        self.shared_services["order_router"])
        
        # Strategy-specific components
        container.create_component({
            'name': 'strategy',
            'class': config['strategy_class'],
            'params': config['strategy_params']
        })
        
        # Virtual portfolio for tracking
        container.create_component({
            'name': 'portfolio',
            'class': 'VirtualPortfolio',
            'params': {
                'strategy_id': strategy_id,
                'initial_capital': config.get('allocated_capital', 100000)
            }
        })
        
        # Risk manager with strategy-specific limits
        container.create_component({
            'name': 'risk_manager',
            'class': 'RiskManager',
            'params': {
                'max_position_size': config.get('max_position_size'),
                'max_portfolio_risk': config.get('max_portfolio_risk')
            }
        })
        
        # Live broker interface
        container.create_component({
            'name': 'broker',
            'class': 'LiveBroker',
            'params': {
                'order_router': 'order_router',  # Resolved from shared services
                'strategy_id': strategy_id
            }
        })
        
        return container
```

### Benefits of Container-Based Execution

1. **Complete State Isolation**: Each backtest/strategy has its own:
   - Portfolio state (positions, cash, P&L)
   - Strategy state (indicators, signals)
   - Risk state (limits, exposure tracking)
   - Event flow (no event leakage)

2. **Reproducible Results**: 
   - Same parameters always produce same results
   - No interference from parallel executions
   - Clean state for each run

3. **Simplified Testing**:
   - Easy to test individual components in isolation
   - Can mock shared services per container
   - Debugging is straightforward with contained state

4. **Resource Management**:
   - Containers are cleaned up after use
   - Memory is released properly
   - No resource leaks between runs

5. **Scalability**:
   - Easy to parallelize backtests
   - Each container is independent
   - Can scale to available CPU cores

### Container Configuration Examples

```yaml
# Backtest container configuration
backtest:
  container_type: "backtest"
  isolation_level: "full"
  components:
    data_handler:
      class: "HistoricalDataHandler"
      params:
        cache_size: 10000
        enable_splits: true
    strategy:
      class: "TrendFollowingStrategy"
      params:
        fast_period: 10
        slow_period: 30
    portfolio:
      class: "Portfolio"
      params:
        initial_cash: 100000
        track_history: true
    risk_manager:
      class: "RiskManager"
      params:
        max_position_size: 1000
        max_portfolio_heat: 0.06
    broker:
      class: "SimulatedBroker"
      params:
        slippage_model: "fixed"
        commission_model: "per_share"

# Optimization container configuration  
optimization:
  container_type: "optimization"
  parallelism: "process"
  max_workers: 8
  container_pool_size: 16  # Pre-create containers
  memory_limit_per_container: "512MB"
  
# Live trading container configuration
live_trading:
  container_type: "live"
  isolation_level: "full"
  monitoring_enabled: true
  health_check_interval: 60
  components:
    strategy:
      class: "ProductionStrategy"
      error_handling: "graceful"
    portfolio:
      class: "VirtualPortfolio"
      reconciliation_interval: 300
```

### Best Practices for Container-Based Execution

1. **Container Lifecycle Management**:
   - Always use try/finally to ensure teardown
   - Set reasonable timeouts for container operations
   - Monitor container health in production

2. **Resource Limits**:
   - Set memory limits per container
   - Limit number of concurrent containers
   - Use container pools for frequent operations

3. **State Management**:
   - Never share mutable state between containers
   - Use shared services only for read-only data
   - Ensure components properly implement reset()

4. **Error Handling**:
   - Containers should gracefully handle component failures
   - Log errors with container context
   - Implement circuit breakers for production

5. **Performance Optimization**:
   - Pre-create containers for hot paths
   - Reuse containers where safe (with proper reset)
   - Profile container creation overhead

### Integration with Execution Modes

The container-based architecture integrates seamlessly with execution modes:

- **BACKTEST_SINGLE**: Single container, single thread
- **BACKTEST_PARALLEL**: Multiple containers, thread pool
- **OPTIMIZATION**: Many containers, process pool
- **LIVE_TRADING**: Long-lived containers per strategy
- **PAPER_TRADING**: Similar to live but with simulated execution

Each execution mode can leverage containers while maintaining its specific threading model and performance characteristics.

### Execution Modes

The ADMF-Trader system supports several execution modes to cater to different threading needs:

* `BACKTEST_SINGLE`: Single-threaded backtesting in a single container
* `BACKTEST_PARALLEL`: Multi-threaded backtest with multiple containers
* `OPTIMIZATION`: Parallel optimization with many isolated containers
* `LIVE_TRADING`: Real-time market trading with per-strategy containers
* `PAPER_TRADING`: Simulated live trading with isolated containers
* `REPLAY`: Event replay mode with configurable container isolation

### Thread Models

Each execution mode is associated with a thread model that works within the container architecture:

* `SINGLE_THREADED`: All operations occur in a single thread within one container
* `MULTI_THREADED`: Operations across multiple threads, each potentially with its own container
* `PROCESS_PARALLEL`: Parallel processes, each managing one or more containers
* `ASYNC_SINGLE`: Single event loop within a container
* `ASYNC_MULTI`: Multiple event loops across containers
* `MIXED`: Mixed model with custom thread and container management

### Order Processing

#### 1. Order Lifecycle

Orders progress through a defined set of states within their container's broker:
- `RECEIVED`, `VALIDATED`, `ROUTED`, `PARTIAL`, `FILLED`, `CANCELLED`, `REJECTED`, `EXPIRED`

#### 2. Order Validation

The `OrderValidator` provides static methods to ensure orders have all necessary fields and valid values. Each container's broker performs its own validation.

#### 3. Order Execution

Different order types have specific execution rules, all processed within the container's isolated broker instance.

### Realistic Market Simulation

Each container's `SimulatedBroker` provides realistic market simulation:

#### 1. OHLC Bar Execution Model

For realistic backtesting, order execution prices are determined based on OHLC data within each container.

#### 2. Volume Constraints

Order sizes can be limited by available volume, configured per container.

#### 3. Realistic Fill Prices

Fill prices incorporate models like VWAP, calculated within each container's broker.

### Container-Based Performance Analysis

#### 1. Per-Container Metrics

Each container tracks its own performance metrics:

```python
class ContainerMetrics:
    """Performance metrics for a single container"""
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.start_time = datetime.now()
        self.events_processed = 0
        self.orders_generated = 0
        self.fills_executed = 0
        self.final_portfolio_value = None
        self.max_drawdown = 0
        self.sharpe_ratio = None
```

#### 2. Aggregated Analysis

The system can aggregate metrics across containers for optimization analysis:

```python
class OptimizationAnalyzer:
    """Analyzes results across multiple containers"""
    
    def analyze_optimization_results(self, 
                                   container_results: List[ContainerMetrics]) -> Dict:
        """Aggregate and analyze results from multiple containers"""
        
        best_sharpe = max(r.sharpe_ratio for r in container_results)
        best_return = max(r.final_portfolio_value for r in container_results)
        
        return {
            'total_containers': len(container_results),
            'best_sharpe_ratio': best_sharpe,
            'best_return': best_return,
            'average_execution_time': np.mean([r.execution_time for r in container_results])
        }
```

### Implementation Strategy Roadmap

The implementation is planned in steps:

1. **Core Container Implementation**: 
   - `ScopedContainer` class with lifecycle management
   - Component factory and registration
   - Container-specific event bus

2. **BacktestCoordinator Enhancement**:
   - Container creation and management
   - Lifecycle orchestration
   - Result collection

3. **Parallel Execution**:
   - `OptimizationExecutor` with container pools
   - Resource management across containers
   - Result aggregation

4. **Live Trading Integration**:
   - `LiveTradingManager` with persistent containers
   - Health monitoring per container
   - Strategy isolation

5. **Testing**:
   - Container isolation verification
   - Performance benchmarks
   - Resource leak detection

### Conclusion

The Execution module, enhanced with scoped containers, provides a robust foundation for ADMF-Trader's various execution modes. By ensuring complete isolation between execution contexts, the system achieves reproducible results, enables safe parallelization, and simplifies debugging and testing. The container-based architecture is fundamental to the system's ability to scale from simple backtests to complex production deployments while maintaining clean separation of concerns.
