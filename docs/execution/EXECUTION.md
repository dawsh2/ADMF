# Execution Module Documentation

## Overview

The Execution module is responsible for order processing, market simulation, and backtest coordination in the ADMF-Trader system. It receives orders from the Risk module, simulates market execution with realistic slippage and commission models, and generates fill events when orders are executed.

> **Important Architectural Note**: The Execution module ONLY processes ORDER events from the Risk module. It does NOT interact directly with SIGNAL events, which are handled exclusively by the Risk module.

## Key Components

### 1. Order Manager

The Order Manager receives ORDER events from the Risk module, validates them, and forwards them to the appropriate broker. It tracks the full order lifecycle and maintains the system's order state.

```python
class OrderManager(Component):
    """
    Order manager implementation.
    
    Manages order lifecycle, tracking, and processing.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize order collections
        self.active_orders = ThreadSafeDict()
        self.completed_orders = ThreadSafeDict()
        
        # Maximum history length
        self.max_completed_orders = self.parameters.get('max_completed_orders', 10000)
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Get broker
        self.broker = self._get_dependency(context, 'broker', required=True)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.ORDER, self.on_order)
        self.subscription_manager.subscribe(EventType.FILL, self.on_fill)
```

Key responsibilities:
- Process and validate incoming orders
- Track order status throughout lifecycle
- Forward valid orders to the broker
- Update order status based on fill events
- Maintain historical record of orders

### 2. Broker

The Broker processes orders received from the Order Manager and generates fill events when orders are executed. It implements slippage and commission models for realistic execution simulation.

```python
class BrokerBase(Component):
    """Abstract broker interface."""
    
    def on_order(self, order_event):
        """Process order event."""
        raise NotImplementedError
        
    def execute_order(self, order_data, price=None):
        """Execute an order."""
        raise NotImplementedError
        
    def cancel_order(self, order_id):
        """Cancel an order."""
        raise NotImplementedError
        
    def get_order_status(self, order_id):
        """Get order status."""
        raise NotImplementedError
```

### 3. Simulated Broker

The SimulatedBroker implements the Broker interface for backtesting with realistic market simulation:

```python
class SimulatedBroker(Component):
    """
    Simulated broker implementation.
    
    Simulates market execution with slippage and commission models.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize pending orders
        self.pending_orders = ThreadSafeDict()
        
        # Initialize latest prices
        self.latest_prices = ThreadSafeDict()
        
        # Create slippage model
        self.slippage_model = self._create_slippage_model()
        
        # Create commission model
        self.commission_model = self._create_commission_model()
```

Key features:
- Processes market, limit, stop, and stop-limit orders
- Applies realistic slippage models
- Calculates appropriate commissions
- Handles order cancellation
- Maintains price feed state

### 4. Slippage Models

Slippage models simulate price impact of orders:

```python
class SlippageModel:
    """
    Base class for slippage models.
    
    Simulates price impact of orders.
    """
    
    def __init__(self, parameters=None):
        """Initialize with parameters."""
        self.parameters = parameters or {}
        
    def apply_slippage(self, price, direction, quantity):
        """Apply slippage to price."""
        raise NotImplementedError
```

Implementations include:
- **FixedSlippageModel**: Adds a fixed percentage to prices
- **PercentageSlippageModel**: Slippage based on order size
- **VolumeBasedSlippageModel**: Slippage based on order size relative to volume

### 5. Commission Models

Commission models calculate trading costs:

```python
class CommissionModel:
    """
    Base class for commission models.
    
    Calculates trading costs.
    """
    
    def __init__(self, parameters=None):
        """Initialize with parameters."""
        self.parameters = parameters or {}
        
    def calculate_commission(self, price, quantity):
        """Calculate commission for a trade."""
        raise NotImplementedError
```

Implementations include:
- **FixedCommissionModel**: Charges a fixed amount per trade
- **PercentageCommissionModel**: Charges a percentage of trade value
- **TieredCommissionModel**: Different rates based on trade size

### 6. Backtest Coordinator

The Backtest Coordinator orchestrates the backtesting process:

```python
class BacktestCoordinator(Component):
    """
    Backtest coordinator.
    
    Orchestrates the backtesting process and collects results.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize component registry
        self.components = {}
        
        # Initialize results
        self.results = None
```

Key responsibilities:
- Manages the component lifecycle
- Orchestrates the data flow
- Closes positions at end of backtest
- Collects and analyzes results
- Calculates performance statistics

## Execution Modes

The ADMF-Trader system supports multiple execution modes with different threading requirements:

```python
from enum import Enum, auto

class ExecutionMode(Enum):
    """Execution mode options for the ADMF-Trader system."""
    BACKTEST_SINGLE = auto()  # Single-threaded backtesting (fast, no thread safety needed)
    BACKTEST_PARALLEL = auto()  # Multi-threaded backtest components (thread safety required)
    OPTIMIZATION = auto()      # Parallel optimization (multiple backtest instances)
    LIVE_TRADING = auto()      # Live market trading (multi-threaded, thread safety required)
    PAPER_TRADING = auto()     # Simulated live trading (multi-threaded)
    REPLAY = auto()            # Event replay mode (configurable threading model)
```

### Thread Models

Each execution mode has an associated thread model:

```python
from enum import Enum, auto

class ThreadModel(Enum):
    """Thread model options for execution contexts."""
    SINGLE_THREADED = auto()    # All operations in a single thread
    MULTI_THREADED = auto()     # Operations can occur across multiple threads
    PROCESS_PARALLEL = auto()   # Parallel processes with internal thread management
    ASYNC_SINGLE = auto()       # Single event loop, asynchronous processing
    ASYNC_MULTI = auto()        # Multiple event loops, asynchronous processing
    MIXED = auto()              # Mixed model with custom thread management
```

Default mapping between execution modes and thread models:

```python
DEFAULT_THREAD_MODELS = {
    ExecutionMode.BACKTEST_SINGLE: ThreadModel.SINGLE_THREADED,
    ExecutionMode.BACKTEST_PARALLEL: ThreadModel.MULTI_THREADED,
    ExecutionMode.OPTIMIZATION: ThreadModel.PROCESS_PARALLEL,
    ExecutionMode.LIVE_TRADING: ThreadModel.ASYNC_MULTI,
    ExecutionMode.PAPER_TRADING: ThreadModel.ASYNC_MULTI,
    ExecutionMode.REPLAY: ThreadModel.SINGLE_THREADED
}
```

## Execution Context

The ExecutionContext class encapsulates the execution environment:

```python
class ExecutionContext:
    """
    Execution context for the ADMF-Trader system.
    
    This class encapsulates the execution mode, thread model, and related
    configuration for a particular execution run.
    """
    
    _current_context = threading.local()  # Thread-local storage for current context
    
    @classmethod
    def get_current(cls):
        """Get current execution context for this thread."""
        return getattr(cls._current_context, 'context', None)
        
    @classmethod
    def set_current(cls, context):
        """Set current execution context for this thread."""
        cls._current_context.context = context
```

## Implementation Structure

```
src/execution/
├── __init__.py
├── interfaces/
│   ├── __init__.py
│   └── broker.py             # Broker interface
├── order/
│   ├── __init__.py
│   ├── order.py              # Order model
│   ├── order_manager.py      # Order management
│   └── order_validator.py    # Order validation
├── broker/
│   ├── __init__.py
│   ├── simulated_broker.py   # Simulated broker implementation
│   └── live_broker.py        # Live broker interface (placeholder)
├── models/
│   ├── __init__.py
│   ├── slippage.py           # Slippage models
│   └── commission.py         # Commission models
└── backtest/
    ├── __init__.py
    ├── coordinator.py        # Backtest coordination
    └── results.py            # Results collection and analysis
```

## Order Processing

### 1. Order Lifecycle

Orders follow a defined lifecycle:

```python
# Order states
ORDER_STATES = [
    'RECEIVED',    # Initial state
    'VALIDATED',   # Validation passed
    'ROUTED',      # Sent to broker
    'PARTIAL',     # Partially filled
    'FILLED',      # Completely filled
    'CANCELLED',   # Cancelled
    'REJECTED',    # Rejected
    'EXPIRED'      # Expired
]
```

### 2. Order Validation

The OrderValidator ensures orders contain all required fields:

```python
class OrderValidator:
    """
    Order validation utility.
    
    Static methods for validating order data.
    """
    
    @staticmethod
    def validate_order(order_data):
        """Validate order data."""
        # Check required fields
        required_fields = ['order_id', 'symbol', 'quantity', 'direction', 'order_type']
        for field in required_fields:
            if field not in order_data:
                return False
                
        # Check quantity
        if not isinstance(order_data['quantity'], (int, float)) or order_data['quantity'] == 0:
            return False
            
        # Check direction
        if order_data['direction'] not in ['BUY', 'SELL']:
            return False
            
        # Check order type
        if order_data['order_type'] not in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
            return False
```

### 3. Order Execution

Different order types are executed according to their rules:

- **Market Orders**: Execute immediately at current price with slippage
- **Limit Orders**: Execute when price reaches or exceeds limit price
- **Stop Orders**: Execute when price touches or crosses stop price
- **Stop-Limit Orders**: Convert to limit order when stop price is reached

## Realistic Market Simulation

### 1. OHLC Bar Execution Model

For realistic backtesting, orders are executed based on bar data:

```python
def determine_execution_price(self, order, bar):
    """Determine execution price within bar range."""
    direction = order['direction']
    order_type = order['order_type']
    
    if order_type == 'MARKET':
        # Use conservative prices for market orders
        if direction == 'BUY':
            # Buy at higher price
            return bar['high'] 
        else:
            # Sell at lower price
            return bar['low']
    elif order_type == 'LIMIT':
        limit_price = order['limit_price']
        
        if direction == 'BUY':
            # Determine if limit price was reached
            if bar['low'] <= limit_price:
                # Use limit price for execution
                return limit_price
        else:
            # Determine if limit price was reached
            if bar['high'] >= limit_price:
                # Use limit price for execution
                return limit_price
```

### 2. Volume Constraints

Orders can be limited by available volume:

```python
def apply_volume_constraints(self, order, bar):
    """Apply volume constraints to order size."""
    volume = bar['volume']
    quantity = abs(order['quantity'])
    
    # Get maximum volume percentage
    max_volume_pct = self.parameters.get('max_volume_pct', 0.1)
    
    # Calculate maximum allowed size
    max_size = int(volume * max_volume_pct)
    
    # Limit order size
    if quantity > max_size:
        # Update quantity in order
        limited_quantity = max_size
```

### 3. Realistic Fill Prices

Fill prices can incorporate VWAP and other realistic price models:

```python
def calculate_fill_price(self, order, bar):
    """Calculate realistic fill price."""
    # Get order details
    direction = order['direction']
    order_type = order['order_type']
    
    # Get bar details
    open_price = bar['open']
    high_price = bar['high']
    low_price = bar['low']
    close_price = bar['close']
    
    # Calculate VWAP if volume data available
    if 'volume' in bar and bar['volume'] > 0:
        # VWAP = (O+H+L+C)/4 as simplified approximation
        vwap = (open_price + high_price + low_price + close_price) / 4
    else:
        vwap = close_price
```

## Passthrough Execution

For strategy testing without execution effects, the system supports a "passthrough" mode:

```python
class PassthroughBroker(BrokerBase):
    """
    Passthrough broker implementation.
    
    Executes orders immediately without slippage or commission.
    Useful for strategy development and testing.
    """
    
    def on_order(self, event):
        """Process order event."""
        # Extract order data
        order_data = event.get_data()
        
        # Generate immediate fill at requested price without slippage
        fill_data = {
            'order_id': order_data['order_id'],
            'symbol': order_data['symbol'],
            'quantity': order_data['quantity'],
            'price': order_data.get('price', order_data.get('limit_price', 0)),
            'direction': order_data['direction'],
            'commission': 0.0,  # No commission in passthrough mode
            'timestamp': datetime.now()
        }
        
        # Emit fill event
        self._emit_fill(fill_data)
```

## Thread Management and Concurrency

### 1. Thread Pool Management

Thread pools are managed according to execution mode:

```python
class ThreadPoolManager:
    """
    Thread pool manager for execution modes.
    
    Creates and manages appropriate thread pools based on execution mode.
    """
    
    def __init__(self, context):
        """Initialize thread pool manager."""
        self.context = context
        self._thread_pools = {}
        
        # Configure based on execution mode and thread model
        self._configure()
        
    def _configure(self):
        """Configure thread pools based on context."""
        if self.context.thread_model == ThreadModel.SINGLE_THREADED:
            # No thread pools needed
            pass
        elif self.context.thread_model == ThreadModel.MULTI_THREADED:
            # Create thread pools based on execution mode
            if self.context.execution_mode == ExecutionMode.BACKTEST_PARALLEL:
                # Limited threads for parallel backtest
                max_workers = min(os.cpu_count(), 4)
                self._thread_pools['data'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, 
                    thread_name_prefix='data_worker'
                )
                self._thread_pools['compute'] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, 
                    thread_name_prefix='compute_worker'
                )
```

### 2. Thread Isolation Guidelines

Different components have different thread isolation requirements:

```python
class ThreadIsolationGuidelines:
    """Thread isolation guidelines for different execution modes."""
    
    @staticmethod
    def get_isolation_level(context, component_type):
        """Get recommended isolation level for component type in context."""
        # Default isolation levels by component type
        default_levels = {
            'data_handler': 'shared',
            'strategy': 'isolated',
            'risk_manager': 'shared',
            'portfolio': 'shared',
            'broker': 'shared',
            'order_manager': 'shared',
        }
        
        # Override based on execution mode
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            # All components shared in single thread
            return 'shared'
        elif context.execution_mode == ExecutionMode.OPTIMIZATION:
            # All components isolated by process
            return 'process_isolated'
        elif context.is_live:
            # Component-specific isolation in live trading
            live_overrides = {
                'data_handler': 'thread_isolated',
                'strategy': 'thread_isolated',
            }
            return live_overrides.get(component_type, default_levels.get(component_type, 'shared'))
```

### 3. Thread Synchronization Guidelines

```python
class ThreadSynchronizationGuidelines:
    """Thread synchronization guidelines for different execution modes."""
    
    @staticmethod
    def get_recommended_sync_primitives(context, component_type):
        """Get recommended synchronization primitives for component in context."""
        # Single-threaded mode needs no synchronization
        if context.execution_mode == ExecutionMode.BACKTEST_SINGLE:
            return ['none']
            
        # Default recommendations by component type
        default_primitives = {
            'data_handler': ['lock', 'thread_local'],
            'strategy': ['lock'],
            'risk_manager': ['lock'],
            'portfolio': ['lock'],
            'broker': ['lock', 'event'],
            'order_manager': ['lock', 'queue'],
        }
```

## Configuration Examples

### 1. Single-Threaded Backtest Configuration

```yaml
system:
  name: ADMF-Trader
  
execution:
  mode: BACKTEST_SINGLE
  thread_model: SINGLE_THREADED
  
backtest:
  start_date: 2022-01-01
  end_date: 2022-12-31
  symbols: [SPY, AAPL, MSFT]
  
components:
  data_handler:
    class: HistoricalDataHandler
    thread_safe: false  # Disable thread safety for performance
    
  strategy:
    class: MovingAverageCrossover
    thread_safe: false  # Disable thread safety for performance
```

### 2. Live Trading Configuration

```yaml
system:
  name: ADMF-Trader
  
execution:
  mode: LIVE_TRADING
  thread_model: MULTI_THREADED
  thread_pools:
    market_data:
      max_workers: 2
    order_processing:
      max_workers: 2
    strategy:
      max_workers: 2
  
live_trading:
  symbols: [SPY, AAPL, MSFT]
  
components:
  data_handler:
    class: LiveMarketDataHandler
    thread_safe: true
    
  strategy:
    class: MovingAverageCrossover
    thread_safe: true
    
  broker:
    class: LiveBrokerHandler
    thread_safe: true
```

## Best Practices

### 1. Order Management

- Track order lifecycle states with history
- Use order IDs for consistent tracking
- Maintain thread-safe order collections
- Implement order validation before processing
- Handle partial fills appropriately

### 2. Slippage Modeling

- Use different slippage models for different market conditions
- Implement volume-based slippage for realistic simulation
- Consider price volatility when calculating slippage
- Adjust slippage based on order size relative to volume

### 3. Backtest Coordination

- Properly close all positions at the end of a backtest
- Calculate comprehensive performance metrics
- Track trades with detailed attribution
- Implement proper component lifecycle management
- Ensure clean state reset between runs

### 4. Thread Safety

- Use thread-safe collections for shared state
- Implement appropriate locking based on thread model
- Use atomic operations where possible
- Document thread safety guarantees for components
- Validate thread safety based on execution mode

## Usage Examples

### Basic Backtesting

```python
# Create execution context
context = ExecutionContext('backtest', ExecutionMode.BACKTEST_SINGLE)

# Run backtest within context
with context:
    # Initialize components
    container = create_container(config, context)
    
    # Run backtest
    backtest_engine = container.resolve('backtest_engine')
    results = backtest_engine.run()
    
    # Analyze results
    performance = results['statistics']
    print(f"Total return: {performance['total_return_pct']:.2f}%")
    print(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
```

### Live Trading

```python
# Create execution context
context = ExecutionContext('live_trading', ExecutionMode.LIVE_TRADING)

# Run live trading within context
with context:
    # Initialize components
    container = create_container(config, context)
    
    # Start live trading system
    trading_system = container.resolve('trading_system')
    trading_system.start()
    
    try:
        # Run until stopped
        while trading_system.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle graceful shutdown
        trading_system.stop()
```

## Conclusion

The Execution module is a critical component of the ADMF-Trader system, responsible for order processing, market simulation, and backtest coordination. It provides:

1. Realistic market simulation with configurable slippage and commissions
2. Comprehensive order lifecycle management
3. Flexible execution modes for different operating contexts
4. Thread management appropriate to each execution mode
5. Performance optimization through context-aware thread safety

By following the guidelines and best practices outlined in this document, you can effectively utilize the Execution module to build robust trading strategies with realistic execution dynamics.