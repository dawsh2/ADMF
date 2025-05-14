# Execution Module Implementation Guide

## Overview

The Execution module is responsible for order processing, market simulation, and backtest coordination in the ADMF-Trader system. It handles the crucial transition from trading decisions to market actions by processing ORDER events from the Risk module and generating FILL events when orders are executed. The module implements realistic market simulation with configurable slippage and commission models for backtesting while providing a pathway to live trading implementations.

> **Important Architectural Note**: The Execution module ONLY processes ORDER events from the Risk module. It does NOT interact directly with SIGNAL events, which are handled exclusively by the Risk module.

## Directory Structure

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
│   ├── passthrough_broker.py # Passthrough broker for strategy testing
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

## Core Components

### 1. Order Manager

The Order Manager is a central component that manages the entire order lifecycle. It receives ORDER events from the Risk module, validates them, and forwards them to the appropriate broker for execution.

#### Key Responsibilities:

- Process incoming orders from the Risk module
- Track order status throughout the order lifecycle
- Manage order history and status transitions
- Handle order cancellations
- Process fill events from the broker
- Maintain order collections (active and completed)

#### Implementation Example:

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
        
    def on_order(self, event):
        """
        Process order event.
        
        Args:
            event: Order event
        """
        # Extract order data
        order_data = event.get_data()
        
        # Validate order
        if not OrderValidator.validate_order(order_data):
            self.logger.warning(f"Invalid order: {order_data}")
            return
            
        # Check for duplicate order
        order_id = order_data['order_id']
        if order_id in self.active_orders:
            self.logger.warning(f"Duplicate order ID: {order_id}")
            return
            
        # Store order with status
        order_data['status'] = 'RECEIVED'
        order_data['received_time'] = order_data.get('timestamp', datetime.now())
        
        # Track order history
        if 'status_history' not in order_data:
            order_data['status_history'] = []
            
        order_data['status_history'].append({
            'timestamp': order_data['received_time'],
            'status': 'RECEIVED',
            'details': 'Order received by order manager'
        })
        
        self.active_orders[order_id] = order_data
        
        # Forward order to broker
        self.broker.process_order(order_data)
        
    def on_fill(self, event):
        """
        Process fill event.
        
        Args:
            event: Fill event
        """
        # Extract fill data
        fill_data = event.get_data()
        
        # Check for order ID
        order_id = fill_data.get('order_id')
        if not order_id:
            self.logger.warning(f"Fill without order ID: {fill_data}")
            return
            
        # Update order if it exists
        if order_id in self.active_orders:
            # Get order
            order = self.active_orders[order_id]
            
            # Update status
            order['status'] = 'FILLED'
            order['filled_time'] = fill_data.get('timestamp', datetime.now())
            
            # Add to status history
            order['status_history'].append({
                'timestamp': order['filled_time'],
                'status': 'FILLED',
                'details': f"Order filled at price {fill_data.get('price')}"
            })
            
            # Add fill information
            if 'fills' not in order:
                order['fills'] = []
                
            order['fills'].append(fill_data.copy())
            
            # Move to completed orders
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]
            
            # Prune completed orders if needed
            self._prune_completed_orders()
```

### 2. Broker Interface

The Broker is an abstract interface for market execution systems. It processes orders and generates fill events when trades are executed.

#### Key Responsibilities:

- Process orders from the Order Manager
- Apply slippage models to simulate realistic price impacts
- Apply commission models to simulate trading costs
- Generate fill events when orders are executed
- Track order status and execution details

#### Interface Definition:

```python
class BrokerBase(Component):
    """
    Base class for broker implementations.
    
    Handles order execution and generates fill events.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
    
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.ORDER, self.on_order)
    
    def on_order(self, event):
        """
        Process order event.
        
        Args:
            event: Order event
        """
        # Extract order data
        order_data = event.get_data()
        
        # Process order
        self.process_order(order_data)
    
    def process_order(self, order_data):
        """
        Process an order.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            bool: Success or failure
        """
        raise NotImplementedError("Subclasses must implement process_order")
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            bool: Success or failure
        """
        raise NotImplementedError("Subclasses must implement cancel_order")
    
    def get_order_status(self, order_id):
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            dict: Order status information
        """
        raise NotImplementedError("Subclasses must implement get_order_status")
```

### 3. Simulated Broker Implementation

The SimulatedBroker implements realistic market simulation for backtesting with configurable slippage and commission models.

#### Key Features:

- Processes orders from the Order Manager
- Receives market data updates (bars)
- Applies slippage models to simulate price impact
- Applies commission models to simulate trading costs
- Handles different order types (market, limit, stop, stop-limit)
- Generates fill events when orders are executed

#### Implementation Example:

```python
class SimulatedBroker(BrokerBase):
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
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.ORDER, self.on_order)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def _create_slippage_model(self):
        """Create slippage model from configuration."""
        slippage_config = self.parameters.get('slippage', {})
        model_type = slippage_config.get('model', 'fixed')
        
        if model_type == 'fixed':
            return FixedSlippageModel(slippage_config)
        elif model_type == 'percentage':
            return PercentageSlippageModel(slippage_config)
        elif model_type == 'volume':
            return VolumeBasedSlippageModel(slippage_config)
        else:
            # Default to fixed slippage
            return FixedSlippageModel({'price_impact': 0.0})
            
    def _create_commission_model(self):
        """Create commission model from configuration."""
        commission_config = self.parameters.get('commission', {})
        model_type = commission_config.get('model', 'fixed')
        
        if model_type == 'fixed':
            return FixedCommissionModel(commission_config)
        elif model_type == 'percentage':
            return PercentageCommissionModel(commission_config)
        elif model_type == 'tiered':
            return TieredCommissionModel(commission_config)
        else:
            # Default to fixed commission
            return FixedCommissionModel({'cost_per_trade': 0.0})
            
    def process_order(self, order_data):
        """
        Process an order.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            bool: Success or failure
        """
        # Validate order
        if not OrderValidator.validate_order(order_data):
            return False
            
        # Get order details
        order_id = order_data['order_id']
        symbol = order_data['symbol']
        order_type = order_data['order_type']
        
        # Check if we have price data for this symbol
        if symbol not in self.latest_prices and order_type != 'MARKET':
            # Store as pending order
            self.pending_orders[order_id] = order_data
            return True
            
        # Process different order types
        if order_type == 'MARKET':
            # Market orders execute immediately if we have a price
            if symbol in self.latest_prices:
                price = self.latest_prices[symbol]
                self._execute_order(order_data, price)
            else:
                # Store as pending order until we get a price
                self.pending_orders[order_id] = order_data
        elif order_type == 'LIMIT':
            # Check if limit price is reached
            limit_price = order_data['limit_price']
            
            if symbol in self.latest_prices:
                price = self.latest_prices[symbol]
                
                if (order_data['direction'] == 'BUY' and price <= limit_price) or \
                   (order_data['direction'] == 'SELL' and price >= limit_price):
                    # Limit price reached, execute order
                    self._execute_order(order_data, price)
                else:
                    # Store as pending order
                    self.pending_orders[order_id] = order_data
            else:
                # Store as pending order
                self.pending_orders[order_id] = order_data
        else:
            # Store as pending order
            self.pending_orders[order_id] = order_data
            
        return True
```

### 4. Slippage Models

Slippage models simulate the price impact of orders in the market. The ADMF-Trader system provides several slippage models to capture different market dynamics.

#### Base Slippage Model Interface:

```python
class SlippageModel:
    """
    Base class for slippage models.
    
    Simulates price impact of orders.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize with parameters.
        
        Args:
            parameters: Slippage model parameters
        """
        self.parameters = parameters or {}
        
    def apply_slippage(self, price, direction, quantity):
        """
        Apply slippage to price.
        
        Args:
            price: Base price
            direction: Order direction ('BUY' or 'SELL')
            quantity: Order quantity
            
        Returns:
            float: Adjusted price
        """
        raise NotImplementedError
```

#### Implementation Types:

1. **Fixed Slippage Model**: Adds a fixed percentage to the price
2. **Percentage Slippage Model**: Slippage based on a percentage of price and quantity
3. **Volume-Based Slippage Model**: Slippage based on quantity relative to volume

```python
class FixedSlippageModel(SlippageModel):
    """
    Fixed slippage model.
    
    Adds a fixed amount to price.
    """
    
    def apply_slippage(self, price, direction, quantity):
        """
        Apply fixed slippage to price.
        
        Args:
            price: Base price
            direction: Order direction ('BUY' or 'SELL')
            quantity: Order quantity
            
        Returns:
            float: Adjusted price
        """
        # Get price impact
        price_impact = self.parameters.get('price_impact', 0.0)
        
        # Apply impact based on direction
        if direction == 'BUY':
            return price * (1.0 + price_impact)
        else:  # SELL
            return price * (1.0 - price_impact)
```

### 5. Commission Models

Commission models simulate trading costs. The system provides several commission models to simulate different broker fee structures.

#### Base Commission Model Interface:

```python
class CommissionModel:
    """
    Base class for commission models.
    
    Calculates trading costs.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize with parameters.
        
        Args:
            parameters: Commission model parameters
        """
        self.parameters = parameters or {}
        
    def calculate_commission(self, price, quantity):
        """
        Calculate commission for a trade.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            
        Returns:
            float: Commission amount
        """
        raise NotImplementedError
```

#### Implementation Types:

1. **Fixed Commission Model**: Charges a fixed amount per trade
2. **Percentage Commission Model**: Charges a percentage of trade value
3. **Tiered Commission Model**: Different rates based on trade size

```python
class PercentageCommissionModel(CommissionModel):
    """
    Percentage commission model.
    
    Charges a percentage of trade value.
    """
    
    def calculate_commission(self, price, quantity):
        """
        Calculate percentage commission.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            
        Returns:
            float: Commission amount
        """
        # Get percentage
        percentage = self.parameters.get('percentage', 0.0)
        
        # Get minimum commission
        min_commission = self.parameters.get('min_commission', 0.0)
        
        # Calculate trade value
        trade_value = abs(price * quantity)
        
        # Calculate commission
        commission = trade_value * percentage
        
        # Apply minimum
        return max(commission, min_commission)
```

### 6. Backtest Coordinator

The BacktestCoordinator orchestrates the backtesting process, manages component lifecycle, and collects results.

#### Key Responsibilities:

- Initialize all system components
- Process market data through the data handler
- Close positions at the end of the backtest
- Collect and analyze backtest results
- Calculate performance statistics

#### Implementation Example:

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
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Register essential components
        self.components['event_bus'] = self._get_dependency(context, 'event_bus', required=True)
        self.components['data_handler'] = self._get_dependency(context, 'data_handler', required=True)
        self.components['portfolio'] = self._get_dependency(context, 'portfolio', required=True)
        
        # Register optional components
        for name in ['strategy', 'risk_manager', 'broker', 'order_manager']:
            component = self._get_dependency(context, name, required=False)
            if component:
                self.components[name] = component
                
    def run(self):
        """
        Run backtest.
        
        Returns:
            dict: Backtest results
        """
        # Reset previous results
        self.results = None
        
        # Notify components that backtest is starting
        self._emit_event(EventType.BACKTEST_START, {
            'timestamp': datetime.now()
        })
        
        # Process all bars
        data_handler = self.components['data_handler']
        while data_handler.update_bars():
            pass
            
        # Close open positions
        self._close_positions()
        
        # Collect results
        self.results = self._collect_results()
        
        # Notify components that backtest is complete
        self._emit_event(EventType.BACKTEST_END, {
            'timestamp': datetime.now(),
            'results': self.results
        })
        
        return self.results
```

### 7. Passthrough Broker

The PassthroughBroker provides a simplified execution mode for strategy testing without the complexity of full market simulation. It executes orders immediately without slippage or commission.

```python
class PassthroughBroker(BrokerBase):
    """
    Passthrough broker implementation.
    
    Executes orders immediately without slippage or commission.
    Useful for strategy development and testing.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
    
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.ORDER, self.on_order)
    
    def on_order(self, event):
        """
        Process order event.
        
        Args:
            event: Order event
        """
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

## Execution Mode Framework

The ADMF-Trader system supports multiple execution modes, each with specific threading and concurrency requirements.

### Execution Modes

```python
class ExecutionMode(Enum):
    """Execution mode options for the ADMF-Trader system."""
    BACKTEST_SINGLE = auto()  # Single-threaded backtesting (fast, no thread safety needed)
    BACKTEST_PARALLEL = auto()  # Multi-threaded backtest components (thread safety required)
    OPTIMIZATION = auto()      # Parallel optimization (multiple backtest instances, thread safety required)
    LIVE_TRADING = auto()      # Live market trading (multi-threaded, thread safety required)
    PAPER_TRADING = auto()     # Simulated live trading (multi-threaded, thread safety required)
    REPLAY = auto()            # Event replay mode (configurable threading model)
```

### Thread Models

Each execution mode has an associated thread model that determines its concurrency requirements:

```python
class ThreadModel(Enum):
    """Thread model options for execution contexts."""
    SINGLE_THREADED = auto()    # All operations in a single thread
    MULTI_THREADED = auto()     # Operations can occur across multiple threads
    PROCESS_PARALLEL = auto()   # Parallel processes with internal thread management
    ASYNC_SINGLE = auto()       # Single event loop, asynchronous processing
    ASYNC_MULTI = auto()        # Multiple event loops, asynchronous processing
    MIXED = auto()              # Mixed model with custom thread management
```

### Execution Context

The ExecutionContext encapsulates the execution environment and provides a framework for components to adapt their behavior based on the execution mode:

```python
class ExecutionContext:
    """
    Execution context for the ADMF-Trader system.
    
    This class encapsulates the execution mode, thread model, and related
    configuration for a particular execution run.
    """
    
    _current_context = threading.local()  # Thread-local storage for current context
    
    @classmethod
    def get_current(cls) -> Optional['ExecutionContext']:
        """Get current execution context for this thread."""
        return getattr(cls._current_context, 'context', None)
        
    @classmethod
    def set_current(cls, context: Optional['ExecutionContext']) -> None:
        """Set current execution context for this thread."""
        cls._current_context.context = context
    
    def __init__(self, 
                 name: str,
                 execution_mode: ExecutionMode,
                 thread_model: Optional[ThreadModel] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize execution context.
        
        Args:
            name: Context name
            execution_mode: Execution mode for this context
            thread_model: Thread model for this context (defaults based on execution mode)
            config: Additional configuration
        """
        self.name = name
        self.execution_mode = execution_mode
        
        # Use default thread model if not specified
        self.thread_model = thread_model or DEFAULT_THREAD_MODELS.get(
            execution_mode, ThreadModel.SINGLE_THREADED)
            
        self.config = config or {}
        self._metadata = {}
        self._active_threads = set()
        self._creation_thread = threading.current_thread()
        self._previous_context = None
```

## Implementation Best Practices

### 1. Order Management

For robust order management, follow these practices:

#### Order States

Define clear order lifecycle states:

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

#### Order Tracking with History

Track order details and maintain a complete status history:

```python
def update_order_status(self, order_id, status, details=None):
    """Update order status with history tracking."""
    if order_id not in self.active_orders:
        return False
        
    order = self.active_orders[order_id]
    
    # Add to status history
    if 'status_history' not in order:
        order['status_history'] = []
        
    order['status_history'].append({
        'timestamp': datetime.now(),
        'old_status': order['status'],
        'new_status': status,
        'details': details
    })
    
    # Update current status
    order['status'] = status
    
    return True
```

#### Thread-Safe Operations

Ensure thread safety in multi-threaded execution modes:

```python
def get_order(self, order_id):
    """Thread-safe order retrieval."""
    # Check active orders
    if order_id in self.active_orders:
        return copy.deepcopy(self.active_orders[order_id])
        
    # Check completed orders
    if order_id in self.completed_orders:
        return copy.deepcopy(self.completed_orders[order_id])
        
    return None
```

### 2. Realistic Execution Simulation

For realistic order execution, consider:

#### OHLC Bar Model

Implement realistic execution prices within the bar range:

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
                
    return None  # Not executed
```

#### Volume Constraints

Limit order sizes based on volume for realistic execution:

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
        
        # Adjust sign based on direction
        if order['direction'] == 'SELL':
            limited_quantity = -limited_quantity
            
        # Create partial fill
        order['partial_fill'] = True
        order['original_quantity'] = order['quantity']
        order['quantity'] = limited_quantity
        
        return True
        
    return False
```

### 3. Performance Optimization

For optimal performance, consider:

#### Batched Processing

Process orders in batches for improved efficiency:

```python
def process_bar(self, bar):
    """Process a bar with batched order handling."""
    symbol = bar['symbol']
    
    # Update price
    self.latest_prices[symbol] = bar['close']
    
    # Collect orders for this symbol
    symbol_orders = []
    for order_id, order in self.pending_orders.items():
        if order['symbol'] == symbol:
            symbol_orders.append(order)
            
    # Process collected orders in batch
    self._process_orders_batch(symbol_orders, bar)
```

#### Optimized Data Structures

Use efficient collections for faster lookups:

```python
# Use symbol-indexed order collections for faster lookup
self.orders_by_symbol = defaultdict(list)

def add_order(self, order):
    """Add order with optimized indexing."""
    order_id = order['order_id']
    symbol = order['symbol']
    
    # Add to main collection
    self.active_orders[order_id] = order
    
    # Add to symbol index
    self.orders_by_symbol[symbol].append(order_id)
```

### 4. Thread Management

Best practices for thread management in different execution modes:

#### Thread Safety Requirements

| Thread Model | Thread Safety Requirements |
|--------------|----------------------------|
| SINGLE_THREADED | No thread safety required; all operations occur in the same thread |
| MULTI_THREADED | Full thread safety required; any operation may occur from any thread |
| PROCESS_PARALLEL | Thread safety within processes; inter-process communication through safe channels |
| MIXED | Component-specific thread safety based on documented requirements |

#### Thread Pool Management

Use appropriate thread pool configurations for different execution modes:

```python
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
        elif self.context.is_live:
            # More threads for live trading
            self._thread_pools['market_data'] = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, 
                thread_name_prefix='market_data_worker'
            )
            self._thread_pools['order_processing'] = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, 
                thread_name_prefix='order_worker'
            )
```

## Implementation Timeline

The execution module will be implemented in phases:

### Phase 1: Core Components (Weeks 1-2)

1. **Week 1**: Basic Interfaces and Models
   - Implement BrokerBase interface
   - Implement OrderManager
   - Implement OrderValidator
   - Define order models and states

2. **Week 2**: Simulation Core
   - Implement SimulatedBroker
   - Implement basic slippage models
   - Implement basic commission models
   - Create unit tests for core components

### Phase 2: Backtesting Framework (Weeks 3-4)

3. **Week 3**: Backtest Coordination
   - Implement BacktestCoordinator
   - Implement result collection
   - Implement performance metrics
   - Create integration tests for backtesting

4. **Week 4**: Advanced Simulation
   - Implement advanced slippage models
   - Implement advanced commission models
   - Implement PassthroughBroker
   - Enhance test coverage for edge cases

### Phase 3: Execution Modes (Weeks 5-6)

5. **Week 5**: Execution Context Framework
   - Implement ExecutionMode and ThreadModel enums
   - Implement ExecutionContext class
   - Update components to use execution context
   - Create tests for different execution modes

6. **Week 6**: Thread Management
   - Implement ThreadPoolManager
   - Implement thread safety mechanisms
   - Update components for thread-safety
   - Test concurrency behavior

### Phase 4: Live Trading (Weeks 7-8)

7. **Week 7**: Live Broker Interface
   - Define LiveBroker interface
   - Implement broker adapter pattern
   - Create mock live broker for testing
   - Test live trading simulation

8. **Week 8**: Finalization
   - Performance optimization
   - Comprehensive documentation
   - End-to-end testing
   - System integration

## Integration with Other Modules

### Risk Module Integration

The Execution module receives orders from the Risk module and provides fill information back:

1. **Order Flow**: Risk Manager → Order Manager → Broker
2. **Fill Flow**: Broker → Order Manager → Risk Manager

```python
# Example risk module integration
def on_signal(self, event):
    """Process signal from Strategy."""
    # Apply risk rules and generate order
    order_event = self.risk_manager.process_signal(event)
    
    # Publish order event to execution module
    if order_event:
        self.event_bus.publish(order_event)
        
def on_fill(self, event):
    """Process fill from Execution module."""
    # Update portfolio with fill information
    self.portfolio.update_position(event.get_data())
```

### Data Module Integration

The Execution module consumes market data from the Data module for realistic execution simulation:

```python
# Example data integration in SimulatedBroker
def on_bar(self, event):
    """Process bar event from Data module."""
    # Extract bar data
    bar_data = event.get_data()
    symbol = bar_data['symbol']
    
    # Update latest price
    self.latest_prices[symbol] = bar_data['close']
    
    # Check pending orders against current bar
    self._check_pending_orders(bar_data)
```

### Strategy Module Integration

The Execution module indirectly interacts with the Strategy module through the Risk module:

```
Strategy → Signal → Risk Manager → Order → Execution → Fill → Portfolio → Strategy
```

## Configuration Examples

### 1. Simple Backtest Configuration

```yaml
execution:
  mode: BACKTEST_SINGLE
  
  order_manager:
    max_completed_orders: 10000
    
  broker:
    class: SimulatedBroker
    slippage:
      model: fixed
      price_impact: 0.0005  # 0.05% slippage
      
    commission:
      model: fixed
      cost_per_trade: 5.0  # $5 per trade
```

### 2. Advanced Backtest Configuration

```yaml
execution:
  mode: BACKTEST_PARALLEL
  
  order_manager:
    max_completed_orders: 50000
    
  broker:
    class: SimulatedBroker
    slippage:
      model: volume
      base_percentage: 0.0001
      volume_percentage: 0.1
      default_volume: 100000
      
    commission:
      model: tiered
      tiers:
        - threshold: 0
          type: percentage
          rate: 0.0025
          min_commission: 1.0
        - threshold: 10000
          type: percentage
          rate: 0.0015
          min_commission: 1.0
        - threshold: 100000
          type: percentage
          rate: 0.0010
          min_commission: 1.0
```

### 3. Live Trading Configuration

```yaml
execution:
  mode: LIVE_TRADING
  
  order_manager:
    max_completed_orders: 10000
    
  broker:
    class: LiveBroker
    adapter: InteractiveBrokers
    connection:
      host: localhost
      port: 7496
      client_id: 1
    
    settings:
      auto_reconnect: true
      max_retry: 3
      timeout: 30000
```

## Conclusion

The Execution module is a critical component of the ADMF-Trader system, providing realistic market simulation for backtesting and a pathway to live trading. By implementing the components and best practices outlined in this guide, the system will achieve robust order management, realistic execution simulation, and efficient performance in both backtesting and live trading scenarios.

The modular design with clear interfaces allows for easy extension and customization, while the execution mode framework provides adaptability across different operating contexts. The phased implementation timeline ensures a systematic approach to building the module, with comprehensive testing at each stage.