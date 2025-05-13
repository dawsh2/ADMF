# Execution Module Implementation Guide

## Overview

The Execution module is responsible for order processing, market simulation, and backtest coordination in the ADMF-Trader system. It receives orders from the Risk module, simulates market execution with realistic slippage and commission models, and generates fill events when orders are executed.

## Key Components

1. **Order Manager**
   - Manages order lifecycle and tracking
   - Validates and processes incoming orders
   - Tracks order status and execution details

2. **Broker**
   - Simulates market execution
   - Applies slippage and commission models
   - Generates fill events

3. **Slippage Models**
   - Realistic market impact simulation
   - Fixed, percentage, and volume-based models
   - Price limit handling

4. **Commission Models**
   - Trading cost simulation
   - Fixed, percentage, and tiered models
   - Per-share and per-trade fees

5. **Backtest Coordinator**
   - Orchestrates the backtesting process
   - Manages system component lifecycle
   - Collects and analyzes backtest results

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

## Component Specifications

### 1. Order Manager

The OrderManager class handles order tracking and lifecycle management:

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
            
            # Add fill information
            if 'fills' not in order:
                order['fills'] = []
                
            order['fills'].append(fill_data.copy())
            
            # Move to completed orders
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]
            
            # Prune completed orders if needed
            self._prune_completed_orders()
            
    def _prune_completed_orders(self):
        """Prune completed orders if too many."""
        if len(self.completed_orders) > self.max_completed_orders:
            # Get oldest orders to remove
            orders = list(self.completed_orders.items())
            orders.sort(key=lambda x: x[1]['received_time'])
            
            # Remove oldest orders
            to_remove = len(orders) - self.max_completed_orders
            for i in range(to_remove):
                del self.completed_orders[orders[i][0]]
                
    def get_order(self, order_id):
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            dict: Order data or None
        """
        # Check active orders
        if order_id in self.active_orders:
            return self.active_orders[order_id].copy()
            
        # Check completed orders
        if order_id in self.completed_orders:
            return self.completed_orders[order_id].copy()
            
        return None
        
    def get_active_orders(self, symbol=None):
        """
        Get active orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            list: List of active orders
        """
        if symbol:
            return [order.copy() for order in self.active_orders.values()
                    if order['symbol'] == symbol]
        else:
            return [order.copy() for order in self.active_orders.values()]
            
    def cancel_order(self, order_id):
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID
            
        Returns:
            bool: Success or failure
        """
        # Check if order exists
        if order_id not in self.active_orders:
            return False
            
        # Get order
        order = self.active_orders[order_id]
        
        # Update status
        order['status'] = 'CANCELLED'
        order['cancelled_time'] = datetime.now()
        
        # Move to completed orders
        self.completed_orders[order_id] = order
        del self.active_orders[order_id]
        
        # Notify broker
        self.broker.cancel_order(order_id)
        
        return True
        
    def reset(self):
        """Reset order manager state."""
        super().reset()
        
        # Clear order collections
        self.active_orders = ThreadSafeDict()
        self.completed_orders = ThreadSafeDict()
```

### 2. Order Validator

The OrderValidator handles order validation:

```python
class OrderValidator:
    """
    Order validation utility.
    
    Static methods for validating order data.
    """
    
    @staticmethod
    def validate_order(order_data):
        """
        Validate order data.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            bool: Whether order is valid
        """
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
            
        # Check additional fields based on order type
        if order_data['order_type'] == 'LIMIT' and 'limit_price' not in order_data:
            return False
            
        if order_data['order_type'] in ['STOP', 'STOP_LIMIT'] and 'stop_price' not in order_data:
            return False
            
        if order_data['order_type'] == 'STOP_LIMIT' and 'limit_price' not in order_data:
            return False
            
        return True
```

### 3. Simulated Broker

The SimulatedBroker class simulates market execution:

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
        
    def on_bar(self, event):
        """
        Process bar event.
        
        Args:
            event: Bar event
        """
        # Extract bar data
        bar_data = event.get_data()
        symbol = bar_data['symbol']
        close_price = bar_data['close']
        
        # Update latest price
        self.latest_prices[symbol] = close_price
        
        # Check pending orders
        self._check_pending_orders(bar_data)
        
    def _check_pending_orders(self, bar_data):
        """
        Check pending orders against current bar data.
        
        Args:
            bar_data: Bar data dictionary
        """
        symbol = bar_data['symbol']
        open_price = bar_data['open']
        high_price = bar_data['high']
        low_price = bar_data['low']
        close_price = bar_data['close']
        
        # Create copy of pending orders to avoid mutation during iteration
        pending_order_ids = list(self.pending_orders.keys())
        
        for order_id in pending_order_ids:
            # Check if order still exists
            if order_id not in self.pending_orders:
                continue
                
            order = self.pending_orders[order_id]
            
            # Skip orders for other symbols
            if order['symbol'] != symbol:
                continue
                
            # Check order conditions
            order_type = order['order_type']
            direction = order['direction']
            
            if order_type == 'MARKET':
                # Market orders execute at current price
                self._execute_order(order, close_price)
                del self.pending_orders[order_id]
            elif order_type == 'LIMIT':
                # Check if limit price was reached during the bar
                limit_price = order['limit_price']
                
                if (direction == 'BUY' and low_price <= limit_price) or \
                   (direction == 'SELL' and high_price >= limit_price):
                    # Limit price reached during the bar
                    # Use limit price for execution
                    execution_price = limit_price
                    self._execute_order(order, execution_price)
                    del self.pending_orders[order_id]
            elif order_type == 'STOP':
                # Check if stop price was reached during the bar
                stop_price = order['stop_price']
                
                if (direction == 'BUY' and high_price >= stop_price) or \
                   (direction == 'SELL' and low_price <= stop_price):
                    # Stop price reached during the bar
                    # Use stop price for execution
                    execution_price = stop_price
                    self._execute_order(order, execution_price)
                    del self.pending_orders[order_id]
            elif order_type == 'STOP_LIMIT':
                # Check if stop price was reached during the bar
                stop_price = order['stop_price']
                limit_price = order['limit_price']
                
                if (direction == 'BUY' and high_price >= stop_price) or \
                   (direction == 'SELL' and low_price <= stop_price):
                    # Stop price reached, convert to limit order
                    order['order_type'] = 'LIMIT'
                    order['limit_price'] = limit_price
                    # Keep in pending orders as a limit order
                    
                    # Check if limit price was also reached
                    if (direction == 'BUY' and low_price <= limit_price) or \
                       (direction == 'SELL' and high_price >= limit_price):
                        # Limit price also reached
                        execution_price = limit_price
                        self._execute_order(order, execution_price)
                        del self.pending_orders[order_id]
                    
    def _execute_order(self, order_data, price):
        """
        Execute an order.
        
        Args:
            order_data: Order data dictionary
            price: Execution price
            
        Returns:
            bool: Success or failure
        """
        # Extract order details
        order_id = order_data['order_id']
        symbol = order_data['symbol']
        quantity = order_data['quantity']
        direction = order_data['direction']
        
        # Apply slippage
        execution_price = self.slippage_model.apply_slippage(price, direction, quantity)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(execution_price, quantity)
        
        # Create fill data
        fill_data = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': execution_price,
            'direction': direction,
            'commission': commission,
            'timestamp': datetime.now()
        }
        
        # Emit fill event
        self._emit_fill(fill_data)
        
        return True
        
    def _emit_fill(self, fill_data):
        """
        Emit fill event.
        
        Args:
            fill_data: Fill data dictionary
            
        Returns:
            bool: Success or failure
        """
        # Create and publish event
        event = Event(EventType.FILL, fill_data)
        return self.event_bus.publish(event)
        
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            bool: Success or failure
        """
        # Check if order exists
        if order_id not in self.pending_orders:
            return False
            
        # Remove from pending orders
        del self.pending_orders[order_id]
        
        return True
        
    def reset(self):
        """Reset broker state."""
        super().reset()
        
        # Clear collections
        self.pending_orders = ThreadSafeDict()
        self.latest_prices = ThreadSafeDict()
```

### 4. Slippage Models

Slippage model implementations:

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

```python
class PercentageSlippageModel(SlippageModel):
    """
    Percentage slippage model.
    
    Adds a percentage of price based on quantity.
    """
    
    def apply_slippage(self, price, direction, quantity):
        """
        Apply percentage slippage to price.
        
        Args:
            price: Base price
            direction: Order direction ('BUY' or 'SELL')
            quantity: Order quantity
            
        Returns:
            float: Adjusted price
        """
        # Get base percentage
        base_percentage = self.parameters.get('percentage', 0.0)
        
        # Get volume factor
        volume_factor = self.parameters.get('volume_factor', 0.0)
        
        # Calculate slippage percentage based on quantity
        slippage_percentage = base_percentage + (abs(quantity) * volume_factor)
        
        # Apply slippage based on direction
        if direction == 'BUY':
            return price * (1.0 + slippage_percentage)
        else:  # SELL
            return price * (1.0 - slippage_percentage)
```

```python
class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model.
    
    Adds slippage based on quantity relative to average volume.
    """
    
    def apply_slippage(self, price, direction, quantity, volume=None):
        """
        Apply volume-based slippage to price.
        
        Args:
            price: Base price
            direction: Order direction ('BUY' or 'SELL')
            quantity: Order quantity
            volume: Bar volume (optional)
            
        Returns:
            float: Adjusted price
        """
        # Get parameters
        base_percentage = self.parameters.get('base_percentage', 0.0)
        volume_percentage = self.parameters.get('volume_percentage', 0.0)
        
        # Default volume if not provided
        if volume is None:
            volume = self.parameters.get('default_volume', 100000)
            
        if volume <= 0:
            volume = 1  # Avoid division by zero
            
        # Calculate volume impact
        volume_ratio = abs(quantity) / volume
        slippage_percentage = base_percentage + (volume_ratio * volume_percentage)
        
        # Apply slippage based on direction
        if direction == 'BUY':
            return price * (1.0 + slippage_percentage)
        else:  # SELL
            return price * (1.0 - slippage_percentage)
```

### 5. Commission Models

Commission model implementations:

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

```python
class FixedCommissionModel(CommissionModel):
    """
    Fixed commission model.
    
    Charges a fixed amount per trade.
    """
    
    def calculate_commission(self, price, quantity):
        """
        Calculate fixed commission.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            
        Returns:
            float: Commission amount
        """
        # Get fixed cost per trade
        cost_per_trade = self.parameters.get('cost_per_trade', 0.0)
        
        return cost_per_trade
```

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

```python
class TieredCommissionModel(CommissionModel):
    """
    Tiered commission model.
    
    Charges different rates based on trade size.
    """
    
    def calculate_commission(self, price, quantity):
        """
        Calculate tiered commission.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            
        Returns:
            float: Commission amount
        """
        # Get tiers
        tiers = self.parameters.get('tiers', [])
        
        # Get share-based flag
        share_based = self.parameters.get('share_based', False)
        
        # Calculate trade value
        trade_value = abs(price * quantity)
        shares = abs(quantity)
        
        # Default to highest tier if no tiers defined
        if not tiers:
            return 0.0
            
        # Get tier based on shares or value
        tier_value = shares if share_based else trade_value
        
        # Find applicable tier
        applicable_tier = None
        for tier in tiers:
            threshold = tier.get('threshold', 0)
            if tier_value >= threshold:
                applicable_tier = tier
            else:
                break
                
        if not applicable_tier:
            # Use first tier as default
            applicable_tier = tiers[0]
            
        # Calculate commission
        if applicable_tier.get('type') == 'percentage':
            percentage = applicable_tier.get('rate', 0.0)
            commission = trade_value * percentage
        else:
            per_share = applicable_tier.get('rate', 0.0)
            commission = shares * per_share
            
        # Apply minimum
        min_commission = applicable_tier.get('min_commission', 0.0)
        return max(commission, min_commission)
```

### 6. Backtest Coordinator

The BacktestCoordinator orchestrates the backtesting process:

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
        
    def _close_positions(self):
        """Close all open positions at the end of the backtest."""
        # Get portfolio
        portfolio = self.components['portfolio']
        
        # Get open positions
        positions = portfolio.get_open_positions()
        
        # Skip if no open positions
        if not positions:
            return
            
        # Get broker and order manager
        broker = self.components.get('broker')
        
        # Close each position
        for symbol, position in positions.items():
            # Skip zero positions
            if position.quantity == 0:
                continue
                
            # Create order to close position
            order_data = {
                'order_id': f"close_{symbol}_{datetime.now().timestamp()}",
                'symbol': symbol,
                'quantity': -position.quantity,  # Reverse position
                'direction': 'BUY' if position.quantity < 0 else 'SELL',
                'order_type': 'MARKET',
                'price': position.market_price,
                'timestamp': datetime.now(),
                'source': 'backtest_close'
            }
            
            # Execute directly through broker
            if broker:
                broker.process_order(order_data)
                
    def _collect_results(self):
        """
        Collect backtest results.
        
        Returns:
            dict: Backtest results
        """
        # Get components
        portfolio = self.components['portfolio']
        
        # Collect equity curve
        equity_curve = portfolio.equity_curve
        
        # Collect trades
        trades = portfolio.trades
        
        # Calculate statistics
        statistics = self._calculate_statistics(equity_curve, trades)
        
        # Create results dictionary
        results = {
            'equity_curve': equity_curve,
            'trades': trades,
            'statistics': statistics,
            'parameters': self._collect_parameters()
        }
        
        return results
        
    def _calculate_statistics(self, equity_curve, trades):
        """
        Calculate performance statistics.
        
        Args:
            equity_curve: List of equity points
            trades: List of trades
            
        Returns:
            dict: Performance statistics
        """
        # Skip if no equity data
        if not equity_curve:
            return {}
            
        # Extract values
        equity_values = [point['portfolio_value'] for point in equity_curve]
        timestamps = [point['timestamp'] for point in equity_curve]
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] / equity_values[i-1]) - 1
            returns.append(ret)
            
        # Calculate basic metrics
        start_equity = equity_values[0]
        end_equity = equity_values[-1]
        total_return = (end_equity / start_equity) - 1
        
        # Calculate trade metrics
        win_count = sum(1 for trade in trades if trade.get('realized_pnl', 0) > 0)
        loss_count = sum(1 for trade in trades if trade.get('realized_pnl', 0) < 0)
        total_trades = len(trades)
        
        # Avoid division by zero
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate drawdown
        max_drawdown, max_drawdown_pct = self._calculate_drawdown(equity_values)
        
        # Create statistics dictionary
        statistics = {
            'start_date': timestamps[0],
            'end_date': timestamps[-1],
            'start_equity': start_equity,
            'end_equity': end_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': self._calculate_annualized_return(total_return, timestamps),
            'volatility': np.std(returns) * np.sqrt(252) if returns else 0,  # Annualized
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct * 100,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate
        }
        
        # Calculate Sharpe ratio
        if returns:
            avg_return = np.mean(returns)
            std_dev = np.std(returns)
            if std_dev > 0:
                statistics['sharpe_ratio'] = avg_return / std_dev * np.sqrt(252)  # Annualized
                
        return statistics
        
    def _calculate_drawdown(self, equity_values):
        """
        Calculate maximum drawdown.
        
        Args:
            equity_values: List of equity values
            
        Returns:
            tuple: (max_drawdown, max_drawdown_pct)
        """
        # Initialize variables
        peak = equity_values[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        # Calculate drawdown
        for value in equity_values:
            if value > peak:
                peak = value
                
            drawdown = peak - value
            drawdown_pct = drawdown / peak if peak > 0 else 0
            
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
                
        return max_drawdown, max_drawdown_pct
        
    def _calculate_annualized_return(self, total_return, timestamps):
        """
        Calculate annualized return.
        
        Args:
            total_return: Total return
            timestamps: List of timestamps
            
        Returns:
            float: Annualized return
        """
        # Calculate years
        start_date = timestamps[0]
        end_date = timestamps[-1]
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Calculate years
        if hasattr(start_date, 'days') and hasattr(end_date, 'days'):
            days = (end_date - start_date).days
        else:
            days = (end_date - start_date) / (24 * 60 * 60)
            
        years = days / 365.0
        
        # Avoid division by zero
        if years <= 0:
            return 0
            
        # Calculate annualized return
        return (1 + total_return) ** (1 / years) - 1
        
    def _collect_parameters(self):
        """
        Collect parameters from all components.
        
        Returns:
            dict: Component parameters
        """
        parameters = {}
        
        for name, component in self.components.items():
            if hasattr(component, 'parameters'):
                parameters[name] = component.parameters
                
        return parameters
        
    def _emit_event(self, event_type, data):
        """
        Emit an event.
        
        Args:
            event_type: Event type
            data: Event data
            
        Returns:
            bool: Success or failure
        """
        event_bus = self.components['event_bus']
        event = Event(event_type, data)
        return event_bus.publish(event)
```

## Passthrough Execution

The Execution module supports a "passthrough" mode to facilitate strategy testing without the complexity of full execution simulation:

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
    
    def _emit_fill(self, fill_data):
        """Emit fill event."""
        event = Event(EventType.FILL, fill_data)
        return self.event_bus.publish(event)
    
    def cancel_order(self, order_id):
        """Cancel an order (no-op in passthrough)."""
        return True
    
    def reset(self):
        """Reset broker state."""
        super().reset()
```

The passthrough broker can be enabled through configuration:

```python
# Config example for passthrough execution
execution_config = {
    'broker': {
        'class': 'PassthroughBroker',
        'parameters': {
            'enabled': True
        }
    }
}
```

## Best Practices

### Order Management

Follow these best practices for order management:

1. **Order States**: Track order lifecycle states
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

2. **Order Tracking**: Track order details and history
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

3. **Thread Safety**: Ensure thread-safe operations
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

### Slippage Modeling

Use these patterns for realistic slippage modeling:

```python
# Implement slippage model based on order size and volatility
def calculate_slippage(self, price, direction, quantity, volatility=None):
    """Calculate price slippage based on order size and volatility."""
    # Get base slippage
    base_slippage = self.parameters.get('base_slippage', 0.0001)
    
    # Get volatility factor
    volatility_factor = self.parameters.get('volatility_factor', 0.5)
    
    # Get size factor
    size_factor = self.parameters.get('size_factor', 0.0001)
    
    # Use default volatility if not provided
    if volatility is None:
        volatility = self.parameters.get('default_volatility', 0.01)
        
    # Calculate slippage
    slippage = base_slippage + (volatility * volatility_factor) + (abs(quantity) * size_factor)
    
    # Apply direction
    if direction == 'BUY':
        slippage_factor = 1.0 + slippage
    else:  # SELL
        slippage_factor = 1.0 - slippage
        
    return price * slippage_factor
```

### Backtest Coordination

Use this pattern for backtest orchestration:

```python
def run_backtest(self):
    """Run a complete backtest."""
    try:
        # 1. Start all components
        self._start_components()
        
        # 2. Notify backtest start
        self._emit_event(EventType.BACKTEST_START, {'timestamp': datetime.now()})
        
        # 3. Process all bars
        data_handler = self.components['data_handler']
        while data_handler.update_bars():
            pass
            
        # 4. Close all positions
        self._close_positions()
        
        # 5. Collect results
        results = self._collect_results()
        
        # 6. Notify backtest end
        self._emit_event(EventType.BACKTEST_END, {
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    finally:
        # Always stop components
        self._stop_components()
```

## Implementation Considerations

### 1. Order Execution

For realistic order execution, consider:

1. **OHLC Bar Model**: Implement realistic execution prices
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

2. **Volume Constraints**: Limit order sizes based on volume
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

3. **Realistic Fill Prices**: Implement more sophisticated fill models
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
           
       # Use different price models based on order type
       if order_type == 'MARKET':
           # Use VWAP for market orders with slippage
           base_price = vwap
       elif order_type == 'LIMIT':
           # Use limit price for limit orders
           limit_price = order['limit_price']
           
           if direction == 'BUY':
               # For buy limit, use the lower of VWAP and limit price
               base_price = min(vwap, limit_price)
           else:
               # For sell limit, use the higher of VWAP and limit price
               base_price = max(vwap, limit_price)
       else:
           # Default to VWAP
           base_price = vwap
           
       # Apply slippage to base price
       return self.slippage_model.apply_slippage(base_price, direction, order['quantity'])
   ```

### 2. Performance Optimization

For better performance, consider:

1. **Batched Processing**: Process orders in batches
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

2. **Optimized Data Structures**: Use efficient collections
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

3. **Lazy Calculation**: Calculate metrics on demand
   ```python
   def get_performance_metrics(self):
       """Calculate performance metrics on demand."""
       # Check if we have cached metrics
       if hasattr(self, '_cached_metrics') and self._metrics_cache_valid:
           return self._cached_metrics
           
       # Calculate metrics
       metrics = self._calculate_metrics()
       
       # Cache results
       self._cached_metrics = metrics
       self._metrics_cache_valid = True
       
       return metrics
   ```

### 3. Result Analysis

For comprehensive result analysis, consider:

1. **Trade Analysis**: Calculate detailed trade metrics
   ```python
   def analyze_trades(self, trades):
       """Analyze trades with detailed metrics."""
       if not trades:
           return {}
           
       # Calculate basic metrics
       win_count = sum(1 for t in trades if t['realized_pnl'] > 0)
       loss_count = sum(1 for t in trades if t['realized_pnl'] < 0)
       
       # Calculate advanced metrics
       holding_times = []
       for trade in trades:
           if 'entry_time' in trade and 'exit_time' in trade:
               holding_time = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
               holding_times.append(holding_time)
               
       # Calculate trade size distribution
       trade_sizes = [abs(t['quantity']) for t in trades]
       
       # Calculate profit distribution
       profits = [t['realized_pnl'] for t in trades if t['realized_pnl'] > 0]
       losses = [t['realized_pnl'] for t in trades if t['realized_pnl'] < 0]
       
       # Return comprehensive metrics
       return {
           'win_rate': win_count / len(trades) if trades else 0,
           'profit_factor': sum(profits) / abs(sum(losses)) if sum(losses) else float('inf'),
           'avg_win': sum(profits) / len(profits) if profits else 0,
           'avg_loss': sum(losses) / len(losses) if losses else 0,
           'avg_holding_time': sum(holding_times) / len(holding_times) if holding_times else 0,
           'max_win': max(profits) if profits else 0,
           'max_loss': min(losses) if losses else 0,
           'avg_trade_size': sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0
       }
   ```

2. **Equity Curve Analysis**: Calculate equity curve metrics
   ```python
   def analyze_equity_curve(self, equity_curve):
       """Analyze equity curve with advanced metrics."""
       if not equity_curve:
           return {}
           
       # Extract equity values
       equity_values = [point['portfolio_value'] for point in equity_curve]
       
       # Calculate returns
       returns = []
       for i in range(1, len(equity_values)):
           ret = (equity_values[i] / equity_values[i-1]) - 1
           returns.append(ret)
           
       # Calculate metrics
       pos_returns = [r for r in returns if r > 0]
       neg_returns = [r for r in returns if r < 0]
       
       # Return detailed metrics
       return {
           'total_return': (equity_values[-1] / equity_values[0]) - 1,
           'volatility': np.std(returns) * np.sqrt(252),  # Annualized
           'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
           'sortino_ratio': np.mean(returns) / np.std(neg_returns) * np.sqrt(252) if np.std(neg_returns) > 0 else 0,
           'max_drawdown': self._calculate_max_drawdown(equity_values),
           'profit_to_drawdown': abs((equity_values[-1] / equity_values[0] - 1) / self._calculate_max_drawdown(equity_values)) if self._calculate_max_drawdown(equity_values) > 0 else 0,
           'winning_days': len(pos_returns) / len(returns) if returns else 0
       }
   ```

By following these guidelines, you'll create a robust Execution module that provides realistic order handling, market simulation, and comprehensive backtest analysis for the ADMF-Trader system.