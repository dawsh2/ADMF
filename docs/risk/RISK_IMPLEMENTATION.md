# Risk Module Implementation Guide

## Overview

The Risk module is responsible for position sizing, risk control, and portfolio management in the ADMF-Trader system. It receives trading signals from strategies, applies risk limits and position sizing algorithms, and converts them to orders for execution.

## Key Components

1. **Risk Manager**
   - Converts signals to orders with proper position sizing
   - Applies risk limits and constraints
   - Rejects signals that violate risk rules

2. **Portfolio Manager**
   - Tracks positions and equity
   - Updates positions based on fill events
   - Calculates portfolio statistics

3. **Position Sizing**
   - Implements various position sizing algorithms
   - Adapts position sizes to market conditions
   - Optimizes capital allocation

4. **Risk Limits**
   - Enforces trading constraints
   - Implements drawdown control
   - Manages exposure and concentration risks

5. **Position Management**
   - Tracks individual security positions
   - Calculates P&L and other position metrics
   - Manages position lifecycle

## Implementation Structure

```
src/risk/
├── __init__.py
├── interfaces/
│   ├── __init__.py
│   ├── risk_manager.py        # Risk manager interface
│   └── position_sizer.py      # Position sizer interface
├── managers/
│   ├── __init__.py
│   ├── risk_manager.py        # Risk manager implementation
│   └── portfolio.py           # Portfolio manager
├── models/
│   ├── __init__.py
│   └── position.py            # Position model
├── sizers/
│   ├── __init__.py
│   ├── fixed_sizer.py         # Fixed position sizing
│   ├── percent_equity_sizer.py # Percentage of equity sizing
│   └── percent_risk_sizer.py  # Risk-based position sizing
└── limits/
    ├── __init__.py
    ├── position_limit.py      # Maximum position size
    ├── exposure_limit.py      # Maximum exposure
    └── drawdown_limit.py      # Drawdown control
```

## Component Specifications

### 1. Risk Manager

The RiskManager converts signals to orders with appropriate risk control:

```python
class RiskManager(Component):
    """
    Risk manager implementation.
    
    Converts signals to orders with position sizing and risk limits.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize order ID counter
        self.order_id_counter = ThreadSafeCounter(1)
        
        # Initialize position sizers
        self.position_sizers = {}
        
        # Initialize risk limits
        self.risk_limits = []
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Get portfolio manager
        self.portfolio = self._get_dependency(context, 'portfolio', required=True)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.SIGNAL, self.on_signal)
        
        # Create position sizers
        self._create_position_sizers()
        
        # Create risk limits
        self._create_risk_limits()
        
    def _create_position_sizers(self):
        """Create position sizers from configuration."""
        sizer_configs = self.parameters.get('position_sizers', {})
        
        for symbol, config in sizer_configs.items():
            sizer_type = config.get('type', 'fixed')
            
            if sizer_type == 'fixed':
                self.position_sizers[symbol] = FixedSizer(config)
            elif sizer_type == 'percent_equity':
                self.position_sizers[symbol] = PercentEquitySizer(config)
            elif sizer_type == 'percent_risk':
                self.position_sizers[symbol] = PercentRiskSizer(config)
                
        # Create default sizer for symbols not explicitly configured
        default_config = self.parameters.get('default_position_sizer', {'type': 'fixed', 'size': 100})
        default_sizer_type = default_config.get('type', 'fixed')
        
        if default_sizer_type == 'fixed':
            self.default_sizer = FixedSizer(default_config)
        elif default_sizer_type == 'percent_equity':
            self.default_sizer = PercentEquitySizer(default_config)
        elif default_sizer_type == 'percent_risk':
            self.default_sizer = PercentRiskSizer(default_config)
            
    def _create_risk_limits(self):
        """Create risk limits from configuration."""
        limit_configs = self.parameters.get('risk_limits', [])
        
        for config in limit_configs:
            limit_type = config.get('type')
            
            if limit_type == 'position':
                self.risk_limits.append(PositionLimit(config))
            elif limit_type == 'exposure':
                self.risk_limits.append(ExposureLimit(config))
            elif limit_type == 'drawdown':
                self.risk_limits.append(DrawdownLimit(config))
                
    def on_signal(self, event):
        """
        Process signal event and generate orders.
        
        Args:
            event: Signal event
        """
        # Extract signal data
        signal = event.get_data()
        
        # Validate signal
        if not self._validate_signal(signal):
            self.logger.warning(f"Invalid signal: {signal}")
            return
            
        # Calculate position size
        quantity = self._calculate_position_size(signal)
        
        # Skip if quantity is zero
        if quantity == 0:
            return
            
        # Check risk limits
        if not self._check_risk_limits(signal, quantity):
            self.logger.info(f"Signal rejected by risk limits: {signal}")
            return
            
        # Create order
        order = self._create_order(signal, quantity)
        
        # Emit order event
        self._emit_order(order)
        
    def _validate_signal(self, signal):
        """
        Validate signal data.
        
        Args:
            signal: Signal data dictionary
            
        Returns:
            bool: Whether signal is valid
        """
        # Check required fields
        required_fields = ['symbol', 'direction', 'price']
        for field in required_fields:
            if field not in signal:
                return False
                
        # Check direction
        if signal['direction'] not in ['BUY', 'SELL']:
            return False
            
        # Check price
        if not isinstance(signal['price'], (int, float)) or signal['price'] <= 0:
            return False
            
        return True
        
    def _calculate_position_size(self, signal):
        """
        Calculate position size for signal.
        
        Args:
            signal: Signal data dictionary
            
        Returns:
            int: Position size
        """
        symbol = signal['symbol']
        
        # Get position sizer for this symbol
        if symbol in self.position_sizers:
            sizer = self.position_sizers[symbol]
        else:
            sizer = self.default_sizer
            
        # Get current position
        current_position = 0
        position = self.portfolio.get_position(symbol)
        if position:
            current_position = position.quantity
            
        # Calculate position size
        quantity = sizer.calculate_position_size(
            signal=signal,
            portfolio=self.portfolio,
            current_position=current_position
        )
        
        return quantity
        
    def _check_risk_limits(self, signal, quantity):
        """
        Check if signal passes all risk limits.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            
        Returns:
            bool: Whether signal passes all limits
        """
        # Check each risk limit
        for limit in self.risk_limits:
            if not limit.check(signal, quantity, self.portfolio):
                return False
                
        return True
        
    def _create_order(self, signal, quantity):
        """
        Create order from signal.
        
        Args:
            signal: Signal data dictionary
            quantity: Position size
            
        Returns:
            dict: Order data
        """
        # Generate order ID
        order_id = self.order_id_counter.increment()
        
        # Create order data
        order = {
            'order_id': order_id,
            'symbol': signal['symbol'],
            'quantity': quantity,
            'direction': signal['direction'],
            'order_type': self.parameters.get('default_order_type', 'MARKET'),
            'price': signal['price'],
            'timestamp': signal.get('timestamp', datetime.now()),
            'source': signal.get('strategy', 'unknown')
        }
        
        # Add limit price for limit orders
        if order['order_type'] == 'LIMIT':
            order['limit_price'] = signal['price']
            
        return order
        
    def _emit_order(self, order):
        """
        Emit order event.
        
        Args:
            order: Order data dictionary
            
        Returns:
            bool: Success or failure
        """
        # Create and publish event
        event = Event(EventType.ORDER, order)
        return self.event_bus.publish(event)
        
    def reset(self):
        """Reset risk manager state."""
        super().reset()
        
        # Reset position sizers
        for sizer in self.position_sizers.values():
            if hasattr(sizer, 'reset'):
                sizer.reset()
                
        # Reset risk limits
        for limit in self.risk_limits:
            if hasattr(limit, 'reset'):
                limit.reset()
```

### 2. Portfolio Manager

The Portfolio class manages positions and equity:

```python
class Portfolio(Component):
    """
    Portfolio manager.
    
    Tracks positions, equity, and portfolio statistics.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize portfolio state
        self.positions = ThreadSafeDict()
        self.initial_cash = self.parameters.get('initial_cash', 100000)
        self.cash = self.initial_cash
        self.equity_curve = []
        self.trades = []
        
        # Maximum history lengths
        self.max_trades = self.parameters.get('max_trades', 10000)
        self.max_equity_points = self.parameters.get('max_equity_points', 10000)
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Set up event subscriptions
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.FILL, self.on_fill)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def on_fill(self, event):
        """
        Process fill event.
        
        Args:
            event: Fill event
        """
        # Extract fill data
        fill = event.get_data()
        
        # Process fill
        self._process_fill(fill)
        
    def on_bar(self, event):
        """
        Process bar event for mark-to-market.
        
        Args:
            event: Bar event
        """
        # Extract bar data
        bar = event.get_data()
        
        # Update position market values
        self._update_position_values(bar)
        
        # Update equity curve
        self._update_equity_curve(bar['timestamp'])
        
    def _process_fill(self, fill):
        """
        Process a fill event.
        
        Args:
            fill: Fill data dictionary
            
        Returns:
            Position: Updated position
        """
        # Extract fill details
        symbol = fill['symbol']
        quantity = fill['quantity']
        price = fill['price']
        commission = fill.get('commission', 0.0)
        timestamp = fill.get('timestamp', datetime.now())
        
        # Get or create position
        position = self._get_or_create_position(symbol)
        
        # Update position
        old_quantity = position.quantity
        realized_pnl = position.update(quantity, price, commission)
        
        # Update cash
        self.cash -= quantity * price  # Reduce by trade value
        self.cash -= commission  # Reduce by commission
        self.cash += realized_pnl  # Add realized P&L
        
        # Record trade if position changed
        if quantity != 0:
            self._record_trade({
                'symbol': symbol,
                'timestamp': timestamp,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'realized_pnl': realized_pnl,
                'order_id': fill.get('order_id'),
                'direction': 'BUY' if quantity > 0 else 'SELL'
            })
            
        # Update equity curve
        self._update_equity_curve(timestamp)
        
        return position
        
    def _update_position_values(self, bar):
        """
        Update position market values with new bar data.
        
        Args:
            bar: Bar data dictionary
        """
        symbol = bar['symbol']
        price = bar['close']
        
        # Update position if we have it
        if symbol in self.positions:
            position = self.positions[symbol]
            position.mark_to_market(price)
            
    def _update_equity_curve(self, timestamp):
        """
        Update equity curve with current portfolio value.
        
        Args:
            timestamp: Current timestamp
        """
        # Calculate portfolio value
        portfolio_value = self.cash
        
        # Add position values
        for position in self.positions.values():
            portfolio_value += position.market_value
            
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash
        })
        
        # Prune equity curve if too long
        if len(self.equity_curve) > self.max_equity_points:
            self.equity_curve = self.equity_curve[-self.max_equity_points:]
            
    def _record_trade(self, trade):
        """
        Record a trade.
        
        Args:
            trade: Trade data dictionary
        """
        self.trades.append(trade)
        
        # Prune trades if too many
        if len(self.trades) > self.max_trades:
            self.trades = self.trades[-self.max_trades:]
            
    def _get_or_create_position(self, symbol):
        """
        Get or create a position for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Position: Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
            
        return self.positions[symbol]
        
    def get_position(self, symbol):
        """
        Get position for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Position: Position object or None
        """
        return self.positions.get(symbol)
        
    def get_portfolio_value(self):
        """
        Get total portfolio value.
        
        Returns:
            float: Portfolio value
        """
        # Start with cash
        value = self.cash
        
        # Add position values
        for position in self.positions.values():
            value += position.market_value
            
        return value
        
    def get_open_positions(self):
        """
        Get all open positions.
        
        Returns:
            dict: Symbol -> Position mapping of non-zero positions
        """
        return {s: p for s, p in self.positions.items() if p.quantity != 0}
        
    def reset(self):
        """Reset portfolio state."""
        super().reset()
        
        # Reset positions
        self.positions = ThreadSafeDict()
        
        # Reset cash
        self.cash = self.initial_cash
        
        # Reset history
        self.equity_curve = []
        self.trades = []
```

### 3. Position Class

The Position class represents individual security positions:

```python
class Position:
    """
    Position model.
    
    Represents a position in a security with quantity, cost basis, and market value.
    """
    
    def __init__(self, symbol):
        """
        Initialize position.
        
        Args:
            symbol: Instrument symbol
        """
        self.symbol = symbol
        self.quantity = 0
        self.cost_basis = 0.0
        self.market_value = 0.0
        self.market_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self._lock = threading.RLock()
        
    def update(self, quantity, price, commission=0.0):
        """
        Update position with a trade.
        
        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Trade price
            commission: Trade commission
            
        Returns:
            float: Realized P&L
        """
        with self._lock:
            # Special case for first trade
            if self.quantity == 0:
                self.quantity = quantity
                self.cost_basis = price
                self.market_price = price
                self.market_value = self.quantity * price
                return 0.0
                
            # Calculate trade value
            trade_value = quantity * price
            
            # Check if reducing or increasing position
            realized_pnl = 0.0
            
            if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
                # Position reduction
                # Calculate realized P&L for the closed portion
                closing_quantity = min(abs(self.quantity), abs(quantity))
                if self.quantity > 0:
                    # Long position being reduced
                    realized_pnl = closing_quantity * (price - self.cost_basis)
                else:
                    # Short position being reduced
                    realized_pnl = closing_quantity * (self.cost_basis - price)
                    
                # Subtract commission
                realized_pnl -= commission
                
                # Update total realized P&L
                self.realized_pnl += realized_pnl
                
            # Update position
            old_quantity = self.quantity
            old_value = old_quantity * self.cost_basis
            
            new_quantity = quantity
            new_value = new_quantity * price
            
            # Calculate new position
            self.quantity += quantity
            
            # Update cost basis
            if self.quantity != 0:
                self.cost_basis = (old_value + new_value) / self.quantity
                
            # Update market values
            self.market_price = price
            self.market_value = self.quantity * price
            
            # Update unrealized P&L
            if self.quantity > 0:
                self.unrealized_pnl = self.quantity * (self.market_price - self.cost_basis)
            elif self.quantity < 0:
                self.unrealized_pnl = self.quantity * (self.cost_basis - self.market_price)
            else:
                self.unrealized_pnl = 0.0
                
            return realized_pnl
            
    def mark_to_market(self, price):
        """
        Update position market value with new price.
        
        Args:
            price: Current market price
            
        Returns:
            float: Change in unrealized P&L
        """
        with self._lock:
            old_value = self.market_value
            
            # Update market values
            self.market_price = price
            self.market_value = self.quantity * price
            
            # Update unrealized P&L
            old_unrealized = self.unrealized_pnl
            
            if self.quantity > 0:
                self.unrealized_pnl = self.quantity * (self.market_price - self.cost_basis)
            elif self.quantity < 0:
                self.unrealized_pnl = self.quantity * (self.cost_basis - self.market_price)
            else:
                self.unrealized_pnl = 0.0
                
            return self.unrealized_pnl - old_unrealized
            
    def close(self, price, commission=0.0):
        """
        Close position at specified price.
        
        Args:
            price: Close price
            commission: Commission
            
        Returns:
            float: Realized P&L
        """
        with self._lock:
            # Calculate closing quantity (opposite of current position)
            closing_quantity = -self.quantity
            
            # Update position
            return self.update(closing_quantity, price, commission)
            
    def get_info(self):
        """
        Get position information.
        
        Returns:
            dict: Position data
        """
        with self._lock:
            return {
                'symbol': self.symbol,
                'quantity': self.quantity,
                'cost_basis': self.cost_basis,
                'market_price': self.market_price,
                'market_value': self.market_value,
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl,
                'total_pnl': self.realized_pnl + self.unrealized_pnl
            }
```

### 4. Position Sizing

The PositionSizer interface and implementations:

```python
class PositionSizer:
    """
    Position sizer interface.
    
    Calculates position sizes for orders based on
    trading signals and portfolio state.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize with parameters.
        
        Args:
            parameters: Position sizer parameters
        """
        self.parameters = parameters or {}
        
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """
        Calculate position size.
        
        Args:
            signal: Signal data dictionary
            portfolio: Portfolio object
            current_position: Current position quantity
            
        Returns:
            int: Position size
        """
        raise NotImplementedError
```

```python
class FixedSizer(PositionSizer):
    """
    Fixed position sizer.
    
    Uses a fixed quantity for all orders.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """
        Calculate position size using fixed quantity.
        
        Args:
            signal: Signal data dictionary
            portfolio: Portfolio object
            current_position: Current position quantity
            
        Returns:
            int: Position size
        """
        # Get fixed size
        size = self.parameters.get('size', 100)
        
        # Apply direction
        if signal['direction'] == 'SELL':
            size = -size
            
        # Check if this is a position reversal
        if (size > 0 and current_position < 0) or (size < 0 and current_position > 0):
            # Include current position size in the order to reverse position
            size = size - current_position
            
        return size
```

```python
class PercentEquitySizer(PositionSizer):
    """
    Percent of equity position sizer.
    
    Sizes positions as a percentage of total portfolio equity.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """
        Calculate position size using percent of equity.
        
        Args:
            signal: Signal data dictionary
            portfolio: Portfolio object
            current_position: Current position quantity
            
        Returns:
            int: Position size
        """
        # Get percentage
        percentage = self.parameters.get('percentage', 5.0)
        
        # Calculate dollar amount
        equity = portfolio.get_portfolio_value()
        dollar_amount = equity * (percentage / 100.0)
        
        # Calculate shares based on price
        price = signal['price']
        shares = int(dollar_amount / price)
        
        # Apply direction
        if signal['direction'] == 'SELL':
            shares = -shares
            
        # Check if this is a position reversal
        if (shares > 0 and current_position < 0) or (shares < 0 and current_position > 0):
            # Include current position size in the order to reverse position
            shares = shares - current_position
            
        return shares
```

```python
class PercentRiskSizer(PositionSizer):
    """
    Percent risk position sizer.
    
    Sizes positions based on the amount of capital risked.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """
        Calculate position size using percent risk.
        
        Args:
            signal: Signal data dictionary
            portfolio: Portfolio object
            current_position: Current position quantity
            
        Returns:
            int: Position size
        """
        # Get risk percentage
        risk_percentage = self.parameters.get('risk_percentage', 1.0)
        
        # Get stop loss price
        if 'stop_price' not in signal:
            # Default to using a fixed percentage from entry price
            stop_percentage = self.parameters.get('stop_percentage', 2.0)
            
            if signal['direction'] == 'BUY':
                stop_price = signal['price'] * (1.0 - stop_percentage / 100.0)
            else:
                stop_price = signal['price'] * (1.0 + stop_percentage / 100.0)
        else:
            stop_price = signal['stop_price']
            
        # Calculate dollar risk
        equity = portfolio.get_portfolio_value()
        dollar_risk = equity * (risk_percentage / 100.0)
        
        # Calculate risk per share
        price = signal['price']
        risk_per_share = abs(price - stop_price)
        
        # Avoid division by zero
        if risk_per_share <= 0:
            return 0
            
        # Calculate shares based on risk
        shares = int(dollar_risk / risk_per_share)
        
        # Apply direction
        if signal['direction'] == 'SELL':
            shares = -shares
            
        # Check if this is a position reversal
        if (shares > 0 and current_position < 0) or (shares < 0 and current_position > 0):
            # Include current position size in the order to reverse position
            shares = shares - current_position
            
        return shares
```

### 5. Risk Limits

Risk limit implementations:

```python
class RiskLimit:
    """
    Risk limit interface.
    
    Checks if a signal passes specific risk criteria.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize with parameters.
        
        Args:
            parameters: Risk limit parameters
        """
        self.parameters = parameters or {}
        
    def check(self, signal, quantity, portfolio):
        """
        Check if signal passes the risk limit.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            bool: Whether signal passes the risk limit
        """
        raise NotImplementedError
```

```python
class PositionLimit(RiskLimit):
    """
    Maximum position size limit.
    
    Limits the maximum position size for any single instrument.
    """
    
    def check(self, signal, quantity, portfolio):
        """
        Check if position size is within limits.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            bool: Whether position size is acceptable
        """
        # Get maximum position size
        max_position = self.parameters.get('max_position', 1000)
        
        # Get symbol
        symbol = signal['symbol']
        
        # Get current position
        current_position = 0
        position = portfolio.get_position(symbol)
        if position:
            current_position = position.quantity
            
        # Calculate new position
        new_position = current_position + quantity
        
        # Check if new position exceeds maximum
        return abs(new_position) <= max_position
```

```python
class ExposureLimit(RiskLimit):
    """
    Maximum exposure limit.
    
    Limits the maximum exposure as a percentage of portfolio equity.
    """
    
    def check(self, signal, quantity, portfolio):
        """
        Check if exposure is within limits.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            bool: Whether exposure is acceptable
        """
        # Get maximum exposure percentage
        max_exposure = self.parameters.get('max_exposure', 20.0)
        
        # Calculate dollar value of this position
        price = signal['price']
        position_value = abs(quantity * price)
        
        # Calculate current portfolio value
        portfolio_value = portfolio.get_portfolio_value()
        
        # Calculate exposure percentage
        exposure_percentage = (position_value / portfolio_value) * 100.0
        
        # Check if exposure exceeds maximum
        return exposure_percentage <= max_exposure
```

```python
class DrawdownLimit(RiskLimit):
    """
    Drawdown control limit.
    
    Reduces or stops trading when drawdown exceeds thresholds.
    """
    
    def check(self, signal, quantity, portfolio):
        """
        Check if drawdown is within limits.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            portfolio: Portfolio object
            
        Returns:
            bool: Whether drawdown is acceptable
        """
        # Get drawdown limits
        max_drawdown = self.parameters.get('max_drawdown', 20.0)
        reduce_threshold = self.parameters.get('reduce_threshold', 10.0)
        reduction_factor = self.parameters.get('reduction_factor', 0.5)
        
        # Calculate current drawdown
        equity_curve = portfolio.equity_curve
        if not equity_curve:
            return True
            
        equity_values = [point['portfolio_value'] for point in equity_curve]
        peak = max(equity_values)
        current = equity_values[-1]
        
        drawdown_percentage = ((peak - current) / peak) * 100.0
        
        # Check if drawdown exceeds maximum
        if drawdown_percentage >= max_drawdown:
            return False
            
        # Check if drawdown requires position reduction
        if drawdown_percentage >= reduce_threshold:
            # Reduce position size
            reduced_quantity = int(quantity * (1.0 - drawdown_percentage / max_drawdown))
            
            # Update quantity in signal (hack, but simplest approach)
            signal['_original_quantity'] = quantity
            signal['_reduced_quantity'] = reduced_quantity
            
            # Use the reduced quantity
            quantity = reduced_quantity
            
        return True
```

## Best Practices

### Position Management

Follow these best practices for position management:

1. **Thread Safety**: Use thread-safe collections and locks
   ```python
   self.positions = ThreadSafeDict()
   
   def update(self, quantity, price, commission=0.0):
       with self._lock:
           # Position update logic
   ```

2. **Defensive Copying**: Return copies of position data
   ```python
   def get_position(self, symbol):
       position = self.positions.get(symbol)
       if position:
           return position.get_info()  # Return a copy
       return None
   ```

3. **Bounded Collections**: Limit history size
   ```python
   # Prune equity curve if too long
   if len(self.equity_curve) > self.max_equity_points:
       self.equity_curve = self.equity_curve[-self.max_equity_points:]
   ```

### Risk Management

Use these patterns for effective risk management:

```python
# Multi-layered approach to risk
def process_signal(self, signal):
    # 1. Validate signal first
    if not self._validate_signal(signal):
        return None
        
    # 2. Size the position
    quantity = self._size_position(signal)
    
    # 3. Apply risk limits
    if not self._check_risk_limits(signal, quantity):
        return None
        
    # 4. Create the order
    return self._create_order(signal, quantity)
```

### Event Flow

Use this pattern for event handling:

```python
def initialize_event_subscriptions(self):
    self.subscription_manager = SubscriptionManager(self.event_bus)
    self.subscription_manager.subscribe(EventType.SIGNAL, self.on_signal)
    self.subscription_manager.subscribe(EventType.FILL, self.on_fill)
    
def on_signal(self, event):
    signal = event.get_data()
    # Process signal...
    
def on_fill(self, event):
    fill = event.get_data()
    # Process fill...
```

## Implementation Considerations

### 1. Position Tracking

Position tracking is a critical and potentially error-prone aspect of the system. The following guidance will help ensure accurate P&L calculation:

#### Common Position Scenarios

Here are examples of how position tracking should work in different scenarios:

**Example 1: Increasing a Long Position**
```python
# Initial position: 100 shares at $50
position = Position("AAPL")
position.update(100, 50.0)  # quantity=100, price=50.0
assert position.quantity == 100
assert position.cost_basis == 50.0
assert position.market_value == 5000.0
assert position.realized_pnl == 0.0

# Add 50 more shares at $52
realized_pnl = position.update(50, 52.0)
assert position.quantity == 150
assert position.cost_basis == 50.67  # (100*50 + 50*52)/150
assert position.market_value == 7800.0  # 150 * 52
assert realized_pnl == 0.0  # No realized P&L when increasing position
```

**Example 2: Decreasing a Long Position**
```python
# Initial position: 100 shares at $50
position = Position("AAPL")
position.update(100, 50.0)

# Sell 60 shares at $55
realized_pnl = position.update(-60, 55.0)
assert position.quantity == 40
assert position.cost_basis == 50.0  # Cost basis doesn't change when decreasing
assert position.market_value == 2200.0  # 40 * 55
assert realized_pnl == 300.0  # 60 * (55 - 50)
```

**Example 3: Position Reversal**
```python
# Initial position: 100 shares at $50
position = Position("AAPL")
position.update(100, 50.0)

# Sell 150 shares at $48 (reversing from long to short)
realized_pnl = position.update(-150, 48.0)
assert position.quantity == -50
assert position.cost_basis == 48.0  # New cost basis for short position
assert position.market_value == -2400.0  # -50 * 48
assert realized_pnl == -200.0  # 100 * (48 - 50)
```

#### Validation Checks for P&L Calculations

To ensure P&L calculations are correct, implement these verification checks:

```python
def verify_position_update(position, old_quantity, old_cost_basis, 
                          old_market_value, trade_quantity, trade_price,
                          expected_new_quantity, expected_new_cost_basis,
                          expected_realized_pnl):
    """Verify that position updates produce expected results."""
    # Store original values
    assert position.quantity == old_quantity
    assert position.cost_basis == old_cost_basis
    assert position.market_value == old_market_value
    
    # Perform update
    realized_pnl = position.update(trade_quantity, trade_price)
    
    # Verify results
    assert position.quantity == expected_new_quantity
    assert abs(position.cost_basis - expected_new_cost_basis) < 0.001
    assert abs(realized_pnl - expected_realized_pnl) < 0.001
    
    # Verify total P&L consistency
    if position.quantity != 0:
        # For non-zero positions, verify unrealized P&L calculation
        if position.quantity > 0:
            expected_unrealized = position.quantity * (position.market_price - position.cost_basis)
        else:
            expected_unrealized = position.quantity * (position.cost_basis - position.market_price)
        
        assert abs(position.unrealized_pnl - expected_unrealized) < 0.001
```

#### Corner Cases to Handle

Be especially careful with these edge cases:

1. **Zero Crossing**: When a position changes from long to short or vice versa
   ```python
   # Handle position reversal
   if (current_position > 0 and current_position + quantity < 0) or \
      (current_position < 0 and current_position + quantity > 0):
       # First close the existing position
       realized_pnl += self._close_position(current_position, price)
       # Then open a new position in the opposite direction
       remaining_quantity = quantity + current_position
       self._open_position(remaining_quantity, price)
   ```

2. **Multiple Fills at Different Prices**: Especially important for calculating cost basis
   ```python
   # Track all fills for accurate FIFO/LIFO accounting
   def add_fill(self, quantity, price, timestamp):
       self.fills.append({
           'quantity': quantity,
           'price': price,
           'timestamp': timestamp
       })
       self._recalculate_position()
   ```

3. **Handling Commissions**: Include in P&L calculations
   ```python
   # Adjust realized P&L for commissions
   realized_pnl = (closing_quantity * (price - self.cost_basis)) - commission
   ```

#### Testing Strategies for Position Calculations

Implement these testing approaches to validate position tracking:

1. **Unit Tests**: Test individual position updates
   ```python
   def test_position_long_to_short():
       position = Position("AAPL")
       position.update(100, 50.0)
       realized_pnl = position.update(-150, 48.0)
       assert position.quantity == -50
       assert position.cost_basis == 48.0
       assert abs(realized_pnl - (-200.0)) < 0.001
   ```

2. **Integration Tests**: Verify position tracking through multiple events
   ```python
   def test_position_lifecycle():
       position = Position("AAPL")
       position.update(100, 50.0)      # Buy 100 @ $50
       position.mark_to_market(52.0)   # Price rises to $52
       position.update(50, 52.0)       # Buy 50 more @ $52
       position.update(-75, 53.0)      # Sell 75 @ $53
       position.mark_to_market(49.0)   # Price falls to $49
       final_pnl = position.close(49.0)  # Close position @ $49
       
       # Verify final P&L matches expectations
       expected_realized = 75 * (53 - 50.67) + 75 * (49 - 50.67)
       assert abs(position.realized_pnl - expected_realized) < 0.001
       assert position.quantity == 0
   ```

3. **P&L Reconciliation**: Verify that unrealized + realized = total P&L
   ```python
   def reconcile_pnl(position, trades):
       """Verify that P&L calculations are consistent."""
       # Sum all cash flows from trades
       cash_flow = sum(trade["quantity"] * trade["price"] for trade in trades)
       
       # Add current position value
       total_value = cash_flow + position.market_value
       
       # Compare with P&L accounting
       expected_total = position.realized_pnl + position.unrealized_pnl
       assert abs(total_value - expected_total) < 0.001
   ```

2. **P&L Calculation**: Track realized and unrealized P&L
   ```python
   # Unrealized P&L calculation
   if quantity > 0:
       unrealized_pnl = quantity * (market_price - cost_basis)
   elif quantity < 0:
       unrealized_pnl = quantity * (cost_basis - market_price)
   ```

3. **Market Value Updates**: Update on every bar
   ```python
   def on_bar(self, event):
       bar = event.get_data()
       symbol = bar['symbol']
       
       if symbol in self.positions:
           self.positions[symbol].mark_to_market(bar['close'])
   ```

### 2. Risk Adaptations

For adaptive risk management, consider:

1. **Dynamic Position Sizing**: Adjust with market conditions
   ```python
   # Adjust position size based on volatility
   def calculate_position_size(self, signal, portfolio):
       volatility = self._calculate_volatility(signal['symbol'])
       base_size = self._calculate_base_size(portfolio)
       return int(base_size / volatility)
   ```

2. **Drawdown Control**: Reduce exposure during drawdowns
   ```python
   # Scale down position sizes during drawdowns
   drawdown = self._calculate_drawdown(portfolio)
   scaling_factor = max(0.2, 1.0 - drawdown)
   adjusted_size = int(base_size * scaling_factor)
   ```

3. **Risk Budgeting**: Allocate risk across positions
   ```python
   # Calculate position sizes using risk parity
   def allocate_risk(self, signals, portfolio):
       total_risk = self.parameters.get('risk_budget', 0.01)
       volatilities = {s['symbol']: self._get_volatility(s['symbol']) for s in signals}
       sum_inv_vol = sum(1.0 / v for v in volatilities.values())
       
       for signal in signals:
           risk_allocation = total_risk * (1.0 / volatilities[signal['symbol']]) / sum_inv_vol
           signal['_risk_allocation'] = risk_allocation
   ```

### 3. Portfolio Optimizations

For efficient portfolio management, consider:

1. **Portfolio Rebalancing**: Periodically rebalance positions
   ```python
   def rebalance_portfolio(self, target_weights):
       current_positions = self.get_open_positions()
       portfolio_value = self.get_portfolio_value()
       
       for symbol, target_weight in target_weights.items():
           target_value = portfolio_value * target_weight
           current_value = 0
           
           if symbol in current_positions:
               position = current_positions[symbol]
               current_value = position.market_value
               
           value_difference = target_value - current_value
           
           if abs(value_difference) > self.min_rebalance_value:
               # Create rebalancing order
               price = self._get_current_price(symbol)
               quantity = int(value_difference / price)
               
               if quantity != 0:
                   self._create_rebalance_order(symbol, quantity, price)
   ```

2. **Performance Tracking**: Track detailed metrics
   ```python
   def calculate_performance_metrics(self):
       if not self.equity_curve:
           return {}
           
       equity_values = [point['portfolio_value'] for point in self.equity_curve]
       timestamps = [point['timestamp'] for point in self.equity_curve]
       
       # Calculate returns
       returns = []
       for i in range(1, len(equity_values)):
           ret = (equity_values[i] / equity_values[i-1]) - 1
           returns.append(ret)
           
       # Calculate metrics
       metrics = {
           'total_return': (equity_values[-1] / equity_values[0]) - 1,
           'volatility': np.std(returns) * np.sqrt(252),  # Annualized
           'sharpe_ratio': self._calculate_sharpe(returns),
           'max_drawdown': self._calculate_max_drawdown(equity_values),
           'win_rate': self._calculate_win_rate(self.trades)
       }
       
       return metrics
   ```

By following these guidelines, you'll create a robust Risk module that provides proper position sizing, risk control, and portfolio management for the ADMF-Trader system.