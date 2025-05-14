# Risk Module Implementation Guide

## Overview

The Risk module is responsible for position sizing, risk control, and portfolio management in the ADMF-Trader system. It receives trading signals from strategies, applies risk limits and position sizing algorithms, and converts them to orders for execution. It also tracks positions and portfolio value with robust handling of complex trading scenarios.

## Directory Structure

```
src/risk/
├── __init__.py
├── interfaces/
│   ├── __init__.py
│   ├── risk_manager.py        # Risk manager interface
│   ├── position_sizer.py      # Position sizer interface
│   └── risk_limit.py          # Risk limit interface
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
│   ├── percent_risk_sizer.py  # Risk-based position sizing
│   ├── kelly_sizer.py         # Kelly criterion sizer
│   └── volatility_sizer.py    # Volatility-based sizing
├── limits/
│   ├── __init__.py
│   ├── position_limit.py      # Maximum position size
│   ├── exposure_limit.py      # Maximum exposure
│   ├── drawdown_limit.py      # Drawdown control
│   ├── max_loss_limit.py      # Maximum loss control
│   └── max_positions_limit.py # Maximum positions count
├── composition/
│   ├── __init__.py
│   ├── registry.py            # Risk limit registry
│   ├── strategies.py          # Composition strategies
│   ├── precedence.py          # Precedence rules
│   ├── validator.py           # Risk constraint validator
│   └── evaluator.py           # Risk limit evaluator
└── utils/
    ├── __init__.py
    ├── reconciliation.py      # Position reconciliation
    └── trade_processor.py     # Trade processing utilities
```

## Core Components

### 1. Risk Manager

The `RiskManager` converts signals to orders with appropriate risk control:

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
        
        # Create risk limit registry
        self.risk_limit_registry = RiskLimitRegistry()
        
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
        
        # Define composition strategy
        self._setup_risk_composition()
        
        # Validate limits
        self._validate_risk_limits()
        
    def on_signal(self, event):
        """Process signal event and generate orders."""
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
        
    def _check_risk_limits(self, signal, quantity):
        """
        Check if signal passes all risk limits using composition.
        
        Args:
            signal: Signal data dictionary
            quantity: Calculated position size
            
        Returns:
            bool: Whether signal passes all limits
        """
        # Evaluate signal against composition rules
        result = self.risk_evaluator.evaluate_signal(signal, quantity, self.portfolio)
        
        # If signal failed but suggested modification exists
        if not result.passed and result.modified_quantity is not None:
            # Update quantity in signal (to be used by order creation)
            signal['_original_quantity'] = quantity
            signal['_modified_quantity'] = result.modified_quantity
            
            # Log modifications
            self.logger.info(
                f"Signal quantity modified from {quantity} to {result.modified_quantity} "
                f"due to risk limits. Violations: {result.violations}"
            )
            
            return True  # Allow to proceed with modified quantity
            
        return result.passed
```

### 2. Portfolio Manager

The `Portfolio` class manages positions and equity:

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
        """Process fill event."""
        # Extract fill data
        fill = event.get_data()
        
        # Process fill
        self._process_fill(fill)
        
    def on_bar(self, event):
        """Process bar event for mark-to-market."""
        # Extract bar data
        bar = event.get_data()
        
        # Update position market values
        self._update_position_values(bar)
        
        # Update equity curve
        self._update_equity_curve(bar['timestamp'])
        
    def get_portfolio_value(self):
        """Get total portfolio value."""
        # Start with cash
        value = self.cash
        
        # Add position values
        for position in self.positions.values():
            value += position.market_value
            
        return value
        
    def get_position(self, symbol):
        """Get position for a symbol."""
        return self.positions.get(symbol)
        
    def get_open_positions(self):
        """Get all open positions."""
        return {s: p for s, p in self.positions.items() if not p.is_flat}
```

### 3. Enhanced Position Class

The `Position` class provides robust position tracking with precise decimal calculations and comprehensive trade history:

```python
from decimal import Decimal, getcontext
from enum import Enum, auto
import threading
import uuid
from datetime import datetime

# Set decimal precision for position calculations
getcontext().prec = 28

class PositionSide(Enum):
    """Position side enum."""
    LONG = auto()
    SHORT = auto()
    FLAT = auto()

class Position:
    """
    Robust position tracking model.
    
    Represents a position in a security with quantity, cost basis, 
    and market value information. Handles position reversals and 
    provides complete trade history.
    """
    
    def __init__(self, symbol):
        """Initialize position."""
        self.symbol = symbol
        self.quantity = Decimal('0')
        self.average_price = Decimal('0')
        self.cost_basis = Decimal('0')
        self.market_price = Decimal('0')
        self.market_value = Decimal('0')
        self.realized_pnl = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.entries = []  # List of entry trades
        self.exits = []  # List of exit trades
        self.trades = []  # All trades in sequence
        self.current_side = PositionSide.FLAT
        self._lock = threading.RLock()
        self._zero = Decimal('0')
        
    @property
    def is_long(self):
        """Check if position is long."""
        return self.quantity > self._zero
        
    @property
    def is_short(self):
        """Check if position is short."""
        return self.quantity < self._zero
        
    @property
    def is_flat(self):
        """Check if position is flat (no position)."""
        return self.quantity == self._zero
        
    @property
    def direction(self):
        """Get position direction as string."""
        if self.is_long:
            return "LONG"
        elif self.is_short:
            return "SHORT"
        else:
            return "FLAT"
            
    @property
    def absolute_quantity(self):
        """Get absolute quantity."""
        return abs(self.quantity)
        
    @property
    def absolute_market_value(self):
        """Get absolute market value."""
        return abs(self.market_value)
        
    def update(self, quantity, price, commission=0.0, timestamp=None, trade_id=None, metadata=None):
        """Update position with a new trade."""
        with self._lock:
            # Convert inputs to Decimal for precise calculation
            quantity = Decimal(str(quantity))
            price = Decimal(str(price))
            commission = Decimal(str(commission))
            timestamp = timestamp or datetime.now()
            trade_id = trade_id or str(uuid.uuid4())
            
            # Create trade record
            trade = {
                'trade_id': trade_id,
                'symbol': self.symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp,
                'commission': commission,
                'metadata': metadata or {}
            }
            
            # Handle position update based on current and new positions
            if self.is_flat:
                # Opening new position
                self.quantity = quantity
                self.average_price = price
                self.cost_basis = quantity * price
                self.market_price = price
                self.market_value = quantity * price
                trade['trade_type'] = 'ENTRY'
                self.entries.append(trade)
                
                # Set position side
                if quantity > self._zero:
                    self.current_side = PositionSide.LONG
                elif quantity < self._zero:
                    self.current_side = PositionSide.SHORT
                else:
                    self.current_side = PositionSide.FLAT
                    
            elif self.is_long:
                if quantity > self._zero:
                    # Adding to long position
                    new_quantity = self.quantity + quantity
                    self.cost_basis += quantity * price
                    self.average_price = self.cost_basis / new_quantity
                    self.quantity = new_quantity
                    trade['trade_type'] = 'ENTRY'
                    self.entries.append(trade)
                    
                elif quantity < self._zero:
                    # Reducing or closing long position
                    abs_quantity = abs(quantity)
                    
                    if abs_quantity < self.quantity:
                        # Partial exit
                        exit_cost = abs_quantity * self.average_price
                        exit_proceeds = abs_quantity * price
                        realized_pnl = exit_proceeds - exit_cost - commission
                        
                        # Update position
                        self.realized_pnl += realized_pnl
                        self.quantity += quantity  # quantity is negative
                        self.cost_basis = self.quantity * self.average_price
                        
                        # Record pnl in trade
                        trade['realized_pnl'] = realized_pnl
                        trade['exit_price'] = price
                        trade['entry_price'] = self.average_price
                        trade['trade_type'] = 'EXIT'
                        self.exits.append(trade)
                        
                    elif abs_quantity == self.quantity:
                        # Full exit
                        exit_cost = self.cost_basis
                        exit_proceeds = abs_quantity * price
                        realized_pnl = exit_proceeds - exit_cost - commission
                        
                        # Update position
                        self.realized_pnl += realized_pnl
                        self.quantity = self._zero
                        self.cost_basis = self._zero
                        self.average_price = self._zero
                        self.current_side = PositionSide.FLAT
                        
                        # Record pnl in trade
                        trade['realized_pnl'] = realized_pnl
                        trade['exit_price'] = price
                        trade['entry_price'] = self.average_price
                        trade['trade_type'] = 'EXIT'
                        self.exits.append(trade)
                        
                    else:
                        # Position reversal (long to short)
                        # First handle full exit of long position
                        exit_cost = self.cost_basis
                        exit_proceeds = self.quantity * price
                        realized_pnl = exit_proceeds - exit_cost - commission
                        
                        # Create exit trade for the full long position
                        exit_trade = trade.copy()
                        exit_trade['quantity'] = -self.quantity
                        exit_trade['realized_pnl'] = realized_pnl
                        exit_trade['exit_price'] = price
                        exit_trade['entry_price'] = self.average_price
                        exit_trade['trade_type'] = 'EXIT'
                        exit_trade['trade_id'] = str(uuid.uuid4())
                        self.exits.append(exit_trade)
                        
                        # Calculate remaining quantity for short position
                        short_quantity = -(abs_quantity - self.quantity)
                        
                        # Update position to new short position
                        self.realized_pnl += realized_pnl
                        self.quantity = short_quantity
                        self.average_price = price
                        self.cost_basis = short_quantity * price
                        self.market_price = price
                        self.market_value = short_quantity * price
                        self.current_side = PositionSide.SHORT
                        
                        # Record entry trade for short position
                        entry_trade = trade.copy()
                        entry_trade['quantity'] = short_quantity
                        entry_trade['trade_type'] = 'ENTRY'
                        entry_trade['trade_id'] = str(uuid.uuid4())
                        self.entries.append(entry_trade)
                        
                        # Special case for position reversal
                        trade['trade_type'] = 'REVERSAL'
                        trade['reversal_details'] = {
                            'from_side': 'LONG',
                            'to_side': 'SHORT',
                            'exit_quantity': float(self.quantity),
                            'entry_quantity': float(short_quantity),
                            'realized_pnl': float(realized_pnl)
                        }
                        
            elif self.is_short:
                # Similar logic for short positions (implementation follows same pattern)
                pass
                
            # Update market values
            self.market_price = price
            self.market_value = self.quantity * price
            
            # Update unrealized P&L
            if self.is_long:
                self.unrealized_pnl = self.quantity * (self.market_price - self.average_price)
            elif self.is_short:
                self.unrealized_pnl = self.quantity * (self.average_price - self.market_price)
            else:
                self.unrealized_pnl = self._zero
                
            # Add trade to complete history
            self.trades.append(trade)
            
            return trade
```

## Position Sizing

### Position Sizer Interface

```python
class PositionSizer:
    """
    Position sizer interface.
    
    Calculates position sizes for orders based on
    trading signals and portfolio state.
    """
    
    def __init__(self, parameters=None):
        """Initialize with parameters."""
        self.parameters = parameters or {}
        
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """Calculate position size."""
        raise NotImplementedError
```

### Fixed Position Sizer

```python
class FixedSizer(PositionSizer):
    """
    Fixed position sizer.
    
    Uses a fixed quantity for all orders.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """Calculate position size using fixed quantity."""
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

### Percent Equity Sizer

```python
class PercentEquitySizer(PositionSizer):
    """
    Percent of equity position sizer.
    
    Sizes positions as a percentage of total portfolio equity.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """Calculate position size using percent of equity."""
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

### Kelly Criterion Sizer

```python
class KellySizer(PositionSizer):
    """
    Kelly criterion position sizer.
    
    Sizes positions based on the Kelly criterion for optimal capital growth.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """Calculate position size using Kelly criterion."""
        # Get parameters
        win_rate = self.parameters.get('win_rate', 0.5)
        win_loss_ratio = self.parameters.get('win_loss_ratio', 2.0)
        max_kelly_percentage = self.parameters.get('max_kelly_percentage', 20.0)
        
        # Get signal parameters
        # If signal contains edge information, use it instead of parameters
        if 'win_rate' in signal and 'win_loss_ratio' in signal:
            win_rate = signal['win_rate']
            win_loss_ratio = signal['win_loss_ratio']
        
        # Calculate Kelly percentage
        kelly_fraction = win_rate - ((1.0 - win_rate) / win_loss_ratio)
        
        # Apply maximum constraint
        kelly_fraction = min(kelly_fraction, max_kelly_percentage / 100.0)
        
        # Don't take negative Kelly positions
        if kelly_fraction <= 0:
            return 0
            
        # Calculate position size
        equity = portfolio.get_portfolio_value()
        dollar_amount = equity * kelly_fraction
        
        # Calculate shares
        price = signal['price']
        shares = int(dollar_amount / price)
        
        # Apply direction
        if signal['direction'] == 'SELL':
            shares = -shares
            
        return shares
```

### Volatility Sizer

```python
class VolatilitySizer(PositionSizer):
    """
    Volatility-based position sizer.
    
    Sizes positions inversely to market volatility.
    """
    
    def calculate_position_size(self, signal, portfolio, current_position=0):
        """Calculate position size based on volatility."""
        # Get parameters
        base_size = self.parameters.get('base_size', 100)
        volatility_lookback = self.parameters.get('volatility_lookback', 20)
        target_volatility = self.parameters.get('target_volatility', 0.01)  # 1%
        
        # Get volatility from signal or calculate
        volatility = signal.get('volatility', None)
        
        if volatility is None:
            # Get historical prices (would be provided by a data handler)
            if 'historical_prices' in signal:
                prices = signal['historical_prices']
                if len(prices) >= volatility_lookback:
                    # Calculate volatility as standard deviation of returns
                    returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
                    volatility = self._calculate_stdev(returns)
                else:
                    volatility = target_volatility
            else:
                # Default if no data is available
                volatility = target_volatility
                
        # Scale position size inversely with volatility
        if volatility > 0:
            size = int(base_size * (target_volatility / volatility))
        else:
            size = base_size
            
        # Apply direction
        if signal['direction'] == 'SELL':
            size = -size
            
        return size
        
    def _calculate_stdev(self, values):
        """Calculate standard deviation."""
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
```

## Risk Limits

### Risk Limit Interface

```python
class RiskLimitBase:
    """
    Base class for all risk limits.
    
    Provides interface for checking limits and supporting composition.
    """
    
    def __init__(self, parameters=None):
        """Initialize with parameters."""
        self.parameters = parameters or {}
        self.limit_id = str(uuid.uuid4())
        self.limit_type = self.__class__.__name__
        self.priority = self.parameters.get('priority', 50)  # Default priority
        
    def get_id(self):
        """Get unique identifier for this limit."""
        return self.limit_id
        
    def get_type(self):
        """Get limit type."""
        return self.limit_type
        
    def get_priority(self):
        """Get limit priority."""
        return self.priority
        
    def check(self, signal, quantity, portfolio):
        """Check if signal passes the risk limit."""
        raise NotImplementedError
        
    def explain_violation(self, signal, quantity, portfolio):
        """Explain why a limit was violated."""
        return f"Risk limit {self.limit_type} was violated"
        
    def modify_quantity(self, signal, quantity, portfolio):
        """Modify quantity to conform to risk limit."""
        return 0  # Default implementation rejects trade completely
```

### Position Limit

```python
class PositionLimit(RiskLimitBase):
    """
    Maximum position size limit.
    
    Limits the maximum position size for any single instrument.
    """
    
    def check(self, signal, quantity, portfolio):
        """Check if position size is within limits."""
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
        
    def explain_violation(self, signal, quantity, portfolio):
        """Explain position limit violation."""
        max_position = self.parameters.get('max_position', 1000)
        symbol = signal['symbol']
        current_position = 0
        position = portfolio.get_position(symbol)
        
        if position:
            current_position = position.quantity
            
        new_position = current_position + quantity
        
        return (f"Position limit violated: {abs(new_position)} exceeds "
                f"maximum of {max_position} for {symbol}")
                
    def modify_quantity(self, signal, quantity, portfolio):
        """Modify quantity to conform to position limit."""
        max_position = self.parameters.get('max_position', 1000)
        symbol = signal['symbol']
        
        # Get current position
        current_position = 0
        position = portfolio.get_position(symbol)
        if position:
            current_position = position.quantity
            
        # Calculate maximum allowed change
        if quantity > 0:
            # Long position limit
            max_change = max_position - current_position
            return min(quantity, max_change)
        else:
            # Short position limit
            max_change = -max_position - current_position
            return max(quantity, max_change)
```

### Max Drawdown Limit

```python
class DrawdownLimit(RiskLimitBase):
    """
    Drawdown control limit.
    
    Reduces or stops trading when drawdown exceeds thresholds.
    """
    
    def check(self, signal, quantity, portfolio):
        """Check if drawdown is within limits."""
        # Get drawdown limits
        max_drawdown = self.parameters.get('max_drawdown', 20.0)
        reduce_threshold = self.parameters.get('reduce_threshold', 10.0)
        
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
            
            # Update quantity in signal
            signal['_original_quantity'] = quantity
            signal['_reduced_quantity'] = reduced_quantity
            
        return True
        
    def explain_violation(self, signal, quantity, portfolio):
        """Explain drawdown limit violation."""
        max_drawdown = self.parameters.get('max_drawdown', 20.0)
        
        equity_curve = portfolio.equity_curve
        if not equity_curve:
            return "Drawdown limit violated: No equity data available"
            
        equity_values = [point['portfolio_value'] for point in equity_curve]
        peak = max(equity_values)
        current = equity_values[-1]
        
        drawdown_percentage = ((peak - current) / peak) * 100.0
        
        return (f"Drawdown limit violated: Current drawdown of {drawdown_percentage:.2f}% "
                f"exceeds maximum of {max_drawdown}%")
                
    def modify_quantity(self, signal, quantity, portfolio):
        """Modify quantity to conform to drawdown limit."""
        # Get drawdown limits
        max_drawdown = self.parameters.get('max_drawdown', 20.0)
        reduce_threshold = self.parameters.get('reduce_threshold', 10.0)
        
        # Calculate current drawdown
        equity_curve = portfolio.equity_curve
        if not equity_curve:
            return quantity  # No modification if no equity data
            
        equity_values = [point['portfolio_value'] for point in equity_curve]
        peak = max(equity_values)
        current = equity_values[-1]
        
        drawdown_percentage = ((peak - current) / peak) * 100.0
        
        # If above max drawdown, reject trade
        if drawdown_percentage >= max_drawdown:
            return 0
            
        # If above reduce threshold, scale down position
        if drawdown_percentage >= reduce_threshold:
            # Linear scaling between threshold and max
            scale_factor = 1.0 - ((drawdown_percentage - reduce_threshold) / 
                                 (max_drawdown - reduce_threshold))
            return int(quantity * scale_factor)
            
        return quantity  # No modification needed
```

## Risk Limit Composition

### Risk Limit Registry

```python
class RiskLimitRegistry:
    """
    Central registry for all risk limits in the system.
    
    Maintains relationships between risk limits and provides
    access to limit configurations.
    """
    
    def __init__(self):
        self.limits = {}  # type_id -> RiskLimit
        self.limit_metadata = {}  # type_id -> metadata
        self.limit_relationships = {}  # (limit_id1, limit_id2) -> relationship
        
    def register_limit(self, limit, metadata=None):
        """Register a risk limit with optional metadata."""
        limit_id = limit.get_id()
        self.limits[limit_id] = limit
        self.limit_metadata[limit_id] = metadata or {}
        
    def set_relationship(self, limit_id1, limit_id2, relationship):
        """Set relationship between two limits."""
        self.limit_relationships[(limit_id1, limit_id2)] = relationship
        
    def get_relationship(self, limit_id1, limit_id2):
        """Get relationship between two limits."""
        return self.limit_relationships.get((limit_id1, limit_id2))
        
    def get_limits_by_type(self, limit_type):
        """Get all limits of a specific type."""
        return [l for l in self.limits.values() if l.get_type() == limit_type]
        
    def get_limit_metadata(self, limit_id):
        """Get metadata for a limit."""
        return self.limit_metadata.get(limit_id, {})
```

### Composition Strategies

```python
class CompositionStrategy:
    """
    Strategy for composing multiple risk limits.
    
    Defines how multiple risk limits are combined and evaluated.
    """
    
    def compose(self, risk_limits, signal, quantity, portfolio):
        """Compose multiple risk limits."""
        raise NotImplementedError
        
class CompositionResult:
    """Result of composing multiple risk limits."""
    
    def __init__(self):
        self.passed = False
        self.modified_quantity = None
        self.violations = []
        self.limit_results = {}  # limit_id -> individual result
        
    def add_limit_result(self, limit_id, passed, explanation=None):
        """Add individual limit result."""
        self.limit_results[limit_id] = {
            'passed': passed,
            'explanation': explanation
        }
        
        if not passed and explanation:
            self.violations.append(explanation)
            
    def set_modified_quantity(self, quantity):
        """Set modified quantity that would pass limits."""
        self.modified_quantity = quantity
        
    def set_passed(self, passed):
        """Set overall pass/fail result."""
        self.passed = passed
        
    def get_result_summary(self):
        """Get summary of composition result."""
        return {
            'passed': self.passed,
            'modified_quantity': self.modified_quantity,
            'violations': self.violations,
            'limit_results': self.limit_results
        }
```

### Risk Limit Evaluator

```python
class RiskLimitEvaluator:
    """
    Evaluates risk limits using composition strategies.
    
    Central component for applying risk constraints to trades.
    """
    
    def __init__(self, registry, composition_strategy, validator=None):
        """Initialize evaluator."""
        self.registry = registry
        self.composition_strategy = composition_strategy
        self.validator = validator
        
    def validate_limits(self, portfolio_context=None):
        """Validate all registered limits."""
        if self.validator:
            limits = list(self.registry.limits.values())
            return self.validator.validate_limits(limits, portfolio_context)
        return None
        
    def evaluate_signal(self, signal, quantity, portfolio):
        """Evaluate a signal against all applicable risk limits."""
        # Get applicable limits
        symbol = signal.get('symbol')
        limits = []
        
        # Filter applicable limits
        for limit_id, limit in self.registry.limits.items():
            metadata = self.registry.get_limit_metadata(limit_id)
            
            # Get applicable symbols for this limit
            applicable_symbols = metadata.get('symbols', None)
            
            # If symbols specified, check if this symbol applies
            if applicable_symbols is not None and symbol not in applicable_symbols:
                continue
                
            limits.append(limit)
            
        # Compose limits
        result = self.composition_strategy.compose(limits, signal, quantity, portfolio)
        
        return result
```

## Position Tracking and Reconciliation

### Position Reconciliation Utility

```python
class PositionReconciliation:
    """
    Position reconciliation utility.
    
    Reconciles internally tracked positions with external sources 
    (broker, exchange, etc.) and resolves discrepancies.
    """
    
    def __init__(self, portfolio):
        """Initialize reconciliation utility."""
        self.portfolio = portfolio
        
    def reconcile(self, external_positions):
        """Reconcile portfolio positions with external positions."""
        results = {
            'matched': [],
            'mismatched': [],
            'missing_internal': [],
            'missing_external': [],
            'adjustments': []
        }
        
        # Check each internal position against external
        for symbol, position in self.portfolio.positions.items():
            if position.is_flat:
                continue  # Skip flat positions
                
            if symbol in external_positions:
                # Position exists in both - compare
                external = external_positions[symbol]
                
                # Extract quantities
                internal_qty = position.quantity
                external_qty = Decimal(str(external.get('quantity', 0)))
                
                if internal_qty == external_qty:
                    # Quantities match
                    results['matched'].append({
                        'symbol': symbol,
                        'internal': float(internal_qty),
                        'external': float(external_qty)
                    })
                else:
                    # Quantities don't match
                    results['mismatched'].append({
                        'symbol': symbol,
                        'internal': float(internal_qty),
                        'external': float(external_qty),
                        'difference': float(external_qty - internal_qty)
                    })
            else:
                # Position exists internally but not externally
                results['missing_external'].append({
                    'symbol': symbol,
                    'internal': float(position.quantity)
                })
                
        # Check for positions in external but not internal
        for symbol, external in external_positions.items():
            if symbol not in self.portfolio.positions or self.portfolio.positions[symbol].is_flat:
                external_qty = Decimal(str(external.get('quantity', 0)))
                
                if external_qty != 0:
                    results['missing_internal'].append({
                        'symbol': symbol,
                        'external': float(external_qty)
                    })
                    
        return results
        
    def apply_adjustments(self, adjustments):
        """Apply position adjustments."""
        results = []
        
        for adjustment in adjustments:
            symbol = adjustment.get('symbol')
            target_quantity = Decimal(str(adjustment.get('target_quantity', 0)))
            price = Decimal(str(adjustment.get('price', 0)))
            reason = adjustment.get('reason', 'Manual adjustment')
            
            # Get current position
            position = self.portfolio.get_position(symbol)
            current_quantity = position.quantity
            
            # Calculate adjustment
            adjustment_quantity = target_quantity - current_quantity
            
            if adjustment_quantity == 0:
                continue  # No adjustment needed
                
            # Apply adjustment
            metadata = {
                'adjustment': True,
                'reason': reason,
                'previous_quantity': float(current_quantity)
            }
            
            trade = self.portfolio.update_position(
                symbol=symbol,
                quantity=adjustment_quantity,
                price=price,
                metadata=metadata
            )
            
            # Record adjustment
            results.append({
                'symbol': symbol,
                'adjustment_quantity': float(adjustment_quantity),
                'previous_quantity': float(current_quantity),
                'new_quantity': float(position.quantity),
                'price': float(price),
                'reason': reason,
                'trade': trade
            })
            
        return {
            'adjustments': results,
            'count': len(results)
        }
```

## Implementation Best Practices

### 1. Thread Safety for Position Tracking

```python
from threading import RLock

class ThreadSafeDict(dict):
    """Thread-safe dictionary implementation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RLock()
        
    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)
            
    def __setitem__(self, key, value):
        with self._lock:
            return super().__setitem__(key, value)
            
    def __delitem__(self, key):
        with self._lock:
            return super().__delitem__(key)
            
    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)
            
    def update(self, *args, **kwargs):
        with self._lock:
            return super().update(*args, **kwargs)
```

### 2. Position Handling Edge Cases

```python
# Handle position reversals correctly
if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
    # Position reduction or reversal
    closing_quantity = min(abs(self.quantity), abs(quantity))
    
    if self.quantity > 0:
        # Long position being reduced
        realized_pnl = closing_quantity * (price - self.average_price) - commission
    else:
        # Short position being reduced
        realized_pnl = closing_quantity * (self.average_price - price) - commission
        
    # Update realized P&L
    self.realized_pnl += realized_pnl
    
    # Check if this is a full reversal
    if abs(quantity) > abs(self.quantity):
        # Handle position reversal logic
        pass
```

### 3. Precise Decimal Calculations

```python
from decimal import Decimal, getcontext

# Set decimal precision for position calculations
getcontext().prec = 28

# Convert inputs to Decimal for precise calculation
quantity = Decimal(str(quantity))
price = Decimal(str(price))
commission = Decimal(str(commission))

# Calculate transaction value
transaction_value = quantity * price

# Calculate P&L with precise decimal arithmetic
realized_pnl = Decimal(str(closing_quantity)) * (price - self.average_price)
```

### 4. Risk Limit Composition Guidelines

```python
# Use clear priorities for limits
drawdown_limit = DrawdownLimit({
    'max_drawdown': 15.0,
    'reduce_threshold': 10.0,
    'priority': 90  # High priority
})

position_limit = PositionLimit({
    'max_position': 1000,
    'priority': 50  # Medium priority
})

# Define explicit precedence rules
registry.set_relationship(drawdown_limit.get_id(), position_limit.get_id(), 'dominates')

# Use appropriate composition strategy
if critical_risk_management_needed:
    composition_strategy = PriorityBasedStrategy()  # Highest priority limits prevail
else:
    composition_strategy = AllPassStrategy()  # All limits must pass
```

### 5. Trade Processing Best Practices

```python
def process_fill(self, fill):
    """Process fill with robust error handling."""
    try:
        # Extract fill data
        symbol = fill.get('symbol')
        quantity = fill.get('quantity')
        price = fill.get('price')
        
        # Validate fill data
        if not all([symbol, quantity, price]):
            self.logger.error(f"Invalid fill data: {fill}")
            return
            
        # Get or create position
        position = self._get_or_create_position(symbol)
        
        # Update position with fill
        trade_result = position.update(
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            commission=Decimal(str(fill.get('commission', 0))),
            timestamp=fill.get('timestamp'),
            trade_id=fill.get('id'),
            metadata={'fill': fill}
        )
        
        # Update cash balance
        self._update_cash(trade_result)
        
        # Update portfolio statistics
        self._update_portfolio_statistics()
        
    except Exception as e:
        self.logger.error(f"Error processing fill: {e}")
```

## Implementation Timeline

1. **Phase 1: Core Infrastructure (Week 1)**
   - Implement robust Position class with decimal precision
   - Create Portfolio and RiskManager base implementations
   - Set up thread-safe collections and proper locking

2. **Phase 2: Position Sizing (Week 2)**
   - Implement basic position sizers (Fixed, PercentEquity)
   - Create advanced position sizers (PercentRisk, Kelly, Volatility)
   - Develop sizing selection framework

3. **Phase 3: Risk Limits (Week 3)**
   - Implement basic risk limits (Position, Exposure)
   - Create advanced risk limits (Drawdown, MaxLoss, MaxPositions)
   - Set up risk limit composition framework

4. **Phase 4: Position Tracking (Week 4)**
   - Implement position reconciliation utilities
   - Create position adjustment mechanisms
   - Develop comprehensive P&L tracking with edge case handling

5. **Phase 5: Integration and Testing (Week 5)**
   - Integrate with Strategy and Execution modules
   - Create comprehensive unit tests for edge cases
   - Set up integration tests for the full risk pipeline