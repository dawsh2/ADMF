# Position Tracking Robustness

## Overview

This document outlines the design for robust position tracking in the ADMF-Trader system. The position tracking system is responsible for accurately maintaining the current state of all positions, handling various edge cases, and providing reconciliation capabilities.

## Problem Statement

Position tracking is a critical component of any trading system, but several edge cases can lead to incorrect position calculations:

1. **Position Reversals**: Transitioning from long to short positions (or vice versa) can lead to calculation errors if not handled carefully

2. **Order Fills with Slippage**: Partial fills or fills at different prices than expected require careful accounting

3. **Position Reconciliation**: Discrepancies between tracked positions and actual positions (from broker or exchange) need detection and resolution

4. **Complex Order Types**: Stop losses, take profits, and other complex order types require special handling

5. **Rounding Issues**: Precision errors in position calculations can accumulate over time

We need a position tracking system that:
- Handles all edge cases correctly
- Maintains accurate position state at all times
- Provides tools for position reconciliation
- Supports various order types and execution models
- Is robust against calculation errors

## Design Solution

### 1. Robust Position Tracking Model

The foundation of our solution is a comprehensive position model:

```python
from decimal import Decimal, getcontext
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import uuid

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
    
    This class maintains all position details including entries,
    exits, average price, realized P&L, and various metrics.
    """
    
    def __init__(self, symbol: str):
        """
        Initialize position.
        
        Args:
            symbol: Position symbol
        """
        self.symbol = symbol
        self.quantity = Decimal('0')
        self.average_price = Decimal('0')
        self.cost_basis = Decimal('0')
        self.realized_pnl = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.entries = []  # List of entry trades
        self.exits = []  # List of exit trades
        self.trades = []  # All trades in sequence
        self.current_side = PositionSide.FLAT
        self._zero = Decimal('0')
        
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > self._zero
        
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < self._zero
        
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == self._zero
        
    @property
    def direction(self) -> str:
        """Get position direction as string."""
        if self.is_long:
            return "LONG"
        elif self.is_short:
            return "SHORT"
        else:
            return "FLAT"
            
    @property
    def side(self) -> PositionSide:
        """Get position side enum."""
        return self.current_side
        
    @property
    def absolute_quantity(self) -> Decimal:
        """Get absolute quantity."""
        return abs(self.quantity)
        
    @property
    def market_value(self) -> Decimal:
        """
        Get position market value.
        
        Returns:
            Decimal: Position market value
            
        Note:
            For long positions, this is positive.
            For short positions, this is negative.
        """
        return self.quantity * self.average_price
        
    @property
    def absolute_market_value(self) -> Decimal:
        """Get absolute market value."""
        return abs(self.market_value)
        
    @property
    def entry_count(self) -> int:
        """Get number of entry trades."""
        return len(self.entries)
        
    @property
    def exit_count(self) -> int:
        """Get number of exit trades."""
        return len(self.exits)
        
    @property
    def trade_count(self) -> int:
        """Get total number of trades."""
        return len(self.trades)
        
    @property
    def net_quantity(self) -> Decimal:
        """Get net quantity from all trades."""
        return sum((t['quantity'] for t in self.trades), self._zero)
        
    def update(self, quantity: Union[Decimal, float, int], price: Union[Decimal, float, int],
              timestamp: Optional[datetime] = None, trade_id: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update position with a new trade.
        
        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Trade price
            timestamp: Trade timestamp
            trade_id: Optional trade ID
            metadata: Optional trade metadata
            
        Returns:
            Dict containing trade details and updated position
        """
        # Convert inputs to Decimal for precise calculation
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))
        timestamp = timestamp or datetime.now()
        trade_id = trade_id or str(uuid.uuid4())
        
        # Create trade record
        trade = {
            'trade_id': trade_id,
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Save previous state for PnL calculation
        prev_quantity = self.quantity
        prev_cost_basis = self.cost_basis
        prev_average_price = self.average_price
        
        # Handle position update based on current and new positions
        if self.is_flat:
            # Opening new position
            self.quantity = quantity
            self.average_price = price
            self.cost_basis = quantity * price
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
                # Use FIFO accounting for average price
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
                    realized_pnl = exit_proceeds - exit_cost
                    
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
                    realized_pnl = exit_proceeds - exit_cost
                    
                    # Update position
                    self.realized_pnl += realized_pnl
                    self.quantity = self._zero
                    self.cost_basis = self._zero
                    self.average_price = self._zero
                    self.current_side = PositionSide.FLAT
                    
                    # Record pnl in trade
                    trade['realized_pnl'] = realized_pnl
                    trade['exit_price'] = price
                    trade['entry_price'] = prev_average_price
                    trade['trade_type'] = 'EXIT'
                    self.exits.append(trade)
                    
                else:
                    # Position reversal (long to short)
                    # First handle full exit of long position
                    exit_cost = self.cost_basis
                    exit_proceeds = self.quantity * price
                    realized_pnl = exit_proceeds - exit_cost
                    
                    # Record exit trade
                    exit_trade = trade.copy()
                    exit_trade['quantity'] = -self.quantity
                    exit_trade['realized_pnl'] = realized_pnl
                    exit_trade['exit_price'] = price
                    exit_trade['entry_price'] = prev_average_price
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
                        'exit_quantity': self.quantity,
                        'entry_quantity': short_quantity,
                        'realized_pnl': realized_pnl
                    }
                    
        elif self.is_short:
            if quantity < self._zero:
                # Adding to short position
                new_quantity = self.quantity + quantity
                self.cost_basis += quantity * price
                self.average_price = self.cost_basis / new_quantity
                self.quantity = new_quantity
                trade['trade_type'] = 'ENTRY'
                self.entries.append(trade)
                
            elif quantity > self._zero:
                # Reducing or closing short position
                abs_current_quantity = abs(self.quantity)
                
                if quantity < abs_current_quantity:
                    # Partial exit
                    exit_cost = quantity * self.average_price
                    exit_proceeds = quantity * price
                    realized_pnl = exit_cost - exit_proceeds  # Reversed for short positions
                    
                    # Update position
                    self.realized_pnl += realized_pnl
                    self.quantity += quantity  # Still negative but closer to zero
                    self.cost_basis = self.quantity * self.average_price
                    
                    # Record pnl in trade
                    trade['realized_pnl'] = realized_pnl
                    trade['exit_price'] = price
                    trade['entry_price'] = self.average_price
                    trade['trade_type'] = 'EXIT'
                    self.exits.append(trade)
                    
                elif quantity == abs_current_quantity:
                    # Full exit
                    exit_cost = abs(self.cost_basis)
                    exit_proceeds = quantity * price
                    realized_pnl = exit_cost - exit_proceeds  # Reversed for short positions
                    
                    # Update position
                    self.realized_pnl += realized_pnl
                    self.quantity = self._zero
                    self.cost_basis = self._zero
                    self.average_price = self._zero
                    self.current_side = PositionSide.FLAT
                    
                    # Record pnl in trade
                    trade['realized_pnl'] = realized_pnl
                    trade['exit_price'] = price
                    trade['entry_price'] = prev_average_price
                    trade['trade_type'] = 'EXIT'
                    self.exits.append(trade)
                    
                else:
                    # Position reversal (short to long)
                    # First handle full exit of short position
                    exit_cost = abs(self.cost_basis)
                    exit_proceeds = abs_current_quantity * price
                    realized_pnl = exit_cost - exit_proceeds  # Reversed for short positions
                    
                    # Record exit trade
                    exit_trade = trade.copy()
                    exit_trade['quantity'] = abs_current_quantity
                    exit_trade['realized_pnl'] = realized_pnl
                    exit_trade['exit_price'] = price
                    exit_trade['entry_price'] = prev_average_price
                    exit_trade['trade_type'] = 'EXIT'
                    exit_trade['trade_id'] = str(uuid.uuid4())
                    self.exits.append(exit_trade)
                    
                    # Calculate remaining quantity for long position
                    long_quantity = quantity - abs_current_quantity
                    
                    # Update position to new long position
                    self.realized_pnl += realized_pnl
                    self.quantity = long_quantity
                    self.average_price = price
                    self.cost_basis = long_quantity * price
                    self.current_side = PositionSide.LONG
                    
                    # Record entry trade for long position
                    entry_trade = trade.copy()
                    entry_trade['quantity'] = long_quantity
                    entry_trade['trade_type'] = 'ENTRY'
                    entry_trade['trade_id'] = str(uuid.uuid4())
                    self.entries.append(entry_trade)
                    
                    # Special case for position reversal
                    trade['trade_type'] = 'REVERSAL'
                    trade['reversal_details'] = {
                        'from_side': 'SHORT',
                        'to_side': 'LONG',
                        'exit_quantity': abs_current_quantity,
                        'entry_quantity': long_quantity,
                        'realized_pnl': realized_pnl
                    }
        
        # Add trade to complete history
        self.trades.append(trade)
        
        # Return updated trade details
        trade['updated_position'] = {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'cost_basis': self.cost_basis,
            'realized_pnl': self.realized_pnl,
            'side': self.direction
        }
        
        return trade
        
    def mark_to_market(self, price: Union[Decimal, float, int]) -> Dict[str, Any]:
        """
        Mark position to market at specified price.
        
        Args:
            price: Market price to mark position at
            
        Returns:
            Dict with updated position details
        """
        # Convert price to Decimal
        price = Decimal(str(price))
        
        # Calculate unrealized P&L
        if self.is_long:
            # Long position: current_value - cost_basis
            current_value = self.quantity * price
            self.unrealized_pnl = current_value - self.cost_basis
        elif self.is_short:
            # Short position: cost_basis - current_value
            current_value = self.quantity * price
            self.unrealized_pnl = abs(self.cost_basis) - abs(current_value)
        else:
            # Flat position
            self.unrealized_pnl = self._zero
            
        # Return updated position details
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'mark_price': price,
            'cost_basis': self.cost_basis,
            'market_value': self.quantity * price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'side': self.direction
        }
        
    def close(self, price: Union[Decimal, float, int], timestamp: Optional[datetime] = None,
             trade_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Close position at specified price.
        
        Args:
            price: Price to close position at
            timestamp: Trade timestamp
            trade_id: Optional trade ID
            metadata: Optional trade metadata
            
        Returns:
            Dict with closed position details
        """
        if self.is_flat:
            return {
                'symbol': self.symbol,
                'message': 'Position already flat',
                'realized_pnl': self.realized_pnl,
                'side': self.direction
            }
            
        # Create closing trade with opposite quantity
        close_quantity = -self.quantity
        return self.update(
            quantity=close_quantity,
            price=price,
            timestamp=timestamp,
            trade_id=trade_id,
            metadata=metadata
        )
        
    def reset(self) -> None:
        """Reset position to initial state."""
        self.quantity = self._zero
        self.average_price = self._zero
        self.cost_basis = self._zero
        self.realized_pnl = self._zero
        self.unrealized_pnl = self._zero
        self.entries = []
        self.exits = []
        self.trades = []
        self.current_side = PositionSide.FLAT
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary.
        
        Returns:
            Dict representation of position
        """
        return {
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'average_price': float(self.average_price),
            'cost_basis': float(self.cost_basis),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'side': self.direction,
            'trade_count': self.trade_count,
            'entry_count': self.entry_count,
            'exit_count': self.exit_count
        }
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get complete trade history.
        
        Returns:
            List of trade dictionaries
        """
        # Convert Decimal values to float for serialization
        result = []
        
        for trade in self.trades:
            trade_copy = trade.copy()
            for key, value in trade_copy.items():
                if isinstance(value, Decimal):
                    trade_copy[key] = float(value)
            result.append(trade_copy)
            
        return result
        
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate position state.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check that quantity matches trade history
        trade_sum = sum((t['quantity'] for t in self.trades), self._zero)
        if trade_sum != self.quantity:
            return False, f"Quantity mismatch: position={self.quantity}, trades={trade_sum}"
            
        # Check entries and exits
        entry_trades = [t for t in self.trades if t.get('trade_type') == 'ENTRY']
        exit_trades = [t for t in self.trades if t.get('trade_type') == 'EXIT']
        
        if len(entry_trades) != len(self.entries):
            return False, f"Entry count mismatch: position={len(self.entries)}, trades={len(entry_trades)}"
            
        if len(exit_trades) != len(self.exits):
            return False, f"Exit count mismatch: position={len(self.exits)}, trades={len(exit_trades)}"
            
        # Check if side matches quantity
        if self.quantity > self._zero and self.current_side != PositionSide.LONG:
            return False, f"Side mismatch: quantity={self.quantity}, side={self.current_side}"
            
        if self.quantity < self._zero and self.current_side != PositionSide.SHORT:
            return False, f"Side mismatch: quantity={self.quantity}, side={self.current_side}"
            
        if self.quantity == self._zero and self.current_side != PositionSide.FLAT:
            return False, f"Side mismatch: quantity={self.quantity}, side={self.current_side}"
            
        # All checks passed
        return True, None
```

### 2. Portfolio with Position Management

The portfolio component integrates position tracking:

```python
class Portfolio:
    """
    Portfolio management with robust position tracking.
    
    This class manages multiple positions and provides
    aggregated portfolio metrics and tools for reconciliation.
    """
    
    def __init__(self, initial_cash: Union[Decimal, float, int] = 100000):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Initial cash amount
        """
        self._cash = Decimal(str(initial_cash))
        self._initial_cash = Decimal(str(initial_cash))
        self._positions = {}  # symbol -> Position
        self._historical_equity = []  # List of equity snapshots
        self._trade_log = []  # Chronological list of all trades
        self._zero = Decimal('0')
        self._open_orders = {}  # order_id -> order
        
    @property
    def cash(self) -> Decimal:
        """Get available cash."""
        return self._cash
        
    @property
    def positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()
        
    @property
    def active_positions(self) -> Dict[str, Position]:
        """Get active (non-zero) positions."""
        return {s: p for s, p in self._positions.items() if not p.is_flat}
        
    @property
    def long_positions(self) -> Dict[str, Position]:
        """Get long positions."""
        return {s: p for s, p in self._positions.items() if p.is_long}
        
    @property
    def short_positions(self) -> Dict[str, Position]:
        """Get short positions."""
        return {s: p for s, p in self._positions.items() if p.is_short}
        
    @property
    def position_count(self) -> int:
        """Get active position count."""
        return len(self.active_positions)
        
    @property
    def long_exposure(self) -> Decimal:
        """Get total long exposure."""
        return sum((p.market_value for p in self.long_positions.values()), self._zero)
        
    @property
    def short_exposure(self) -> Decimal:
        """Get total short exposure."""
        return sum((p.market_value for p in self.short_positions.values()), self._zero)
        
    @property
    def net_exposure(self) -> Decimal:
        """Get net market exposure."""
        return self.long_exposure + self.short_exposure
        
    @property
    def gross_exposure(self) -> Decimal:
        """Get gross market exposure."""
        return sum((p.absolute_market_value for p in self.active_positions.values()), self._zero)
        
    @property
    def realized_pnl(self) -> Decimal:
        """Get total realized P&L."""
        return sum((p.realized_pnl for p in self._positions.values()), self._zero)
        
    @property
    def unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L."""
        return sum((p.unrealized_pnl for p in self._positions.values()), self._zero)
        
    @property
    def total_pnl(self) -> Decimal:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
        
    @property
    def portfolio_value(self) -> Decimal:
        """Get total portfolio value (cash + positions)."""
        return self._cash + self.net_exposure
        
    @property
    def equity(self) -> Decimal:
        """Get current equity value (synonym for portfolio_value)."""
        return self.portfolio_value
        
    def get_position(self, symbol: str) -> Position:
        """
        Get position for a symbol.
        
        Args:
            symbol: Position symbol
            
        Returns:
            Position object (creates new one if doesn't exist)
        """
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol)
            
        return self._positions[symbol]
        
    def update_position(self, symbol: str, quantity: Union[Decimal, float, int],
                       price: Union[Decimal, float, int], timestamp: Optional[datetime] = None,
                       trade_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update position with a new trade.
        
        Args:
            symbol: Position symbol
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp
            trade_id: Optional trade ID
            metadata: Optional trade metadata
            
        Returns:
            Dict with trade details
        """
        # Convert inputs to Decimal
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))
        timestamp = timestamp or datetime.now()
        
        # Get position
        position = self.get_position(symbol)
        
        # Update cash
        trade_value = quantity * price
        self._cash -= trade_value
        
        # Update position
        trade = position.update(
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            trade_id=trade_id,
            metadata=metadata
        )
        
        # Add to trade log
        self._trade_log.append(trade)
        
        # Update equity history
        self._update_equity_history(timestamp)
        
        return trade
        
    def place_order(self, order_type: str, symbol: str, quantity: Union[Decimal, float, int],
                   price: Optional[Union[Decimal, float, int]] = None, 
                   timestamp: Optional[datetime] = None,
                   order_id: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            order_type: Order type (MARKET, LIMIT, etc.)
            symbol: Symbol to trade
            quantity: Order quantity
            price: Order price (required for some order types)
            timestamp: Order timestamp
            order_id: Optional order ID
            metadata: Optional order metadata
            
        Returns:
            Dict with order details
        """
        # Generate order ID if not provided
        order_id = order_id or str(uuid.uuid4())
        timestamp = timestamp or datetime.now()
        
        # Create order
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'order_type': order_type,
            'quantity': Decimal(str(quantity)),
            'price': Decimal(str(price)) if price is not None else None,
            'timestamp': timestamp,
            'status': 'PENDING',
            'executed_quantity': self._zero,
            'remaining_quantity': Decimal(str(quantity)),
            'fills': [],
            'metadata': metadata or {}
        }
        
        # Store order
        self._open_orders[order_id] = order
        
        return order
        
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict with cancelled order details
        """
        if order_id not in self._open_orders:
            return {'error': f"Order not found: {order_id}"}
            
        order = self._open_orders[order_id]
        
        # Check if order can be cancelled
        if order['status'] in ('FILLED', 'CANCELLED', 'REJECTED'):
            return {'error': f"Cannot cancel order in status: {order['status']}"}
            
        # Update order status
        order['status'] = 'CANCELLED'
        
        # Remove from open orders
        if order_id in self._open_orders:
            del self._open_orders[order_id]
            
        return order
        
    def execute_order(self, order_id: str, executed_quantity: Union[Decimal, float, int],
                     executed_price: Union[Decimal, float, int], timestamp: Optional[datetime] = None,
                     fill_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an open order (fully or partially).
        
        Args:
            order_id: Order ID to execute
            executed_quantity: Quantity executed
            executed_price: Execution price
            timestamp: Execution timestamp
            fill_id: Optional fill ID
            metadata: Optional fill metadata
            
        Returns:
            Dict with execution details
        """
        if order_id not in self._open_orders:
            return {'error': f"Order not found: {order_id}"}
            
        order = self._open_orders[order_id]
        
        # Convert inputs to Decimal
        executed_quantity = Decimal(str(executed_quantity))
        executed_price = Decimal(str(executed_price))
        timestamp = timestamp or datetime.now()
        fill_id = fill_id or str(uuid.uuid4())
        
        # Check if order can be executed
        if order['status'] in ('FILLED', 'CANCELLED', 'REJECTED'):
            return {'error': f"Cannot execute order in status: {order['status']}"}
            
        # Create fill record
        fill = {
            'fill_id': fill_id,
            'order_id': order_id,
            'symbol': order['symbol'],
            'quantity': executed_quantity,
            'price': executed_price,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Update order
        order['executed_quantity'] += executed_quantity
        order['remaining_quantity'] -= executed_quantity
        order['fills'].append(fill)
        
        # Check if order is completely filled
        if order['remaining_quantity'] <= self._zero:
            order['status'] = 'FILLED'
            
            # Remove from open orders
            if order_id in self._open_orders:
                del self._open_orders[order_id]
        else:
            order['status'] = 'PARTIALLY_FILLED'
            
        # Update position
        trade = self.update_position(
            symbol=order['symbol'],
            quantity=executed_quantity,
            price=executed_price,
            timestamp=timestamp,
            trade_id=fill_id,
            metadata={'order_id': order_id, 'fill': True, **metadata or {}}
        )
        
        # Link fill to trade
        fill['trade'] = trade
        
        return {
            'order': order,
            'fill': fill,
            'trade': trade
        }
        
    def mark_to_market(self, prices: Dict[str, Union[Decimal, float, int]],
                     timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Mark all positions to market.
        
        Args:
            prices: Dict mapping symbols to prices
            timestamp: Marking timestamp
            
        Returns:
            Dict with marked positions and portfolio value
        """
        timestamp = timestamp or datetime.now()
        marked_positions = {}
        
        # Mark each position
        for symbol, position in self._positions.items():
            if symbol in prices:
                price = prices[symbol]
                marked_positions[symbol] = position.mark_to_market(price)
                
        # Update equity history
        self._update_equity_history(timestamp)
        
        return {
            'timestamp': timestamp,
            'positions': marked_positions,
            'portfolio_value': self.portfolio_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'cash': self._cash
        }
        
    def _update_equity_history(self, timestamp: datetime) -> None:
        """
        Update equity history.
        
        Args:
            timestamp: Update timestamp
        """
        equity_point = {
            'timestamp': timestamp,
            'cash': self._cash,
            'position_value': self.net_exposure,
            'portfolio_value': self.portfolio_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'position_count': self.position_count
        }
        
        self._historical_equity.append(equity_point)
        
    def get_equity_history(self) -> List[Dict[str, Any]]:
        """
        Get equity history.
        
        Returns:
            List of equity history points
        """
        return self._historical_equity
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Returns:
            List of all trades
        """
        return self._trade_log
        
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Returns:
            List of open orders
        """
        return list(self._open_orders.values())
        
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._cash = self._initial_cash
        
        # Reset positions
        for position in self._positions.values():
            position.reset()
            
        # Clear collections
        self._positions.clear()
        self._historical_equity.clear()
        self._trade_log.clear()
        self._open_orders.clear()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert portfolio to dictionary.
        
        Returns:
            Dict representation of portfolio
        """
        return {
            'cash': float(self._cash),
            'positions': {s: p.to_dict() for s, p in self._positions.items()},
            'portfolio_value': float(self.portfolio_value),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.total_pnl),
            'position_count': self.position_count,
            'long_exposure': float(self.long_exposure),
            'short_exposure': float(self.short_exposure),
            'net_exposure': float(self.net_exposure),
            'gross_exposure': float(self.gross_exposure),
            'open_order_count': len(self._open_orders)
        }
```

### 3. Position Reconciliation Utility

To verify positions against external sources:

```python
class PositionReconciliation:
    """
    Position reconciliation utility.
    
    This class reconciles internally tracked positions with
    external sources (broker, exchange, etc.) and resolves discrepancies.
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize reconciliation utility.
        
        Args:
            portfolio: Portfolio to reconcile
        """
        self._portfolio = portfolio
        
    def reconcile(self, external_positions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile portfolio positions with external positions.
        
        Args:
            external_positions: Dict mapping symbols to position details
            
        Returns:
            Dict with reconciliation results
        """
        results = {
            'matched': [],
            'mismatched': [],
            'missing_internal': [],
            'missing_external': [],
            'adjustments': []
        }
        
        # Get internal positions
        internal_positions = self._portfolio.positions
        
        # Check each internal position against external
        for symbol, position in internal_positions.items():
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
            if symbol not in internal_positions or internal_positions[symbol].is_flat:
                external_qty = Decimal(str(external.get('quantity', 0)))
                
                if external_qty != 0:
                    results['missing_internal'].append({
                        'symbol': symbol,
                        'external': float(external_qty)
                    })
                    
        return results
        
    def apply_adjustments(self, adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply position adjustments.
        
        Args:
            adjustments: List of adjustment specifications
            
        Returns:
            Dict with adjustment results
        """
        results = []
        
        for adjustment in adjustments:
            symbol = adjustment.get('symbol')
            target_quantity = Decimal(str(adjustment.get('target_quantity', 0)))
            price = Decimal(str(adjustment.get('price', 0)))
            reason = adjustment.get('reason', 'Manual adjustment')
            
            # Get current position
            position = self._portfolio.get_position(symbol)
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
            
            trade = self._portfolio.update_position(
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
        
    def reconcile_and_adjust(self, external_positions: Dict[str, Dict[str, Any]],
                           price_source: Dict[str, Union[Decimal, float, int]],
                           auto_adjust: bool = False,
                           adjustment_reason: str = 'Auto-reconciliation') -> Dict[str, Any]:
        """
        Reconcile positions and optionally apply automatic adjustments.
        
        Args:
            external_positions: Dict mapping symbols to position details
            price_source: Dict mapping symbols to current prices
            auto_adjust: Whether to automatically apply adjustments
            adjustment_reason: Reason for auto-adjustments
            
        Returns:
            Dict with reconciliation and adjustment results
        """
        # Reconcile positions
        reconciliation = self.reconcile(external_positions)
        
        if not auto_adjust:
            # Return reconciliation without adjustments
            return {
                'reconciliation': reconciliation,
                'adjustments': None
            }
            
        # Prepare adjustments
        adjustments = []
        
        # Handle mismatches
        for mismatch in reconciliation['mismatched']:
            symbol = mismatch['symbol']
            external_qty = Decimal(str(mismatch['external']))
            
            if symbol in price_source:
                price = price_source[symbol]
                
                adjustments.append({
                    'symbol': symbol,
                    'target_quantity': external_qty,
                    'price': price,
                    'reason': f"{adjustment_reason} - Quantity mismatch"
                })
                
        # Handle missing internal positions
        for missing in reconciliation['missing_internal']:
            symbol = missing['symbol']
            external_qty = Decimal(str(missing['external']))
            
            if symbol in price_source:
                price = price_source[symbol]
                
                adjustments.append({
                    'symbol': symbol,
                    'target_quantity': external_qty,
                    'price': price,
                    'reason': f"{adjustment_reason} - Missing internal position"
                })
                
        # Handle missing external positions
        for missing in reconciliation['missing_external']:
            symbol = missing['symbol']
            
            if symbol in price_source:
                price = price_source[symbol]
                
                adjustments.append({
                    'symbol': symbol,
                    'target_quantity': 0,  # Close position
                    'price': price,
                    'reason': f"{adjustment_reason} - Missing external position"
                })
                
        # Apply adjustments
        adjustment_results = self.apply_adjustments(adjustments)
        
        return {
            'reconciliation': reconciliation,
            'adjustments': adjustment_results
        }
```

### 4. Robust Trade Processing

Enhanced trade processing with error handling:

```python
class TradeProcessor:
    """
    Trade processor with robust error handling.
    
    This class processes trades and order fills with comprehensive
    validation and error handling for all edge cases.
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize trade processor.
        
        Args:
            portfolio: Portfolio to process trades for
        """
        self._portfolio = portfolio
        self._error_handlers = {}  # error_type -> handler_function
        self._validation_rules = []  # List of validation functions
        self._default_error_handler = None
        self._trade_processors = {}  # trade_type -> processor_function
        
    def register_error_handler(self, error_type: str, handler: Callable) -> None:
        """
        Register error handler for specific error type.
        
        Args:
            error_type: Error type to handle
            handler: Handler function
        """
        self._error_handlers[error_type] = handler
        
    def set_default_error_handler(self, handler: Callable) -> None:
        """
        Set default error handler.
        
        Args:
            handler: Default handler function
        """
        self._default_error_handler = handler
        
    def register_validation_rule(self, rule: Callable) -> None:
        """
        Register validation rule.
        
        Args:
            rule: Validation function
        """
        self._validation_rules.append(rule)
        
    def register_trade_processor(self, trade_type: str, processor: Callable) -> None:
        """
        Register trade processor for specific trade type.
        
        Args:
            trade_type: Trade type to process
            processor: Processor function
        """
        self._trade_processors[trade_type] = processor
        
    def validate_trade(self, trade: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate trade before processing.
        
        Args:
            trade: Trade to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['symbol', 'quantity', 'price']
        for field in required_fields:
            if field not in trade:
                return False, f"Missing required field: {field}"
                
        # Apply validation rules
        for rule in self._validation_rules:
            is_valid, error = rule(trade, self._portfolio)
            if not is_valid:
                return False, error
                
        return True, None
        
    def process_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trade.
        
        Args:
            trade: Trade to process
            
        Returns:
            Dict with processing results
        """
        try:
            # Validate trade
            is_valid, error = self.validate_trade(trade)
            if not is_valid:
                return self._handle_error('validation_error', error, trade)
                
            # Extract trade type
            trade_type = trade.get('trade_type', 'standard')
            
            # Process based on trade type
            if trade_type in self._trade_processors:
                # Use specialized processor
                return self._trade_processors[trade_type](trade, self._portfolio)
            else:
                # Use standard processing
                symbol = trade['symbol']
                quantity = Decimal(str(trade['quantity']))
                price = Decimal(str(trade['price']))
                timestamp = trade.get('timestamp', datetime.now())
                trade_id = trade.get('trade_id')
                metadata = trade.get('metadata', {})
                
                # Process trade in portfolio
                result = self._portfolio.update_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_id=trade_id,
                    metadata=metadata
                )
                
                return {
                    'success': True,
                    'trade': result
                }
                
        except Exception as e:
            return self._handle_error('processing_error', str(e), trade)
            
    def process_fill(self, fill: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an order fill.
        
        Args:
            fill: Fill to process
            
        Returns:
            Dict with processing results
        """
        try:
            # Extract order ID
            order_id = fill.get('order_id')
            if not order_id:
                return self._handle_error('missing_order_id', "Fill missing order ID", fill)
                
            # Process execution
            result = self._portfolio.execute_order(
                order_id=order_id,
                executed_quantity=Decimal(str(fill['quantity'])),
                executed_price=Decimal(str(fill['price'])),
                timestamp=fill.get('timestamp', datetime.now()),
                fill_id=fill.get('fill_id'),
                metadata=fill.get('metadata', {})
            )
            
            # Check for errors
            if 'error' in result:
                return self._handle_error('execution_error', result['error'], fill)
                
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            return self._handle_error('processing_error', str(e), fill)
            
    def _handle_error(self, error_type: str, error_message: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle processing error.
        
        Args:
            error_type: Type of error
            error_message: Error message
            data: Data being processed when error occurred
            
        Returns:
            Dict with error details
        """
        # Use specific handler if available
        if error_type in self._error_handlers:
            return self._error_handlers[error_type](error_message, data)
            
        # Use default handler if available
        if self._default_error_handler:
            return self._default_error_handler(error_type, error_message, data)
            
        # Default error response
        return {
            'success': False,
            'error_type': error_type,
            'error_message': error_message,
            'data': data
        }
        
    def process_batch(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process batch of trades.
        
        Args:
            trades: List of trades to process
            
        Returns:
            Dict with batch processing results
        """
        results = []
        errors = []
        
        for trade in trades:
            result = self.process_trade(trade)
            results.append(result)
            
            if not result.get('success', False):
                errors.append(result)
                
        return {
            'success': len(errors) == 0,
            'total': len(trades),
            'successful': len(trades) - len(errors),
            'errors': len(errors),
            'results': results,
            'error_details': errors
        }
```

## Edge Case Handling

### 1. Position Reversal Edge Case

The position model carefully handles position reversals:

```python
# Position reversal (long to short)
elif abs_quantity > self.quantity:
    # First handle full exit of long position
    exit_cost = self.cost_basis
    exit_proceeds = self.quantity * price
    realized_pnl = exit_proceeds - exit_cost
    
    # Record exit trade
    exit_trade = trade.copy()
    exit_trade['quantity'] = -self.quantity
    exit_trade['realized_pnl'] = realized_pnl
    exit_trade['exit_price'] = price
    exit_trade['entry_price'] = prev_average_price
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
        'exit_quantity': self.quantity,
        'entry_quantity': short_quantity,
        'realized_pnl': realized_pnl
    }
```

### 2. Partial Fill Handling

The portfolio handles partial fills correctly:

```python
def execute_order(self, order_id: str, executed_quantity: Union[Decimal, float, int],
                executed_price: Union[Decimal, float, int], timestamp: Optional[datetime] = None,
                fill_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute an open order (fully or partially)."""
    order = self._open_orders[order_id]
    
    # Update order
    order['executed_quantity'] += executed_quantity
    order['remaining_quantity'] -= executed_quantity
    order['fills'].append(fill)
    
    # Check if order is completely filled
    if order['remaining_quantity'] <= self._zero:
        order['status'] = 'FILLED'
        
        # Remove from open orders
        if order_id in self._open_orders:
            del self._open_orders[order_id]
    else:
        order['status'] = 'PARTIALLY_FILLED'
```

### 3. Precise Decimal Calculations

The system uses Decimal for precise calculations:

```python
# Set decimal precision for position calculations
getcontext().prec = 28

# Convert inputs to Decimal for precise calculation
quantity = Decimal(str(quantity))
price = Decimal(str(price))
```

### 4. Rounding and Zero Handling

Special handling for rounding and zero values:

```python
def validate(self) -> Tuple[bool, Optional[str]]:
    """Validate position state."""
    # Check quantity is near zero for flat positions
    if abs(self.quantity) < Decimal('0.0000001'):
        # Reset to exact zero
        self.quantity = self._zero
        self.cost_basis = self._zero
        self.average_price = self._zero
        self.current_side = PositionSide.FLAT
```

### 5. Position Reconciliation

Comprehensive reconciliation of positions:

```python
def reconcile(self, external_positions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Reconcile portfolio positions with external positions."""
    results = {
        'matched': [],
        'mismatched': [],
        'missing_internal': [],
        'missing_external': [],
        'adjustments': []
    }
    
    # Detailed reconciliation logic
    # ...
```

## Implementation Strategy

### 1. Position Class Implementation

1. Create robust position tracking class:
   - Implement precise decimal calculations
   - Handle position reversals and edge cases
   - Track all trade history

2. Add validation and verification:
   - Add position validation rules
   - Create summary and reporting tools

### 2. Portfolio Integration

1. Update portfolio with enhanced position tracking:
   - Use robust position tracking for all positions
   - Implement comprehensive portfolio metrics
   - Add exposure and risk tracking

2. Add order and execution handling:
   - Implement order placement and tracking
   - Create execution and fill processing
   - Handle partial fills and cancellations

### 3. Reconciliation Tools

1. Create position reconciliation utility:
   - Implement position comparison with external sources
   - Create adjustment mechanisms for discrepancies
   - Add reporting for reconciliation results

2. Add error handling and recovery:
   - Implement error detection for position anomalies
   - Create recovery procedures for data issues
   - Build logging for position events

## Testing Strategy

### 1. Unit Tests for Position Tracking

```python
def test_position_tracking_edge_cases():
    """Test position tracking edge cases."""
    position = Position("AAPL")
    
    # Test position reversal
    position.update(quantity=100, price=150)
    assert position.quantity == 100
    assert position.average_price == 150
    assert position.is_long
    
    # Reverse position (long to short)
    position.update(quantity=-150, price=160)
    assert position.quantity == -50
    assert position.is_short
    assert position.realized_pnl == 1000  # 100 * (160 - 150)
    
    # Verify trade history
    trades = position.get_trade_history()
    assert len(trades) == 2
    assert trades[1]['trade_type'] == 'REVERSAL'
    
    # Test zero position
    position.update(quantity=50, price=170)
    assert position.is_flat
    assert position.quantity == 0
```

### 2. Portfolio Edge Case Tests

```python
def test_portfolio_edge_cases():
    """Test portfolio edge cases."""
    portfolio = Portfolio(initial_cash=10000)
    
    # Test partial fills
    order = portfolio.place_order(
        order_type="LIMIT",
        symbol="AAPL",
        quantity=100,
        price=150
    )
    
    # Execute partial fill
    execution1 = portfolio.execute_order(
        order_id=order['order_id'],
        executed_quantity=60,
        executed_price=150
    )
    
    assert order['status'] == 'PARTIALLY_FILLED'
    assert order['executed_quantity'] == 60
    assert order['remaining_quantity'] == 40
    
    # Execute rest of order
    execution2 = portfolio.execute_order(
        order_id=order['order_id'],
        executed_quantity=40,
        executed_price=151
    )
    
    assert order['status'] == 'FILLED'
    
    # Verify position average price is weighted correctly
    position = portfolio.get_position("AAPL")
    assert position.quantity == 100
    assert 150 < position.average_price < 151
```

### 3. Reconciliation Tests

```python
def test_position_reconciliation():
    """Test position reconciliation."""
    portfolio = Portfolio(initial_cash=10000)
    
    # Create positions
    portfolio.update_position(symbol="AAPL", quantity=100, price=150)
    portfolio.update_position(symbol="MSFT", quantity=-50, price=200)
    portfolio.update_position(symbol="GOOGL", quantity=25, price=2500)
    
    # External positions (with discrepancy)
    external_positions = {
        "AAPL": {"quantity": 100},  # Matches
        "MSFT": {"quantity": -30},  # Mismatched
        "AMZN": {"quantity": 10}    # Missing internal
        # GOOGL is missing external
    }
    
    # Create reconciliation utility
    reconciliation = PositionReconciliation(portfolio)
    
    # Reconcile positions
    results = reconciliation.reconcile(external_positions)
    
    assert len(results['matched']) == 1
    assert len(results['mismatched']) == 1
    assert len(results['missing_internal']) == 1
    assert len(results['missing_external']) == 1
    
    # Apply adjustments
    price_source = {
        "AAPL": 155,
        "MSFT": 205,
        "GOOGL": 2550,
        "AMZN": 3000
    }
    
    result = reconciliation.reconcile_and_adjust(
        external_positions=external_positions,
        price_source=price_source,
        auto_adjust=True
    )
    
    # Verify adjustments
    assert len(result['adjustments']['adjustments']) == 3
    
    # Verify portfolio now matches external positions
    assert portfolio.get_position("MSFT").quantity == -30
    assert portfolio.get_position("AMZN").quantity == 10
    assert portfolio.get_position("GOOGL").quantity == 0
```

### 4. Performance Stress Tests

```python
def test_performance_stress():
    """Test performance under stress."""
    portfolio = Portfolio(initial_cash=1000000)
    
    # Generate many trades
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
    num_trades = 10000
    
    import random
    import time
    
    start_time = time.time()
    
    for i in range(num_trades):
        symbol = random.choice(symbols)
        quantity = random.choice([-100, -50, -10, 10, 50, 100])
        price = random.uniform(100, 1000)
        
        portfolio.update_position(
            symbol=symbol,
            quantity=quantity,
            price=price
        )
        
    end_time = time.time()
    
    # Verify performance
    duration = end_time - start_time
    trades_per_second = num_trades / duration
    
    print(f"Processed {num_trades} trades in {duration:.2f} seconds")
    print(f"Trades per second: {trades_per_second:.2f}")
    
    # Verify final state
    for symbol in symbols:
        position = portfolio.get_position(symbol)
        trades = position.get_trade_history()
        
        # Verify that quantity matches trade history
        calculated_quantity = sum(t['quantity'] for t in trades)
        assert position.quantity == calculated_quantity
```

## Best Practices

### 1. Position Tracking Guidelines

- **Always use Decimal for financial calculations:**
  ```python
  quantity = Decimal(str(quantity))
  price = Decimal(str(price))
  cost_basis = quantity * price
  ```

- **Track all trade history for audit and verification:**
  ```python
  # Add to trade history
  self.trades.append(trade)
  
  # Add to entries or exits as appropriate
  if trade['trade_type'] == 'ENTRY':
      self.entries.append(trade)
  elif trade['trade_type'] == 'EXIT':
      self.exits.append(trade)
  ```

- **Regularly validate position state:**
  ```python
  def validate_portfolio():
      """Validate all positions in portfolio."""
      for symbol, position in portfolio.positions.items():
          is_valid, error = position.validate()
          if not is_valid:
              log.error(f"Position validation failed for {symbol}: {error}")
  ```

### 2. Reconciliation Guidelines

- **Reconcile positions regularly with external sources:**
  ```python
  # Daily reconciliation
  def daily_reconciliation():
      external_positions = broker_api.get_positions()
      market_prices = market_data_api.get_current_prices()
      
      reconciliation = PositionReconciliation(portfolio)
      results = reconciliation.reconcile_and_adjust(
          external_positions=external_positions,
          price_source=market_prices,
          auto_adjust=False  # Review discrepancies before adjusting
      )
      
      # Log results
      log.info(f"Daily reconciliation: {len(results['matched'])} positions matched, "
              f"{len(results['mismatched'])} mismatched, "
              f"{len(results['missing_internal'])} missing internal, "
              f"{len(results['missing_external'])} missing external")
  ```

- **Document all position adjustments:**
  ```python
  # Adjustment metadata
  metadata = {
      'adjustment': True,
      'reason': 'Broker reconciliation',
      'previous_quantity': float(current_quantity),
      'adjustment_date': datetime.now().isoformat(),
      'authorized_by': 'System',
      'reconciliation_id': reconciliation_id
  }
  ```

### 3. Edge Case Handling Guidelines

- **Handle position reversals explicitly:**
  ```python
  # Check for position reversal
  if (position.is_long and quantity < 0 and abs(quantity) > position.quantity) or 
     (position.is_short and quantity > 0 and abs(quantity) > abs(position.quantity)):
      log.info(f"Position reversal detected for {symbol}")
      # Process reversal...
  ```

- **Validate inputs to prevent errors:**
  ```python
  # Validate trade before processing
  def validate_trade(trade):
      if 'symbol' not in trade:
          return False, "Missing symbol"
      
      if 'quantity' not in trade:
          return False, "Missing quantity"
      
      if 'price' not in trade:
          return False, "Missing price"
      
      if not isinstance(trade['quantity'], (int, float, Decimal)) or trade['quantity'] == 0:
          return False, "Invalid quantity"
      
      if not isinstance(trade['price'], (int, float, Decimal)) or trade['price'] <= 0:
          return False, "Invalid price"
      
      return True, None
  ```

## Conclusion

The robust position tracking system presented in this document provides a solid foundation for accurate trading operations in the ADMF-Trader system. By implementing comprehensive position tracking with careful handling of edge cases, the system can reliably manage complex trading scenarios including position reversals, partial fills, and reconciliation with external sources.

The design emphasizes accuracy, auditability, and recoverability, ensuring that the system can handle real-world trading scenarios while maintaining correct position state. The reconciliation tools enable detection and resolution of discrepancies, maintaining the integrity of the system even when external factors introduce inconsistencies.

By following the implementation strategy and best practices outlined in this document, the ADMF-Trader system will have a robust and reliable position tracking foundation for its trading operations.