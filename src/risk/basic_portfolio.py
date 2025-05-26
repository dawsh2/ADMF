# src/risk/basic_portfolio.py
import logging
import datetime 
from typing import Dict, Any, Optional, List, Tuple 
import uuid

from ..core.component_base import ComponentBase
from ..core.event import Event, EventType
from ..core.exceptions import ComponentError, DependencyNotFoundError 

class BasicPortfolio(ComponentBase):
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Initialize internal state (no external dependencies)
        
        # Configuration parameters (will be set in initialize)
        self.initial_cash: float = 100000.0
        self.current_cash: float = 100000.0
        
        # Stores open positions:
        # {symbol: {
        #     'quantity': float,  # Signed: + for long, - for short
        #     'cost_basis': float, # For longs: total cost of shares. For shorts: negative of proceeds (liability)
        #     'avg_entry_price': float, # Weighted average price of current open position
        #     'entry_timestamp': Optional[datetime.datetime], # Timestamp of initial entry into current overall position
        #     'trade_id': str, # Unique ID for the entire lifecycle of holding this symbol (flips get new ID)
        #     'current_segment_entry_price': float, # Entry price for the current regime segment
        #     'current_segment_regime': str,
        #     'initial_entry_regime_for_trade': str # Regime when this specific trade_id was initiated
        # }}
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self._trade_log: List[Dict[str, Any]] = []
        
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.current_holdings_value: float = 0.0 # Market value of open positions
        self.current_total_value: float = self.initial_cash
        self._last_bar_prices: Dict[str, float] = {}

        self._regime_detector: Optional[Any] = None
        self.regime_detector_key: str = "MyPrimaryRegimeDetector"
        self._current_market_regime: Optional[str] = "default"

        self._portfolio_value_history: List[Tuple[Optional[datetime.datetime], float]] = []
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Load configuration
        self.initial_cash = self.get_specific_config('initial_cash', 100000.0)
        self.current_cash = self.initial_cash
        self.regime_detector_key = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        
        self.logger.info(f"BasicPortfolio '{self.instance_name}' initialized with initial cash: {self.initial_cash:.2f}")
    
    def get_specific_config(self, key: str, default=None):
        """Helper method to get configuration values."""
        # First try component_config set by ComponentBase
        if hasattr(self, 'component_config') and self.component_config:
            value = self.component_config.get(key, None)
            if value is not None:
                return value
        
        # Fall back to config_loader
        if not self.config_loader:
            return default
        config_key = self.config_key or self.instance_name
        config = self.config_loader.get_component_config(config_key)
        return config.get(key, default) if config else default
        
    def reset(self):
        """Reset the portfolio to its initial state for a fresh backtest run."""
        trade_count_before = len(self._trade_log) if hasattr(self, '_trade_log') else 0
        self.logger.debug(f"Resetting portfolio '{self.instance_name}' - had {trade_count_before} trades")
        
        # DEBUG: Add stack trace to identify what's calling reset
        import traceback
        stack_trace = ''.join(traceback.format_stack()[-5:])  # Last 5 stack frames
        self.logger.debug(f"Portfolio reset called from: {stack_trace.split()[-1] if stack_trace else 'unknown'}")
        
        self.logger.info(f"Resetting portfolio '{self.instance_name}' to initial state")
        
        # Unsubscribe from events first to prevent duplicate subscriptions
        if self.event_bus:
            try:
                self.event_bus.unsubscribe(EventType.FILL, self.on_fill)
                self.event_bus.unsubscribe(EventType.BAR, self.on_bar)
                self.event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.debug(f"'{self.instance_name}' unsubscribed from events during reset.")
            except Exception as e:
                self.logger.warning(f"Error unsubscribing from events during reset: {e}")
        
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
        
        self.logger.info(f"Portfolio '{self.instance_name}' reset successfully. Cash: {self.current_cash:.2f}, Total Value: {self.current_total_value:.2f}")

    def setup(self):
        """Set up the portfolio."""
        self.logger.info(f"Setting up BasicPortfolio '{self.instance_name}'...")
        if self.event_bus:
            self.event_bus.subscribe(EventType.FILL, self.on_fill)
            self.event_bus.subscribe(EventType.BAR, self.on_bar)
            self.event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.instance_name}' subscribed to FILL, BAR, and CLASSIFICATION events.")
        else:
            self.logger.error(f"Event bus not available for '{self.instance_name}'. Cannot subscribe to events.")
            # Mark as failed
            return

        try:
            self._regime_detector = self.container.resolve(self.regime_detector_key)
            if self._regime_detector and hasattr(self._regime_detector, 'get_current_classification') and callable(getattr(self._regime_detector, 'get_current_classification')):
                self.logger.info(f"Successfully resolved and linked RegimeDetector: {self._regime_detector.name}")
                initial_regime = self._regime_detector.get_current_classification()
                self._current_market_regime = initial_regime if initial_regime else "default"
            else:
                self.logger.warning(f"Failed to resolve or use RegimeDetector '{self.regime_detector_key}'. Defaulting regime.")
                self._current_market_regime = "default"
        except DependencyNotFoundError:
            self.logger.warning(f"Dependency '{self.regime_detector_key}' (RegimeDetector) not found. Defaulting regime.")
            self._current_market_regime = "default"
        except Exception as e:
            self.logger.error(f"Error resolving RegimeDetector '{self.regime_detector_key}': {e}", exc_info=True)
            self._current_market_regime = "default"
        self.logger.info(f"Initial market regime set to: {self._current_market_regime}")

        self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc))
        self.logger.info(f"BasicPortfolio '{self.instance_name}' setup complete.")

    def start(self):
        """Start the portfolio."""
        super().start()
        
        # Parent class handles state checking
            
        # Ensure we're subscribed to events (needed for restarts)
        if self.event_bus:
            self.event_bus.subscribe(EventType.FILL, self.on_fill)
            self.event_bus.subscribe(EventType.BAR, self.on_bar)
            self.event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.debug(f"'{self.instance_name}' re-subscribed to FILL, BAR, and CLASSIFICATION events on start/restart")
        
        # Component is now running
        self.logger.info(f"BasicPortfolio '{self.instance_name}' started.")
        if not self._portfolio_value_history:
             self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc))

    def stop(self):
        """Stop the portfolio."""
        self.logger.info(f"Stopping {self.instance_name}...")
        self._log_final_performance_summary()
        if self.event_bus:
            try:
                self.event_bus.unsubscribe(EventType.FILL, self.on_fill)
                self.event_bus.unsubscribe(EventType.BAR, self.on_bar)
                self.event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.debug(f"'{self.instance_name}' unsubscribed from events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.instance_name}': {e}", exc_info=True)
        super().stop()
        self.logger.debug(f"{self.instance_name} stopped.")

    def on_classification_change(self, event: Event):
        payload = event.payload
        if not isinstance(payload, dict): return
        new_regime = payload.get('classification')
        timestamp = payload.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
        if new_regime and new_regime != self._current_market_regime:
            self.logger.info(f"Market regime changed from '{self._current_market_regime}' to '{new_regime}' at {timestamp} for '{self.instance_name}'.")
            self._current_market_regime = new_regime
        elif new_regime:
            # Log when we receive classification events even if regime hasn't changed
            self.logger.debug(f"Classification event received with same regime '{new_regime}' at {timestamp} for '{self.instance_name}'.")

    def _get_current_regime(self) -> str:
        return self._current_market_regime or "default"

    def on_fill(self, event: Event):
        self.logger.debug(f"{self.instance_name} received FILL event")
        fill_data = event.payload
        fill_id = fill_data.get('fill_id', 'NO_ID')
        self.logger.debug(f"Processing FILL {fill_id} - current trades: {len(self._trade_log)}")
        if not (isinstance(fill_data, dict) and all(k in fill_data for k in ['symbol', 'timestamp', 'quantity_filled', 'fill_price', 'direction'])):
            self.logger.error(f"'{self.instance_name}' received incomplete or invalid FILL data: {fill_data}"); return

        symbol = fill_data['symbol']
        timestamp = fill_data['timestamp']
        quantity_filled = float(fill_data['quantity_filled'])
        fill_price = float(fill_data['fill_price'])
        commission = float(fill_data.get('commission', 0.0))
        fill_direction_is_buy = fill_data['direction'].upper() == 'BUY'

        # Record the pre-fill state for validation
        pre_fill_cash = self.current_cash
        
        self.current_cash -= commission
        self.logger.debug(f"Commission deducted: {commission:.2f}, Cash before fill: {pre_fill_cash:.2f}, Cash after commission: {self.current_cash:.2f}")
        
        # Log trade for debugging train/test performance
        side = "BUY" if fill_direction_is_buy else "SELL"
        self.logger.info(f"TRADE: {side} {quantity_filled} {symbol} @ ${fill_price:.2f}, Portfolio Value: ${self.current_total_value:.2f}")
        
        active_regime = self._get_current_regime()

        pos = self.open_positions.get(symbol)
        
        # Initialize new position if none exists for the symbol
        if not pos:
            pos = {
                'quantity': 0.0, 'avg_entry_price': 0.0, 'cost_basis': 0.0,
                'entry_timestamp': timestamp, 'trade_id': str(uuid.uuid4()),
                'current_segment_entry_price': fill_price,
                'current_segment_regime': active_regime,
                'initial_entry_regime_for_trade': active_regime
            }
            self.open_positions[symbol] = pos
            self.logger.info(f"New position structure initialized for {symbol} due to fill.")

        current_pos_qty = pos['quantity']
        
        # Determine P&L if fill reduces or closes an existing position
        pnl_from_this_fill = 0.0
        qty_closed = 0.0

        if (current_pos_qty > 0 and not fill_direction_is_buy) or \
           (current_pos_qty < 0 and fill_direction_is_buy): # Closing/reducing existing position
            
            qty_closed = min(abs(current_pos_qty), quantity_filled)
            # Use actual average entry price for P&L calculation, not segment entry price
            actual_entry_price = pos['avg_entry_price']
            segment_entry_price = pos['current_segment_entry_price']  # Keep for logging

            if current_pos_qty > 0: # Closing/reducing long
                pnl_from_this_fill = (fill_price - actual_entry_price) * qty_closed
                self.logger.info(f"Closed LONG segment {qty_closed:.2f} {symbol} at {fill_price:.2f} (entry {actual_entry_price:.2f}). PnL: {pnl_from_this_fill:.2f} in regime '{pos['current_segment_regime']}'.")
            else: # Covering/reducing short
                pnl_from_this_fill = (actual_entry_price - fill_price) * qty_closed
                self.logger.info(f"Covered SHORT segment {qty_closed:.2f} {symbol} at {fill_price:.2f} (entry {actual_entry_price:.2f}). PnL: {pnl_from_this_fill:.2f} in regime '{pos['current_segment_regime']}'.")
            
            self.realized_pnl += pnl_from_this_fill
            # Track both entry and exit regimes for boundary trades
            entry_regime = pos['current_segment_regime']
            exit_regime = self._get_current_regime()
            
            # Create a unique identifier for this trade segment
            segment_id = str(uuid.uuid4())
            
            # Create more detailed trade log entry
            trade_entry = {
                'symbol': symbol,
                'trade_id': pos['trade_id'],
                'segment_id': segment_id,
                'entry_timestamp': pos['entry_timestamp'],
                'exit_timestamp': timestamp,
                'direction': 'LONG' if current_pos_qty > 0 else 'SHORT',
                'entry_price': actual_entry_price,  # Use actual entry price for accurate records
                'exit_price': fill_price,
                'quantity': qty_closed,
                'commission': commission,
                'pnl': pnl_from_this_fill,
                'entry_regime': entry_regime,
                'exit_regime': exit_regime,
                'is_boundary_trade': (entry_regime != exit_regime),
                # For backward compatibility, keep the 'regime' field as the entry regime
                'regime': entry_regime
            }
            
            self._trade_log.append(trade_entry)
            if len(self._trade_log) % 10 == 0:  # Log every 10 trades
                self.logger.info(f"Trade #{len(self._trade_log)} completed")
            
            # Log boundary trades specifically for analysis
            if entry_regime != exit_regime:
                self.logger.info(
                    f"Boundary trade detected: {trade_entry['direction']} {trade_entry['quantity']:.2f} {symbol} " +
                    f"opened in '{entry_regime}' and closed in '{exit_regime}'. " +
                    f"PnL: {pnl_from_this_fill:.2f}, Segment ID: {segment_id}"
                )

        # Update position quantity and cash
        pre_update_cash = self.current_cash
        pre_update_quantity = pos['quantity']
        
        if fill_direction_is_buy:
            pos['quantity'] += quantity_filled
            # Always deduct full cost when buying (whether opening long or closing short)
            self.current_cash -= quantity_filled * fill_price
            self.logger.debug(f"BUY - Cash impact: -{quantity_filled * fill_price:.2f} (on {quantity_filled} shares, {qty_closed} shares closed existing short), Cash before: {pre_update_cash:.2f}, Cash after: {self.current_cash:.2f}")
            self.logger.debug(f"BUY - Quantity impact: +{quantity_filled:.2f}, Quantity before: {pre_update_quantity:.2f}, Quantity after: {pos['quantity']:.2f}")
        else: # SELL
            pos['quantity'] -= quantity_filled
            # Always add full proceeds when selling (whether opening short or closing long)
            self.current_cash += quantity_filled * fill_price
            self.logger.debug(f"SELL - Cash impact: +{quantity_filled * fill_price:.2f} (on {quantity_filled} shares, {qty_closed} shares closed existing long), Cash before: {pre_update_cash:.2f}, Cash after: {self.current_cash:.2f}")
            self.logger.debug(f"SELL - Quantity impact: -{quantity_filled:.2f}, Quantity before: {pre_update_quantity:.2f}, Quantity after: {pos['quantity']:.2f}")

        # Handle position state changes (new, flipped, modified average price)
        if abs(current_pos_qty) < 1e-9 and abs(pos['quantity']) > 1e-9 : # Opened new position (was flat)
            pos['trade_id'] = str(uuid.uuid4()) # New trade ID for this new holding period
            pos['entry_timestamp'] = timestamp
            pos['avg_entry_price'] = fill_price
            pos['cost_basis'] = pos['quantity'] * fill_price # Signed quantity
            pos['current_segment_entry_price'] = fill_price
            pos['current_segment_regime'] = active_regime
            pos['initial_entry_regime_for_trade'] = active_regime
            direction_str = "LONG" if pos['quantity'] > 0 else "SHORT"
            self.logger.info(f"Opened {direction_str} {abs(pos['quantity']):.2f} {symbol} at {fill_price:.2f} in regime '{active_regime}'. Trade ID: {pos['trade_id']}")
        
        elif (current_pos_qty > 0 and pos['quantity'] < -1e-9) or \
             (current_pos_qty < 0 and pos['quantity'] > 1e-9): # Flipped position
            # A position has flipped from long to short or vice versa
            pos['trade_id'] = str(uuid.uuid4()) # New trade ID for the new direction
            pos['entry_timestamp'] = timestamp
            
            # Log the flip details for debugging
            old_direction = "LONG" if current_pos_qty > 0 else "SHORT"
            new_direction = "SHORT" if current_pos_qty > 0 else "LONG"
            self.logger.debug(
                f"Position flip from {old_direction} {abs(current_pos_qty):.2f} to {new_direction} {abs(pos['quantity']):.2f} for {symbol}. "
                f"Fill price: {fill_price:.2f}, Old avg price: {pos['avg_entry_price']:.2f}"
            )
            
            # The flip price is the new basis - this is correct for the new position
            pos['avg_entry_price'] = fill_price 
            
            # When a position flips, the cost basis calculation needs special handling
            # For the new position, cost basis is just the flip price × new quantity
            pos['cost_basis'] = pos['quantity'] * fill_price # Signed quantity
            
            # Reset segment tracking for the new position
            pos['current_segment_entry_price'] = fill_price
            pos['current_segment_regime'] = active_regime
            pos['initial_entry_regime_for_trade'] = active_regime
            
            direction_str = "LONG" if pos['quantity'] > 0 else "SHORT"
            self.logger.info(f"Flipped to {direction_str} {abs(pos['quantity']):.2f} {symbol} at {fill_price:.2f} in regime '{active_regime}'. New Trade ID: {pos['trade_id']}")

        elif abs(pos['quantity']) > 1e-9 : # Added to existing position (or reduced without flipping/closing)
            if qty_closed == 0: # Only adding to position, not closing any part
                if pos['current_segment_regime'] == active_regime:
                    # Different logic for average price calculation depending on position type (long vs short)
                    new_total_abs_qty = abs(current_pos_qty) + quantity_filled
                    
                    if current_pos_qty > 0 and fill_direction_is_buy:  # Adding to long position
                        # Weighted average for longs is straightforward
                        pos['avg_entry_price'] = ((current_pos_qty * pos['avg_entry_price']) + (quantity_filled * fill_price)) / (current_pos_qty + quantity_filled)
                        self.logger.debug(f"Added to LONG {symbol}. Calc: ({current_pos_qty} * {pos['avg_entry_price']}) + ({quantity_filled} * {fill_price}) / {current_pos_qty + quantity_filled}")
                    
                    elif current_pos_qty < 0 and not fill_direction_is_buy:  # Adding to short position
                        # For shorts, we average the prices at which we borrowed/sold shares
                        # The sign is important here - short quantity is negative
                        total_value_borrowed = (abs(current_pos_qty) * pos['avg_entry_price']) + (quantity_filled * fill_price)
                        pos['avg_entry_price'] = total_value_borrowed / (abs(current_pos_qty) + quantity_filled)
                        self.logger.debug(f"Added to SHORT {symbol}. Calc: ({abs(current_pos_qty)} * {pos['avg_entry_price']}) + ({quantity_filled} * {fill_price}) / {abs(current_pos_qty) + quantity_filled}")
                    
                    pos['current_segment_entry_price'] = pos['avg_entry_price'] # Assuming segment avg price tracks overall avg if not flipping regime
                else: # Regime changed while adding to position
                    pos['current_segment_entry_price'] = fill_price
                    pos['current_segment_regime'] = active_regime
                
                pos['cost_basis'] = pos['quantity'] * pos['avg_entry_price'] # Update cost basis based on new average
                direction_str = "LONG" if pos['quantity'] > 0 else "SHORT"
                self.logger.info(f"Added to {direction_str} {symbol}. Old Qty: {current_pos_qty:.2f}, New Qty: {pos['quantity']:.2f}, New Avg Price: {pos['avg_entry_price']:.2f}")
            # If qty_closed > 0, it means part of the position was closed, and if pos['quantity'] is still non-zero,
            # it means the fill was larger than the existing position (a flip), which is handled above.
            # If fill was smaller than existing pos, it's a partial close, P&L logged, quantity reduced.
            # avg_entry_price of remaining position does not change on partial close.
            # current_segment_entry_price also does not change unless regime changes.
            if pos['current_segment_regime'] != active_regime:
                 self.logger.info(f"Regime for open position {symbol} changed from {pos['current_segment_regime']} to {active_regime} during fill.")
                 pos['current_segment_entry_price'] = fill_price # Start new segment at current fill price
                 pos['current_segment_regime'] = active_regime


        if abs(pos['quantity']) < 1e-9: # Position is now flat
            self.logger.info(f"Position in {symbol} fully closed and flattened after fill processing.")
            del self.open_positions[symbol]
        
        self._update_portfolio_value(timestamp)

    def on_bar(self, event: Event):
        bar_data = event.payload
        if not isinstance(bar_data, dict): return
        
        symbol = bar_data.get('symbol')
        close_price = bar_data.get('close')
        timestamp = bar_data.get('timestamp')

        if symbol and isinstance(close_price, (int, float)) and timestamp:
            self._last_bar_prices[symbol] = float(close_price)
            if symbol in self.open_positions:
                pos = self.open_positions[symbol]
                # If market regime changed and position's current segment regime is different
                if pos['current_segment_regime'] != self._get_current_regime():
                    self.logger.info(f"Regime for open position {symbol} tracking update: from '{pos['current_segment_regime']}' to '{self._get_current_regime()}' at {timestamp}.")
                    # Update the regime but DON'T reset the entry price
                    # P&L should be calculated from actual entry price, not segment boundaries
                    # pos['current_segment_entry_price'] = self._last_bar_prices[symbol]  # REMOVED - This was causing P&L calculation errors
                    pos['current_segment_regime'] = self._get_current_regime()
        
        self._update_portfolio_value(timestamp)
    
    def _update_unrealized_pnl(self):
        self.unrealized_pnl = 0.0
        for symbol, position_data in self.open_positions.items():
            if symbol in self._last_bar_prices and abs(position_data['quantity']) > 1e-9:
                current_price = self._last_bar_prices[symbol]
                avg_entry_price = position_data['avg_entry_price'] # Use overall average for total unrealized PnL
                quantity = position_data['quantity'] 
                self.unrealized_pnl += (current_price - avg_entry_price) * quantity
        
    def _update_portfolio_value(self, timestamp: Optional[datetime.datetime]):
        if timestamp is None: 
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            self.logger.debug(f"'{self.instance_name}' _update_portfolio_value called with None timestamp, using current time: {timestamp}")

        # Store previous values for validation
        prev_holdings_value = self.current_holdings_value
        prev_total_value = self.current_total_value
        
        # Calculate holdings value based on current market prices
        self.current_holdings_value = 0.0
        holdings_details = []  # For detailed logging
        
        for symbol, position_data in self.open_positions.items():
            position_qty = position_data['quantity']
            if abs(position_qty) < 1e-9:
                continue  # Skip positions with near-zero quantity
                
            if symbol in self._last_bar_prices:
                # Using current market price
                current_price = self._last_bar_prices[symbol]
                position_value = current_price * position_qty
                holdings_details.append(f"{symbol}: {position_qty:.2f} @ {current_price:.2f} = {position_value:.2f}")
            else:
                # Fallback to entry price if no market price available
                entry_price = position_data['avg_entry_price']
                position_value = entry_price * position_qty
                holdings_details.append(f"{symbol}: {position_qty:.2f} @ {entry_price:.2f} (entry price) = {position_value:.2f}")
                
            self.current_holdings_value += position_value
        
        # Log detailed holdings breakdown for debugging
        if holdings_details:
            self.logger.debug(f"Holdings breakdown: {', '.join(holdings_details)}")
        
        # Update unrealized P&L 
        self._update_unrealized_pnl()
        
        # Calculate total portfolio value
        self.current_total_value = self.current_cash + self.current_holdings_value
        
        # Validate holdings value calculation
        expected_holdings_value = sum([
            self._last_bar_prices.get(symbol, position_data['avg_entry_price']) * position_data['quantity']
            for symbol, position_data in self.open_positions.items() 
            if abs(position_data['quantity']) > 1e-9
        ])
        
        epsilon = 1e-6  # Tolerance for floating point comparison
        if abs(self.current_holdings_value - expected_holdings_value) > epsilon:
            self.logger.warning(
                f"Potential holdings value calculation inconsistency detected: " 
                f"Holdings value {self.current_holdings_value:.2f} doesn't match expected value {expected_holdings_value:.2f} "
                f"(difference: {(self.current_holdings_value - expected_holdings_value):.6f})"
            )
        
        # Validate unrealized P&L calculation
        expected_unrealized_pnl = sum([
            (self._last_bar_prices.get(symbol, position_data['avg_entry_price']) - position_data['avg_entry_price']) * position_data['quantity']
            for symbol, position_data in self.open_positions.items() 
            if abs(position_data['quantity']) > 1e-9 and symbol in self._last_bar_prices
        ])
        
        if abs(self.unrealized_pnl - expected_unrealized_pnl) > epsilon:
            self.logger.warning(
                f"Potential unrealized P&L calculation inconsistency detected: " 
                f"Unrealized P&L {self.unrealized_pnl:.2f} doesn't match expected value {expected_unrealized_pnl:.2f} "
                f"(difference: {(self.unrealized_pnl - expected_unrealized_pnl):.6f})"
            )
        
        # Log significant changes in portfolio value
        if prev_total_value > 0 and abs((self.current_total_value - prev_total_value) / prev_total_value) > 0.01:  # >1% change
            self.logger.debug(
                f"Significant portfolio value change: {prev_total_value:.2f} -> {self.current_total_value:.2f} "
                f"({((self.current_total_value / prev_total_value) - 1) * 100:.2f}%). "
                f"Cash: {self.current_cash:.2f}, Holdings: {prev_holdings_value:.2f} -> {self.current_holdings_value:.2f}"
            )
        
        # Record history
        self._portfolio_value_history.append((timestamp, self.current_total_value))
        
        # Include current regime in the portfolio update
        current_regime = self._get_current_regime()
        self.logger.info(
            f"Portfolio Update at {timestamp} [{current_regime}]: "
            f"Cash={self.current_cash:.2f}, Holdings Value={self.current_holdings_value:.2f}, "
            f"Total Value={self.current_total_value:.2f}, Realized PnL={self.realized_pnl:.2f}, Unrealized PnL={self.unrealized_pnl:.2f}" 
        )
        
    def get_current_position_quantity(self, symbol: str) -> float:
        if symbol in self.open_positions:
            return self.open_positions[symbol].get('quantity', 0.0)
        return 0.0

    def get_last_processed_timestamp(self) -> Optional[datetime.datetime]:
        if self._portfolio_value_history:
            return self._portfolio_value_history[-1][0]
        return None

    def close_all_positions(self, timestamp: datetime.datetime, data_for_closure: Optional[Dict[str, float]] = None) -> None:
        self.logger.info(f"'{self.instance_name}' initiating closing of all open positions at {timestamp}.")
        symbols_to_close = list(self.open_positions.keys()) 

        for symbol in symbols_to_close:
            pos_data = self.open_positions.get(symbol)
            if not pos_data or abs(pos_data['quantity']) < 1e-9 : 
                if symbol in self.open_positions: del self.open_positions[symbol] # Clean up if zero quantity somehow remained
                continue

            # Try to get a close price in this order: 
            # 1. From provided data_for_closure
            # 2. From last_bar_prices
            # 3. If no recent market data is available, use segment entry price as a last resort
            close_price = data_for_closure.get(symbol) if data_for_closure else self._last_bar_prices.get(symbol)
            if close_price is None:
                self.logger.warning(f"No close price available for '{symbol}' to synthetically close. Using its current segment entry price {pos_data['current_segment_entry_price']}.")
                self.logger.warning(f"This may result in zero P&L for this position, which could cause equity miscalculations.")
                close_price = pos_data['current_segment_entry_price'] 

            fill_direction = 'SELL' if pos_data['quantity'] > 0 else 'BUY'
            
            synthetic_fill_data = {
                'symbol': symbol, 'timestamp': timestamp,
                'quantity_filled': abs(pos_data['quantity']), 
                'fill_price': close_price, 'commission': 0.0, 
                'direction': fill_direction,
                'exchange': 'SYNTHETIC_CLOSE', 'fill_id': f"syn_fill_{uuid.uuid4()}", 'order_id': f"syn_ord_{uuid.uuid4()}"
            }
            self.logger.info(f"Synthetically closing position for {symbol}: {fill_direction} {abs(pos_data['quantity']):.2f} at {close_price:.2f}.")
            self.on_fill(Event(EventType.FILL, synthetic_fill_data)) 
        
        self._update_portfolio_value(timestamp) 
        self.logger.info(f"'{self.instance_name}' finished attempting to close all open positions.")
        
    # Alias for method name compatibility with main.py
    def close_all_open_positions(self, timestamp: datetime.datetime, data_for_closure: Optional[Dict[str, float]] = None) -> None:
        """Alias for close_all_positions to maintain compatibility with main.py."""
        self.logger.info(f"close_all_open_positions called (alias for close_all_positions)")
        return self.close_all_positions(timestamp, data_for_closure)

    def get_final_portfolio_value(self) -> float:
        """
        Returns the final portfolio value (current_total_value).
        
        First checks if there is a history entry, otherwise returns current value. 
        Also validates the return value to ensure it is not None or NaN.
        """
        result = None
        # Check history first - most accurate for backtest results
        if self._portfolio_value_history and len(self._portfolio_value_history) > 0:
            result = self._portfolio_value_history[-1][1]
            logging.debug(f"get_final_portfolio_value returning from history: {result}")
        
        # If no history or invalid result, use current value
        if result is None or (isinstance(result, float) and (result != result or result == float('inf') or result == float('-inf'))):
            result = self.current_total_value
            logging.debug(f"get_final_portfolio_value returning current_total_value: {result}")
        
        # As a last resort, return initial cash if everything else fails
        if result is None or (isinstance(result, float) and (result != result or result == float('inf') or result == float('-inf'))):
            result = self.initial_cash
            logging.warning(f"WARNING: Using initial_cash as fallback for get_final_portfolio_value: {result}")
            
        return result

    def _calculate_performance_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics broken down by market regime.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with regime names as keys and performance metrics as values
        """
        # Initialize performance dictionaries
        regime_performance: Dict[str, Dict[str, Any]] = {}
        boundary_trades_performance: Dict[str, Dict[str, Any]] = {}
        
        # Helper function to ensure regime exists in performance dictionary
        def ensure_regime_exists(regime, performance_dict):
            if regime not in performance_dict:
                performance_dict[regime] = {
                    'pnl': 0.0,
                    'commission': 0.0,
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl_values': [],
                    'boundary_trade_count': 0,
                    'boundary_trades_pnl': 0.0,
                    'pure_regime_count': 0,
                    'pure_regime_pnl': 0.0,
                    'sharpe_ratio': None,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'std_dev_pnl': 0.0,
                    'gross_pnl': 0.0,
                    'net_pnl': 0.0
                }
        
        # Process all trades, tracking boundary trades separately
        for trade in self._trade_log:
            # Determine entry and exit regimes
            entry_regime = trade.get('entry_regime', trade.get('regime', 'unknown'))
            exit_regime = trade.get('exit_regime', entry_regime)
            is_boundary_trade = trade.get('is_boundary_trade', (entry_regime != exit_regime))
            
            # Extract trade metrics
            trade_pnl = trade.get('pnl', 0.0)
            trade_commission = trade.get('commission', 0.0)
            
            # Primary attribution is to the entry regime (consistent with previous behavior)
            ensure_regime_exists(entry_regime, regime_performance)
            regime_data = regime_performance[entry_regime]
            
            # Update metrics for the entry regime
            regime_data['pnl'] += trade_pnl
            regime_data['commission'] += trade_commission
            regime_data['count'] += 1
            regime_data['pnl_values'].append(trade_pnl)
            
            # Track if this is a boundary trade
            if is_boundary_trade:
                regime_data['boundary_trade_count'] += 1
                regime_data['boundary_trades_pnl'] += trade_pnl
                
                # Also track this in a special boundary_trades_performance dict for analysis
                ensure_regime_exists(f"{entry_regime}_to_{exit_regime}", boundary_trades_performance)
                boundary_data = boundary_trades_performance[f"{entry_regime}_to_{exit_regime}"]
                boundary_data['pnl'] += trade_pnl
                boundary_data['commission'] += trade_commission
                boundary_data['count'] += 1
                boundary_data['pnl_values'].append(trade_pnl)
                
                if trade_pnl > 0:
                    boundary_data['wins'] += 1
                elif trade_pnl < 0:
                    boundary_data['losses'] += 1
            else:
                # This is a "pure" regime trade (opened and closed in same regime)
                regime_data['pure_regime_count'] += 1
                regime_data['pure_regime_pnl'] += trade_pnl
            
            # Update win/loss counts for the primary regime attribution
            if trade_pnl > 0:
                regime_data['wins'] += 1
            elif trade_pnl < 0:
                regime_data['losses'] += 1
        
        # Calculate additional metrics for each regime
        for regime_dict in [regime_performance, boundary_trades_performance]:
            for regime, metrics in regime_dict.items():
                # Skip if no trades
                if metrics['count'] == 0:
                    continue
                    
                # Calculate win rate
                metrics['win_rate'] = (metrics['wins'] / metrics['count']) if metrics['count'] > 0 else 0.0
                
                # Calculate gross and net PnL
                metrics['gross_pnl'] = metrics['pnl'] - metrics['commission']
                metrics['net_pnl'] = metrics['pnl']
                
                # Calculate average PnL per trade
                metrics['avg_pnl'] = sum(metrics['pnl_values']) / metrics['count']
                
                # Calculate Sharpe ratio if we have enough trades
                if metrics['count'] > 1:
                    pnl_values = metrics['pnl_values']
                    avg_pnl = metrics['avg_pnl']
                    
                    # Calculate standard deviation
                    variance = sum((x - avg_pnl) ** 2 for x in pnl_values) / (metrics['count'] - 1)
                    metrics['std_dev_pnl'] = variance**0.5 if variance > 0 else 0
                    
                    # Calculate annualized Sharpe ratio (assuming 252 trading days per year)
                    if metrics['std_dev_pnl'] > 0:
                        metrics['sharpe_ratio'] = (avg_pnl / metrics['std_dev_pnl']) * (252**0.5)
                    elif avg_pnl > 0:
                        metrics['sharpe_ratio'] = float('inf')  # Infinite Sharpe for positive return with no volatility
                    else:
                        metrics['sharpe_ratio'] = 0.0  # Zero Sharpe for zero or negative return with no volatility
        
        # Add boundary trades summary to the main performance dict
        regime_performance['_boundary_trades_summary'] = boundary_trades_performance
        
        return regime_performance
    
    def get_performance_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """
        Public method to access performance metrics broken down by market regime.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with regime names as keys and performance metrics as values
        """
        return self._calculate_performance_by_regime()
        
    def get_performance(self) -> Dict[str, Any]:
        """
        Returns a dictionary of overall portfolio performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of performance metrics
        """
        # Calculate overall performance metrics
        metrics = {
            "final_portfolio_value": self.current_total_value,
            "initial_portfolio_value": self.initial_cash,
            "total_return": (self.current_total_value / self.initial_cash) - 1.0 if self.initial_cash > 0 else 0.0,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "current_cash": self.current_cash,
            "current_holdings_value": self.current_holdings_value
        }
        
        # Add additional metrics from any regime calculations if they exist
        return metrics
        
    
    def _log_final_performance_summary(self):
        # Skip detailed summary if no trades were executed (e.g., in main context during optimization)
        if len(self._trade_log) == 0 and self.current_cash == self.initial_cash:
            self.logger.debug(f"{self.instance_name}: No trades executed, skipping detailed summary")
            return
            
        self.logger.info(f"--- {self.instance_name} Final Summary ---")
        self.logger.info(f"Initial Cash: {self.initial_cash:.2f}")
        self.logger.info(f"Final Cash: {self.current_cash:.2f}")
        
        open_pos_display = {sym: data['quantity'] for sym, data in self.open_positions.items() if data.get('quantity', 0) != 0}
        if open_pos_display: self.logger.info(f"Final Open Holdings: {open_pos_display}")
        else: self.logger.info("Final Open Holdings: None (all positions were closed).")

        self.logger.info(f"Final Portfolio Value: {self.current_total_value:.2f}")
        self.logger.info(f"Total Realized P&L: {self.realized_pnl:.2f}")
        self.logger.info(f"Number of Trade Segments Logged: {len(self._trade_log)}")

        self.logger.info("--- Performance by Regime ---")
        regime_performance = self._calculate_performance_by_regime()
        
        # Log performance for each regime (excluding boundary trades summary)
        for regime, metrics in regime_performance.items():
            # Skip the boundary trades summary section, we'll handle it separately
            if regime == '_boundary_trades_summary':
                continue
                
            self.logger.info(f"  Regime: {regime}")
            self.logger.info(f"    Total Gross Pnl: {metrics['gross_pnl']:.2f}") 
            self.logger.info(f"    Trade Segments: {metrics['count']}")
            
            # Add boundary trade information if applicable
            if 'boundary_trade_count' in metrics and metrics['boundary_trade_count'] > 0:
                pure_count = metrics.get('pure_regime_count', 0)
                boundary_count = metrics.get('boundary_trade_count', 0)
                boundary_pnl = metrics.get('boundary_trades_pnl', 0.0)
                pure_pnl = metrics.get('pure_regime_pnl', 0.0)
                
                self.logger.info(f"    - Pure Regime Trades: {pure_count} (PnL: {pure_pnl:.2f})")
                self.logger.info(f"    - Boundary Trades: {boundary_count} (PnL: {boundary_pnl:.2f})")
            
            self.logger.info(f"    Winning Segments: {metrics['wins']}")
            self.logger.info(f"    Losing Segments: {metrics['losses']}")
            self.logger.info(f"    Total Commission: {metrics['commission']:.2f}")
            self.logger.info(f"    Net Pnl Sum: {metrics['net_pnl']:.2f}") 
            self.logger.info(f"    Win Rate: {metrics['win_rate']:.2f}")
            
            if metrics.get('sharpe_ratio') is not None:
                self.logger.info(f"    Sharpe Ratio (annualized from segment PnLs): {metrics['sharpe_ratio']:.2f}")
            else:
                self.logger.info("    Sharpe Ratio (from segment PnLs): N/A")
        
        # Log boundary trades details if any exist
        boundary_trades = regime_performance.get('_boundary_trades_summary', {})
        if boundary_trades:
            self.logger.info("--- Boundary Trades Details ---")
            total_boundary_trades = sum(metrics['count'] for metrics in boundary_trades.values())
            total_boundary_pnl = sum(metrics['pnl'] for metrics in boundary_trades.values())
            
            self.logger.info(f"  Total Boundary Trades: {total_boundary_trades}")
            self.logger.info(f"  Total Boundary Trades PnL: {total_boundary_pnl:.2f}")
            
            for transition, metrics in boundary_trades.items():
                self.logger.info(f"  Transition: {transition}")
                self.logger.info(f"    Count: {metrics['count']}")
                self.logger.info(f"    PnL: {metrics['pnl']:.2f}")
                self.logger.info(f"    Win Rate: {metrics['win_rate']:.2f}")
                
                if metrics.get('sharpe_ratio') is not None and metrics['count'] > 1:
                    self.logger.info(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                
                # Add separation line between transitions
                self.logger.info("    " + "-" * 40)
    
    def calculate_portfolio_sharpe_ratio(self, risk_free_rate: float = 0.0) -> Optional[float]:
        """
        Calculate Sharpe ratio based on portfolio value changes (returns).
        
        This is the correct way to calculate Sharpe ratio - from actual portfolio
        returns, not from individual trade P&Ls.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0.0)
            
        Returns:
            Annualized Sharpe ratio or None if insufficient data
        """
        if len(self._portfolio_value_history) < 2:
            self.logger.warning("Insufficient portfolio history for Sharpe ratio calculation")
            return None
            
        # Calculate returns between each portfolio value update
        returns = []
        for i in range(1, len(self._portfolio_value_history)):
            prev_time, prev_value = self._portfolio_value_history[i-1]
            curr_time, curr_value = self._portfolio_value_history[i]
            
            if prev_value > 0:  # Avoid division by zero
                period_return = (curr_value - prev_value) / prev_value
                returns.append(period_return)
                
        if not returns:
            return None
            
        # Calculate statistics
        import numpy as np
        returns_array = np.array(returns)
        
        # Average return
        avg_return = np.mean(returns_array)
        
        # Standard deviation of returns
        std_return = np.std(returns_array, ddof=1) if len(returns) > 1 else 0
        
        if std_return == 0:
            self.logger.warning("Zero standard deviation in returns")
            return None
            
        # Calculate time scaling factor
        # Assume each bar is 1 minute for intraday data
        # 252 trading days * 6.5 hours * 60 minutes = 98,280 bars per year
        # But we'll use 252 * 390 minutes = 98,280 for consistency
        bars_per_year = 252 * 390  # Standard trading minutes per year
        
        # Estimate bars per period from timestamps if available
        if len(self._portfolio_value_history) > 1:
            # Get average time between updates
            time_diffs = []
            for i in range(1, min(10, len(self._portfolio_value_history))):  # Sample first 10
                if self._portfolio_value_history[i][0] and self._portfolio_value_history[i-1][0]:
                    diff = (self._portfolio_value_history[i][0] - self._portfolio_value_history[i-1][0]).total_seconds()
                    time_diffs.append(diff)
                    
            if time_diffs:
                avg_seconds = sum(time_diffs) / len(time_diffs)
                if avg_seconds > 0:
                    # Assume 1-minute bars if average is around 60 seconds
                    if 30 <= avg_seconds <= 90:
                        annualization_factor = np.sqrt(bars_per_year)
                    else:
                        # For other timeframes, scale appropriately
                        minutes_per_bar = avg_seconds / 60
                        bars_per_year_adjusted = bars_per_year / minutes_per_bar
                        annualization_factor = np.sqrt(bars_per_year_adjusted)
                else:
                    annualization_factor = np.sqrt(252)  # Default to daily
            else:
                annualization_factor = np.sqrt(252)  # Default to daily
        else:
            annualization_factor = np.sqrt(252)  # Default to daily
            
        # Calculate Sharpe ratio
        # Annualized return
        annualized_return = avg_return * bars_per_year
        
        # Annualized volatility
        annualized_vol = std_return * annualization_factor
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
        
        # Log calculation details
        self.logger.info(
            f"Portfolio Sharpe Ratio Calculation: "
            f"Avg Return={avg_return:.6f}, Std Dev={std_return:.6f}, "
            f"Annualization Factor={annualization_factor:.1f}, "
            f"Sharpe={sharpe_ratio:.4f}"
        )
        
        return sharpe_ratio
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics including both trade-based
        and portfolio-based calculations.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'initial_value': self.initial_cash,
            'final_value': self.current_total_value,
            'total_return': (self.current_total_value - self.initial_cash) / self.initial_cash,
            'total_return_pct': ((self.current_total_value - self.initial_cash) / self.initial_cash) * 100,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'num_trades': len(self._trade_log),
            'portfolio_sharpe_ratio': self.calculate_portfolio_sharpe_ratio()
        }
        
        # Add regime-based metrics
        regime_performance = self._calculate_performance_by_regime()
        metrics['regime_performance'] = regime_performance
        
        return metrics
        
    def get_performance_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """
        Get regime-specific performance metrics.
        
        Note: This method returns trade-based Sharpe ratios which may be incorrect
        due to the segment P&L calculation bug. Use get_performance_by_regime_with_portfolio_sharpe()
        for corrected Sharpe ratios.
        
        Returns:
            Dictionary mapping regime names to performance metrics
        """
        return self._calculate_performance_by_regime()
        
    def get_performance_by_regime_with_portfolio_sharpe(self) -> Dict[str, Dict[str, Any]]:
        """
        Get regime-specific performance metrics with corrected Sharpe ratios.
        
        This method provides regime performance but replaces the trade-based
        Sharpe ratios with the correct portfolio-based Sharpe ratio.
        
        Returns:
            Dictionary mapping regime names to performance metrics
        """
        # Get the standard regime performance (with incorrect Sharpe ratios)
        regime_performance = self._calculate_performance_by_regime()
        
        # Calculate the correct portfolio-based Sharpe ratio
        portfolio_sharpe = self.calculate_portfolio_sharpe_ratio()
        
        # For optimization purposes, we'll use the portfolio Sharpe for all regimes
        # since the trade-based Sharpe ratios are incorrect due to the P&L bug
        if portfolio_sharpe is not None:
            for regime, metrics in regime_performance.items():
                if regime != '_boundary_trades_summary' and isinstance(metrics, dict):
                    # Replace the incorrect trade-based Sharpe with portfolio-based
                    metrics['sharpe_ratio'] = portfolio_sharpe
                    metrics['sharpe_ratio_source'] = 'portfolio_returns'
                    
        return regime_performance
    
    def teardown(self):
        """Clean up resources."""
        super().teardown()
        self.open_positions.clear()
        self._trade_log.clear()
        self._portfolio_value_history.clear()
        self._last_bar_prices.clear()
