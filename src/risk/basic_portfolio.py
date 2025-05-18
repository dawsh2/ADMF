# src/risk/basic_portfolio.py
import logging
import datetime 
from typing import Dict, Any, Optional, List, Tuple 
import uuid

from ..core.component import BaseComponent
from ..core.event import Event, EventType
from ..core.exceptions import ComponentError, DependencyNotFoundError 

class BasicPortfolio(BaseComponent):
    def __init__(self, 
                 instance_name: str, 
                 config_loader, 
                 event_bus, 
                 container, 
                 component_config_key: Optional[str] = None):
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        self._container = container 
        self.initial_cash: float = self.get_specific_config('initial_cash', 100000.0)
        self.current_cash: float = self.initial_cash
        
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
        self.regime_detector_key: str = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        self._current_market_regime: Optional[str] = "default" 

        self._portfolio_value_history: List[Tuple[Optional[datetime.datetime], float]] = []
        
        self.logger.info(f"BasicPortfolio '{self.name}' initialized with initial cash: {self.initial_cash:.2f}")

    def setup(self):
        super().setup() 
        self.logger.info(f"Setting up BasicPortfolio '{self.name}'...")
        if self._event_bus:
            self._event_bus.subscribe(EventType.FILL, self.on_fill)
            self._event_bus.subscribe(EventType.BAR, self.on_bar)
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.name}' subscribed to FILL, BAR, and CLASSIFICATION events.")
        else:
            self.logger.error(f"Event bus not available for '{self.name}'. Cannot subscribe to events.")
            self.state = BaseComponent.STATE_FAILED
            return

        try:
            self._regime_detector = self._container.resolve(self.regime_detector_key)
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
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"BasicPortfolio '{self.name}' setup complete. State: {self.state}")

    def start(self):
        super().start()
        self.logger.info(f"BasicPortfolio '{self.name}' started.")
        if not self._portfolio_value_history:
             self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc))

    def stop(self):
        self.logger.info(f"Stopping {self.name}...")
        self._log_final_performance_summary()
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.FILL, self.on_fill)
                self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.info(f"'{self.name}' unsubscribed from events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.name}': {e}", exc_info=True)
        super().stop()
        self.logger.info(f"{self.name} stopped. State: {self.state}")

    def on_classification_change(self, event: Event):
        payload = event.payload
        if not isinstance(payload, dict): return
        new_regime = payload.get('classification')
        timestamp = payload.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
        if new_regime and new_regime != self._current_market_regime:
            self.logger.info(f"Market regime changed from '{self._current_market_regime}' to '{new_regime}' at {timestamp} for '{self.name}'.")
            self._current_market_regime = new_regime

    def _get_current_regime(self) -> str:
        return self._current_market_regime or "default"

    def on_fill(self, event: Event):
        fill_data = event.payload
        if not (isinstance(fill_data, dict) and all(k in fill_data for k in ['symbol', 'timestamp', 'quantity_filled', 'fill_price', 'direction'])):
            self.logger.error(f"'{self.name}' received incomplete or invalid FILL data: {fill_data}"); return

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
            segment_entry_price = pos['current_segment_entry_price']

            if current_pos_qty > 0: # Closing/reducing long
                pnl_from_this_fill = (fill_price - segment_entry_price) * qty_closed
                self.logger.info(f"Closed LONG segment {qty_closed:.2f} {symbol} at {fill_price:.2f} (entry {segment_entry_price:.2f}). PnL: {pnl_from_this_fill:.2f} in regime '{pos['current_segment_regime']}'.")
            else: # Covering/reducing short
                pnl_from_this_fill = (segment_entry_price - fill_price) * qty_closed
                self.logger.info(f"Covered SHORT segment {qty_closed:.2f} {symbol} at {fill_price:.2f} (entry {segment_entry_price:.2f}). PnL: {pnl_from_this_fill:.2f} in regime '{pos['current_segment_regime']}'.")
            
            self.realized_pnl += pnl_from_this_fill
            self._trade_log.append({
                'symbol': symbol, 'trade_id': pos['trade_id'], 'segment_id': str(uuid.uuid4()),
                'entry_timestamp': pos['entry_timestamp'], 'exit_timestamp': timestamp, 
                'direction': 'LONG' if current_pos_qty > 0 else 'SHORT', 
                'entry_price': segment_entry_price, 'exit_price': fill_price,
                'quantity': qty_closed, 'commission': commission, 
                'pnl': pnl_from_this_fill, 'regime': pos['current_segment_regime']
            })

        # Update position quantity and cash
        pre_update_cash = self.current_cash
        pre_update_quantity = pos['quantity']
        
        if fill_direction_is_buy:
            pos['quantity'] += quantity_filled
            self.current_cash -= quantity_filled * fill_price
            self.logger.debug(f"BUY - Cash impact: -{quantity_filled * fill_price:.2f}, Cash before: {pre_update_cash:.2f}, Cash after: {self.current_cash:.2f}")
            self.logger.debug(f"BUY - Quantity impact: +{quantity_filled:.2f}, Quantity before: {pre_update_quantity:.2f}, Quantity after: {pos['quantity']:.2f}")
        else: # SELL
            pos['quantity'] -= quantity_filled
            self.current_cash += quantity_filled * fill_price
            self.logger.debug(f"SELL - Cash impact: +{quantity_filled * fill_price:.2f}, Cash before: {pre_update_cash:.2f}, Cash after: {self.current_cash:.2f}")
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
                    # This indicates a regime change happened between fills. 
                    # For accurate segment P&L, one might close the old segment and start a new one.
                    # For simplicity here, we update the segment's regime.
                    # The entry price for this "new segment" would ideally be the current market price.
                    pos['current_segment_entry_price'] = self._last_bar_prices[symbol] # Mark-to-market for new segment
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
            self.logger.debug(f"'{self.name}' _update_portfolio_value called with None timestamp, using current time: {timestamp}")

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
        
        self.logger.info(
            f"Portfolio Update at {timestamp}: "
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
        self.logger.info(f"'{self.name}' initiating closing of all open positions at {timestamp}.")
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
        self.logger.info(f"'{self.name}' finished attempting to close all open positions.")
        
    # Alias for method name compatibility with main.py
    def close_all_open_positions(self, timestamp: datetime.datetime, data_for_closure: Optional[Dict[str, float]] = None) -> None:
        """Alias for close_all_positions to maintain compatibility with main.py."""
        self.logger.info(f"close_all_open_positions called (alias for close_all_positions)")
        return self.close_all_positions(timestamp, data_for_closure)

    def get_final_portfolio_value(self) -> float:
        if self._portfolio_value_history:
            return self._portfolio_value_history[-1][1] 
        return self.current_total_value 

    def _log_final_performance_summary(self):
        self.logger.info(f"--- {self.name} Final Summary ---")
        self.logger.info(f"Initial Cash: {self.initial_cash:.2f}")
        self.logger.info(f"Final Cash: {self.current_cash:.2f}")
        
        open_pos_display = {sym: data['quantity'] for sym, data in self.open_positions.items() if data.get('quantity', 0) != 0}
        if open_pos_display: self.logger.info(f"Final Open Holdings: {open_pos_display}")
        else: self.logger.info("Final Open Holdings: None (all positions were closed).")

        self.logger.info(f"Final Portfolio Value: {self.current_total_value:.2f}")
        self.logger.info(f"Total Realized P&L: {self.realized_pnl:.2f}")
        self.logger.info(f"Number of Trade Segments Logged: {len(self._trade_log)}")

        self.logger.info("--- Performance by Regime ---")
        regime_performance: Dict[str, Dict[str, Any]] = {}
        for trade in self._trade_log:
            regime = trade.get('regime', 'unknown') 
            if regime not in regime_performance:
                regime_performance[regime] = {'pnl': 0.0, 'commission': 0.0, 'count': 0, 'wins': 0, 'losses': 0, 'pnl_values': []}
            
            trade_pnl = trade.get('pnl', 0.0); trade_commission = trade.get('commission', 0.0)
            regime_performance[regime]['pnl'] += trade_pnl
            regime_performance[regime]['commission'] += trade_commission
            regime_performance[regime]['count'] += 1
            regime_performance[regime]['pnl_values'].append(trade_pnl) 
            if trade_pnl > 0: regime_performance[regime]['wins'] += 1
            elif trade_pnl < 0: regime_performance[regime]['losses'] += 1

        for regime, metrics in regime_performance.items():
            self.logger.info(f"  Regime: {regime}")
            gross_pnl = metrics['pnl'] + metrics['commission']; net_pnl_sum = metrics['pnl'] 
            self.logger.info(f"    Total Gross Pnl: {gross_pnl:.2f}") 
            self.logger.info(f"    Trade Segments: {metrics['count']}")
            self.logger.info(f"    Winning Segments: {metrics['wins']}")
            self.logger.info(f"    Losing Segments: {metrics['losses']}")
            self.logger.info(f"    Total Commission: {metrics['commission']:.2f}")
            self.logger.info(f"    Net Pnl Sum: {net_pnl_sum:.2f}") 
            win_rate = (metrics['wins'] / metrics['count']) if metrics['count'] > 0 else 0
            self.logger.info(f"    Win Rate: {win_rate:.2f}")
            
            if metrics['count'] > 1:
                pnl_values = metrics['pnl_values']; avg_pnl = sum(pnl_values) / metrics['count']
                variance = sum((x - avg_pnl) ** 2 for x in pnl_values) / (metrics['count'] -1) 
                std_dev_pnl = variance**0.5 if variance > 0 else 0
                sharpe = (avg_pnl / std_dev_pnl) * (252**0.5) if std_dev_pnl > 0 else float('inf') if avg_pnl > 0 else 0.0 
                self.logger.info(f"    Sharpe Ratio (annualized from segment PnLs): {sharpe:.2f}")
            else: self.logger.info("    Sharpe Ratio (from segment PnLs): N/A")
