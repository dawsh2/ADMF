# src/risk/basic_portfolio.py
import logging
import datetime # For type hinting and operations
from typing import Dict, Any, Optional, List, Tuple 
import uuid

from ..core.component import BaseComponent
from ..core.event import Event, EventType
from ..core.exceptions import ComponentError, DependencyNotFoundError 

class BasicPortfolio(BaseComponent):
    """
    Manages the portfolio's positions, cash, and tracks performance.
    This version will be enhanced to be regime-aware.
    """
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
        
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self._trade_log: List[Dict[str, Any]] = []
        
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.current_holdings_value: float = 0.0
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
            if self._regime_detector:
                self.logger.info(f"Successfully resolved and linked RegimeDetector: {self._regime_detector.name}")
                if hasattr(self._regime_detector, 'get_current_classification') and callable(getattr(self._regime_detector, 'get_current_classification')):
                    initial_regime = self._regime_detector.get_current_classification()
                    self._current_market_regime = initial_regime if initial_regime else "default"
                else:
                    self._current_market_regime = "default" # Fallback
                self.logger.info(f"Initial market regime set to: {self._current_market_regime}")
            else: # Should ideally not happen if resolve doesn't raise error but returns None
                self.logger.warning(f"Failed to resolve RegimeDetector with key '{self.regime_detector_key}'. Regime-aware tracking will use 'default' regime.")
                self._current_market_regime = "default"
        except DependencyNotFoundError:
            self.logger.warning(f"Dependency '{self.regime_detector_key}' (RegimeDetector) not found. Regime-aware tracking will use 'default' regime.")
            self._current_market_regime = "default"
        except Exception as e:
            self.logger.error(f"Error resolving RegimeDetector '{self.regime_detector_key}': {e}", exc_info=True)
            self._current_market_regime = "default"

        self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc)) 
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"BasicPortfolio '{self.name}' setup complete. State: {self.state}")

    def start(self):
        super().start()
        self.logger.info(f"BasicPortfolio '{self.name}' started. Monitoring FILL, BAR and CLASSIFICATION events...")
        if not self._portfolio_value_history: # Should have been populated by setup
             self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc))


    def stop(self):
        self.logger.info(f"Stopping {self.name}...")
        self._log_final_performance_summary()
        
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.FILL, self.on_fill)
                self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.info(f"'{self.name}' unsubscribed from FILL, BAR, and CLASSIFICATION events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.name}': {e}", exc_info=True)
        
        super().stop()
        self.logger.info(f"{self.name} stopped. State: {self.state}")

    def on_classification_change(self, event: Event):
        if not event.payload or not isinstance(event.payload, dict):
            self.logger.warning(f"'{self.name}' received CLASSIFICATION event with invalid payload.")
            return
            
        new_regime = event.payload.get('classification')
        timestamp = event.payload.get('timestamp', datetime.datetime.now(datetime.timezone.utc))

        if new_regime and new_regime != self._current_market_regime:
            self.logger.info(f"Market regime changed from '{self._current_market_regime}' to '{new_regime}' at {timestamp} for '{self.name}'.")
            self._current_market_regime = new_regime
        elif not new_regime:
            self.logger.warning(f"Received CLASSIFICATION event with no 'classification' in payload for '{self.name}'.")

    def _get_current_regime(self) -> str:
        return self._current_market_regime or "default"

    def on_fill(self, event: Event):
        fill_data = event.payload
        if not isinstance(fill_data, dict):
            self.logger.error(f"'{self.name}' received FILL event with non-dict payload: {fill_data}")
            return

        symbol = fill_data.get('symbol')
        timestamp = fill_data.get('timestamp') 
        quantity_filled = float(fill_data.get('quantity_filled', 0.0))
        fill_price = float(fill_data.get('fill_price', 0.0))
        commission = float(fill_data.get('commission', 0.0))
        direction = fill_data.get('direction', '').upper()  # 'BUY' or 'SELL'
        
        if not all([symbol, quantity_filled > 0, fill_price > 0, direction in ['BUY', 'SELL'], timestamp is not None]):
            self.logger.error(f"'{self.name}' received incomplete or invalid FILL data: {fill_data}")
            return

        self.current_cash -= commission
        current_regime = self._get_current_regime()

        if symbol not in self.open_positions:
            self.open_positions[symbol] = {
                'quantity': 0.0,  # Signed: positive for long, negative for short
                'avg_entry_price': 0.0, # Weighted average entry price for the current overall holding
                'cost_basis': 0.0, # Net cost of establishing the current quantity (negative for shorts if proceeds held)
                'entry_timestamp': None,
                'trade_id': str(uuid.uuid4()), # ID for the current continuous holding (flips get new ID)
                'current_segment_entry_price': 0.0,
                'current_segment_regime': current_regime,
                'initial_entry_regime_for_trade': current_regime # Regime when this trade_id started
            }
        
        pos = self.open_positions[symbol]
        current_pos_quantity = pos['quantity']
        current_pos_direction_is_long = current_pos_quantity > 0
        current_pos_direction_is_short = current_pos_quantity < 0

        pnl_from_fill = 0.0
        
        if direction == 'BUY':
            if current_pos_direction_is_short: # Buying to cover/reduce a short position
                qty_to_cover = min(quantity_filled, abs(current_pos_quantity))
                pnl_from_fill = (pos['current_segment_entry_price'] - fill_price) * qty_to_cover
                self.realized_pnl += pnl_from_fill
                self.current_cash -= (qty_to_cover * fill_price) # Cash out to buy back shares

                self.logger.info(f"Covered SHORT {qty_to_cover:.2f} {symbol} at {fill_price:.2f}. PnL: {pnl_from_fill:.2f} in regime '{pos['current_segment_regime']}'.")
                self._trade_log.append({
                    'symbol': symbol, 'trade_id': pos['trade_id'], 'segment_id': str(uuid.uuid4()),
                    'entry_timestamp': pos['entry_timestamp'], 'exit_timestamp': timestamp, 
                    'direction': 'SHORT', 'entry_price': pos['current_segment_entry_price'], 'exit_price': fill_price,
                    'quantity': qty_to_cover, 'commission': commission, 
                    'pnl': pnl_from_fill, 'regime': pos['current_segment_regime']
                })
                pos['quantity'] += qty_to_cover # Becomes less negative or zero
                pos['cost_basis'] += qty_to_cover * pos['current_segment_entry_price'] # Reducing liability from short sale proceeds

                if pos['quantity'] == 0: # Short position fully closed
                    if quantity_filled > qty_to_cover: # Fill results in flipping to long
                        remaining_qty = quantity_filled - qty_to_cover
                        pos['trade_id'] = str(uuid.uuid4()) # New trade for the new long position
                        pos['initial_entry_regime_for_trade'] = current_regime
                        pos['entry_timestamp'] = timestamp
                        pos['quantity'] = remaining_qty
                        pos['cost_basis'] = remaining_qty * fill_price
                        pos['current_segment_entry_price'] = fill_price
                        pos['current_segment_regime'] = current_regime
                        self.current_cash -= remaining_qty * fill_price
                        self.logger.info(f"Flipped to LONG {remaining_qty:.2f} {symbol} at {fill_price:.2f} in regime '{current_regime}'. New Trade ID: {pos['trade_id']}")
                else: # Still short, just reduced
                    # Avg entry price for shorts might need re-evaluation if partial cover. For now, segment price is key.
                    self.logger.info(f"Reduced SHORT {symbol} by {qty_to_cover:.2f}. Remaining: {pos['quantity']:.2f}")

            else: # Opening new long or adding to existing long
                if current_pos_quantity == 0: # Opening new long
                    pos['entry_timestamp'] = timestamp
                    pos['trade_id'] = str(uuid.uuid4()) # Ensure new trade_id
                    pos['initial_entry_regime_for_trade'] = current_regime
                    pos['current_segment_entry_price'] = fill_price
                    pos['current_segment_regime'] = current_regime
                    self.logger.info(f"Opened LONG {quantity_filled:.2f} {symbol} at {fill_price:.2f} in regime '{current_regime}'. Trade ID: {pos['trade_id']}")
                else: # Adding to existing long
                    self.logger.info(f"Adding to LONG {symbol}. Old Qty: {pos['quantity']:.2f}, New Qty: {pos['quantity'] + quantity_filled:.2f}")
                    # Update average entry price for the segment if regime hasn't changed
                    if pos['current_segment_regime'] == current_regime:
                         new_total_cost = (pos['current_segment_entry_price'] * pos['quantity']) + (fill_price * quantity_filled)
                         pos['current_segment_entry_price'] = new_total_cost / (pos['quantity'] + quantity_filled)
                    else: # Regime changed, start new segment accounting
                        pos['current_segment_entry_price'] = fill_price
                        pos['current_segment_regime'] = current_regime

                pos['quantity'] += quantity_filled
                pos['cost_basis'] += quantity_filled * fill_price
                self.current_cash -= quantity_filled * fill_price
        
        elif direction == 'SELL':
            if current_pos_direction_is_long: # Selling to close/reduce a long position
                qty_to_sell = min(quantity_filled, current_pos_quantity)
                pnl_from_fill = (fill_price - pos['current_segment_entry_price']) * qty_to_sell
                self.realized_pnl += pnl_from_fill
                self.current_cash += qty_to_sell * fill_price

                self.logger.info(f"Closed LONG {qty_to_sell:.2f} {symbol} at {fill_price:.2f}. PnL: {pnl_from_fill:.2f} in regime '{pos['current_segment_regime']}'.")
                self._trade_log.append({
                    'symbol': symbol, 'trade_id': pos['trade_id'], 'segment_id': str(uuid.uuid4()),
                    'entry_timestamp': pos['entry_timestamp'], 'exit_timestamp': timestamp, 
                    'direction': 'LONG', 'entry_price': pos['current_segment_entry_price'], 'exit_price': fill_price,
                    'quantity': qty_to_sell, 'commission': commission, 
                    'pnl': pnl_from_fill, 'regime': pos['current_segment_regime']
                })
                pos['quantity'] -= qty_to_sell
                pos['cost_basis'] -= pos['current_segment_entry_price'] * qty_to_sell 

                if pos['quantity'] == 0: # Long position fully closed
                    if quantity_filled > qty_to_sell: # Fill results in flipping to short
                        remaining_qty = quantity_filled - qty_to_sell
                        pos['trade_id'] = str(uuid.uuid4()) # New trade for the new short position
                        pos['initial_entry_regime_for_trade'] = current_regime
                        pos['entry_timestamp'] = timestamp
                        pos['quantity'] = -remaining_qty # Negative for short
                        pos['cost_basis'] = -(remaining_qty * fill_price) # Representing value of borrowed shares
                        pos['current_segment_entry_price'] = fill_price
                        pos['current_segment_regime'] = current_regime
                        self.current_cash += remaining_qty * fill_price # Proceeds from short sale
                        self.logger.info(f"Flipped to SHORT {remaining_qty:.2f} {symbol} at {fill_price:.2f} in regime '{current_regime}'. New Trade ID: {pos['trade_id']}")
                else: # Still long, just reduced
                     self.logger.info(f"Reduced LONG {symbol} by {qty_to_sell:.2f}. Remaining: {pos['quantity']:.2f}")


            else: # Opening new short or adding to existing short
                if current_pos_quantity == 0: # Opening new short
                    pos['entry_timestamp'] = timestamp
                    pos['trade_id'] = str(uuid.uuid4())
                    pos['initial_entry_regime_for_trade'] = current_regime
                    pos['current_segment_entry_price'] = fill_price
                    pos['current_segment_regime'] = current_regime
                    self.logger.info(f"Opened SHORT {quantity_filled:.2f} {symbol} at {fill_price:.2f} in regime '{current_regime}'. Trade ID: {pos['trade_id']}")
                else: # Adding to existing short
                    self.logger.info(f"Adding to SHORT {symbol}. Old Qty: {pos['quantity']:.2f}, New Qty: {pos['quantity'] - quantity_filled:.2f}")
                    if pos['current_segment_regime'] == current_regime:
                        new_total_proceeds = (abs(pos['quantity']) * pos['current_segment_entry_price']) + (quantity_filled * fill_price)
                        pos['current_segment_entry_price'] = new_total_proceeds / (abs(pos['quantity']) + quantity_filled)
                    else: # Regime changed
                        pos['current_segment_entry_price'] = fill_price
                        pos['current_segment_regime'] = current_regime
                
                pos['quantity'] -= quantity_filled # More negative
                pos['cost_basis'] -= quantity_filled * fill_price # Accumulating proceeds (negative cost)
                self.current_cash += quantity_filled * fill_price
        
        # Clean up if position truly flat
        if pos['quantity'] == 0:
            self.logger.info(f"Position in {symbol} fully closed and flattened.")
            del self.open_positions[symbol]
        else:
            pos['avg_entry_price'] = abs(pos['cost_basis'] / pos['quantity']) if pos['quantity'] != 0 else 0
            pos['direction'] = 'LONG' if pos['quantity'] > 0 else 'SHORT'

        self._update_portfolio_value(timestamp)

    def on_bar(self, event: Event):
        bar_data = event.payload
        if not isinstance(bar_data, dict):
            self.logger.warning(f"'{self.name}' received BAR event with non-dict payload: {bar_data}")
            return
        
        symbol = bar_data.get('symbol')
        close_price = bar_data.get('close')
        timestamp = bar_data.get('timestamp', datetime.datetime.now(datetime.timezone.utc))

        if symbol and isinstance(close_price, (int, float)):
            self._last_bar_prices[symbol] = float(close_price)
            if symbol in self.open_positions:
                # If current market regime changed (detected by on_classification_change)
                # and it's different from the segment's current regime, update the segment.
                # This is a simplified way to handle intra-trade regime changes for P&L segmenting.
                # More advanced: close old segment, open new one.
                if self.open_positions[symbol]['current_segment_regime'] != self._get_current_regime():
                    self.logger.info(f"Regime for open position {symbol} updated from '{self.open_positions[symbol]['current_segment_regime']}' to '{self._get_current_regime()}' based on latest classification.")
                    # For P&L: The segment would effectively end here, and a new one starts.
                    # A simpler approach for now: update the regime for subsequent PnL calculations on close.
                    # The current_segment_entry_price would ideally reset to current market price here
                    # if we were creating a new distinct segment for accounting.
                    self.open_positions[symbol]['current_segment_regime'] = self._get_current_regime()
                    # self.open_positions[symbol]['current_segment_entry_price'] = close_price # If starting new segment accounting
        
        self._update_portfolio_value(timestamp)
    
    def _update_unrealized_pnl(self):
        self.unrealized_pnl = 0.0
        for symbol, position_data in self.open_positions.items():
            if symbol in self._last_bar_prices and position_data['quantity'] != 0:
                current_price = self._last_bar_prices[symbol]
                segment_entry_price = position_data['current_segment_entry_price']
                quantity = position_data['quantity'] # Signed quantity
                
                # For long (quantity > 0): (current_price - entry_price) * quantity
                # For short (quantity < 0): (entry_price - current_price) * abs(quantity)
                # which is also (current_price - entry_price) * quantity (since quantity is negative)
                self.unrealized_pnl += (current_price - segment_entry_price) * quantity
        
    def _update_portfolio_value(self, timestamp: Optional[datetime.datetime]):
        if timestamp is None: 
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            self.logger.debug(f"'{self.name}' _update_portfolio_value called with None timestamp, using current time: {timestamp}")

        self.current_holdings_value = 0.0
        for symbol, position_data in self.open_positions.items():
            if symbol in self._last_bar_prices and position_data['quantity'] != 0:
                self.current_holdings_value += self._last_bar_prices[symbol] * position_data['quantity'] # quantity is signed
            elif position_data['quantity'] != 0: # Position exists but no bar price yet (e.g., just opened)
                 # This case is less ideal, means we might not have latest mark-to-market
                 # For shorts, holding value is negative.
                 self.current_holdings_value += position_data['current_segment_entry_price'] * position_data['quantity'] 
        
        self._update_unrealized_pnl() 
        # Total value is cash + current market value of long positions - current market value of short positions (obligations)
        # Since current_holdings_value uses signed quantity, it already accounts for this.
        self.current_total_value = self.current_cash + self.current_holdings_value 
        
        self._portfolio_value_history.append((timestamp, self.current_total_value))
        self.logger.info(
            f"Portfolio Update at {timestamp}: "
            f"Cash={self.current_cash:.2f}, Holdings Value={self.current_holdings_value:.2f}, "
            f"Total Value={self.current_total_value:.2f}, Realized PnL={self.realized_pnl:.2f}" 
        )
        
    def get_current_position_quantity(self, symbol: str) -> float:
        """ Returns the current quantity of the given symbol held. Positive for long, negative for short."""
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
            if symbol not in self.open_positions: 
                continue

            position_data = self.open_positions[symbol]
            if position_data['quantity'] == 0: # Should not happen if cleanup is good, but as a safe guard
                del self.open_positions[symbol]
                continue

            close_price = None
            if data_for_closure and symbol in data_for_closure:
                close_price = data_for_closure[symbol]
            elif symbol in self._last_bar_prices:
                close_price = self._last_bar_prices[symbol]
            else:
                self.logger.warning(f"No close price available for '{symbol}' to synthetically close position. Using its current segment entry price {position_data['current_segment_entry_price']}.")
                close_price = position_data['current_segment_entry_price'] 

            fill_direction = 'SELL' if position_data['quantity'] > 0 else 'BUY'
            
            synthetic_fill_data = {
                'symbol': symbol, 'timestamp': timestamp,
                'quantity_filled': abs(position_data['quantity']), 
                'fill_price': close_price, 'commission': 0.0, 
                'direction': fill_direction,
                'exchange': 'SYNTHETIC_CLOSE', 'fill_id': f"syn_fill_{uuid.uuid4()}", 'order_id': f"syn_ord_{uuid.uuid4()}"
            }
            self.logger.info(f"Synthetically closing {position_data['direction']} {abs(position_data['quantity']):.2f} {symbol} at {close_price:.2f}.")
            self.on_fill(Event(EventType.FILL, synthetic_fill_data)) 
        
        self._update_portfolio_value(timestamp) 
        self.logger.info(f"'{self.name}' finished attempting to close all open positions.")

    def get_final_portfolio_value(self) -> float:
        if self._portfolio_value_history:
            # The last entry in history should reflect the final state after all events and closures
            return self._portfolio_value_history[-1][1] 
        # Fallback if no history (e.g., no events processed, or error before first update)
        # This might happen if optimizer calls it on a fresh, un-run portfolio instance.
        return self.current_total_value 

    def _log_final_performance_summary(self):
        self.logger.info(f"--- {self.name} Final Summary ---")
        self.logger.info(f"Initial Cash: {self.initial_cash:.2f}")
        self.logger.info(f"Final Cash: {self.current_cash:.2f}")
        
        open_pos_display = {sym: data['quantity'] for sym, data in self.open_positions.items() if data.get('quantity', 0) != 0}
        if open_pos_display:
            self.logger.info(f"Final Open Holdings: {open_pos_display}")
        else:
            self.logger.info("Final Open Holdings: None (all positions were closed).")

        self.logger.info(f"Final Portfolio Value: {self.current_total_value:.2f}")
        self.logger.info(f"Total Realized P&L: {self.realized_pnl:.2f}")
        self.logger.info(f"Number of Trade Segments Logged: {len(self._trade_log)}")

        self.logger.info("--- Performance by Regime ---")
        regime_performance: Dict[str, Dict[str, Any]] = {}
        for trade in self._trade_log:
            regime = trade.get('regime', 'unknown') 
            if regime not in regime_performance:
                regime_performance[regime] = {'pnl': 0.0, 'commission': 0.0, 'count': 0, 'wins': 0, 'losses': 0, 'pnl_values': []}
            
            trade_pnl = trade.get('pnl', 0.0)
            trade_commission = trade.get('commission', 0.0)

            regime_performance[regime]['pnl'] += trade_pnl
            regime_performance[regime]['commission'] += trade_commission
            regime_performance[regime]['count'] += 1
            regime_performance[regime]['pnl_values'].append(trade_pnl) # Store net PnL for Sharpe
            if trade_pnl > 0: # Net PnL > 0 is a win
                regime_performance[regime]['wins'] += 1
            elif trade_pnl < 0:
                 regime_performance[regime]['losses'] += 1

        for regime, metrics in regime_performance.items():
            self.logger.info(f"  Regime: {regime}")
            gross_pnl = metrics['pnl'] + metrics['commission'] # Recalculate gross PnL for clarity
            net_pnl_sum = metrics['pnl'] 
            self.logger.info(f"    Total Gross Pnl: {gross_pnl:.2f}") 
            self.logger.info(f"    Trade Segments: {metrics['count']}")
            self.logger.info(f"    Winning Segments: {metrics['wins']}")
            self.logger.info(f"    Losing Segments: {metrics['losses']}")
            self.logger.info(f"    Total Commission: {metrics['commission']:.2f}")
            self.logger.info(f"    Net Pnl Sum: {net_pnl_sum:.2f}") 
            win_rate = (metrics['wins'] / metrics['count']) if metrics['count'] > 0 else 0
            self.logger.info(f"    Win Rate: {win_rate:.2f}")
            
            if metrics['count'] > 1: # Need at least 2 trades for meaningful std dev
                pnl_values = metrics['pnl_values']
                avg_pnl = sum(pnl_values) / metrics['count']
                variance = sum((x - avg_pnl) ** 2 for x in pnl_values) / (metrics['count'] -1) # Sample variance
                std_dev_pnl = variance**0.5 if variance > 0 else 0
                
                # Basic Sharpe: (Mean of PnL per trade) / (Std Dev of PnL per trade)
                # This is a simplified Sharpe, not annualized return over risk-free rate.
                # For a more standard Sharpe, you'd typically use periodic returns (e.g., daily).
                sharpe = (avg_pnl / std_dev_pnl) if std_dev_pnl > 0 else float('inf') if avg_pnl > 0 else 0.0 
                # Simple annualization factor for daily-ish data (sqrt(252)), adjust if segment PnLs are not daily-like
                sharpe_annualized_factor = (252**0.5) if metrics['count'] > 30 else 1 # Arbitrary: only annualize if "enough" data
                sharpe_display = sharpe * sharpe_annualized_factor if std_dev_pnl > 0 else "N/A"

                self.logger.info(f"    Sharpe Ratio (from segment PnLs): {sharpe_display if isinstance(sharpe_display, str) else f'{sharpe_display:.2f}'}")
            else:
                self.logger.info("    Sharpe Ratio (from segment PnLs): N/A (insufficient data or zero volatility)")
