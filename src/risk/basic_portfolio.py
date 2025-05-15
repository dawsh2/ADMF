# src/risk/basic_portfolio.py
import logging
import datetime 
from typing import Dict, Any, Optional, List, Tuple
import uuid
import statistics 
import copy 

from ..core.component import BaseComponent
from ..core.event import Event, EventType
from ..core.exceptions import ComponentError, DependencyNotFoundError

class BasicPortfolio(BaseComponent):
    """
    Manages the portfolio's positions, cash, and tracks performance.
    Enhanced to be regime-aware for P&L attribution.
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
        
        self.total_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.current_holdings_value: float = 0.0
        self.current_total_value: float = self.initial_cash
        self._last_bar_prices: Dict[str, float] = {}

        self._regime_detector: Optional[Any] = None 
        self.regime_detector_key: str = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        self._current_market_regime: Optional[str] = "default" 

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
                    self.logger.info(f"Initial market regime set to: {self._current_market_regime}")
            else:
                self.logger.error(f"Failed to resolve RegimeDetector with key '{self.regime_detector_key}'.")
        except DependencyNotFoundError:
            self.logger.error(f"Dependency '{self.regime_detector_key}' (RegimeDetector) not found. Regime-aware tracking will be limited.")
        except Exception as e:
            self.logger.error(f"Error resolving RegimeDetector '{self.regime_detector_key}': {e}", exc_info=True)

        self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc))
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"BasicPortfolio '{self.name}' setup complete. State: {self.state}")

    def get_current_position_quantity(self, symbol: str) -> float:
        position = self.open_positions.get(symbol)
        if position:
            if position['direction'] == 'LONG':
                return position.get('quantity', 0.0)
            elif position['direction'] == 'SHORT':
                return -position.get('quantity', 0.0) 
        return 0.0
        
    def get_current_position_direction(self, symbol: str) -> Optional[str]:
        position = self.open_positions.get(symbol)
        if position:
            return position.get('direction')
        return None

    def on_classification_change(self, event: Event):
        if not self._regime_detector:
            return

        payload = event.payload
        classifier_name = payload.get('classifier_name')
        new_regime = payload.get('classification')
        event_timestamp = payload.get('timestamp') 
        bar_close_price = payload.get('bar_close_price') 

        if not event_timestamp: 
            self.logger.warning(f"Classification event for '{classifier_name}' missing timestamp. Skipping regime change processing.")
            return

        if classifier_name != self._regime_detector.name:
            return
        
        if new_regime == self._current_market_regime:
            return 

        self.logger.info(f"Portfolio '{self.name}': Market regime changed from '{self._current_market_regime}' to '{new_regime}' at {event_timestamp} (price: {bar_close_price}).")
        
        old_market_regime = self._current_market_regime
        self._current_market_regime = new_regime

        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]
            segment_end_price = bar_close_price if bar_close_price is not None else self._last_bar_prices.get(symbol)

            if segment_end_price is None:
                self.logger.warning(f"Cannot segment trade for {symbol} due to missing price at regime change. Position regime updated, P&L segment not logged.")
                position['current_segment_regime'] = new_regime
                continue

            segment_pnl = 0
            if position['direction'] == 'LONG':
                segment_pnl = (segment_end_price - position['current_segment_entry_price']) * position['quantity']
            elif position['direction'] == 'SHORT':
                segment_pnl = (position['current_segment_entry_price'] - segment_end_price) * position['quantity']
            
            self._log_trade_segment(
                symbol=symbol, trade_id=position['trade_id'],
                segment_entry_timestamp=position['current_segment_entry_timestamp'],
                segment_exit_timestamp=event_timestamp, 
                direction=position['direction'],
                segment_entry_price=position['current_segment_entry_price'],
                segment_exit_price=segment_end_price,
                quantity=position['quantity'], commission=0.0, 
                pnl=segment_pnl, regime=old_market_regime
            )
            
            position['current_segment_entry_price'] = segment_end_price
            position['current_segment_entry_timestamp'] = event_timestamp
            position['current_segment_regime'] = new_regime
            self.logger.info(f"Position for {symbol} segmented. Old segment in '{old_market_regime}' PnL: {segment_pnl:.2f}. New segment starts in '{new_regime}' at {segment_end_price}.")

        self._update_portfolio_value(event_timestamp)


    def on_fill(self, fill_event: Event):
        payload = fill_event.payload
        self.logger.info(f"BasicPortfolio '{self.name}' received FILL event. Payload keys: {list(payload.keys())}. Full payload: {payload}")
        
        symbol = payload['symbol']
        fill_price = float(payload['fill_price'])
        # --- CORRECTED KEY: Use 'quantity_filled' from FILL event payload ---
        quantity = float(payload['quantity_filled']) 
        # --------------------------------------------------------------------
        direction = payload['direction'] 
        commission = float(payload.get('commission', 0.0))
        fill_timestamp = payload.get('timestamp') 
        
        if not isinstance(fill_timestamp, datetime.datetime):
            self.logger.warning(f"Fill event for {symbol} has non-datetime timestamp '{fill_timestamp}' of type {type(fill_timestamp)}. Attempting conversion or fallback.")
            try:
                if hasattr(fill_timestamp, 'to_pydatetime'): 
                    fill_timestamp = fill_timestamp.to_pydatetime()
                elif isinstance(fill_timestamp, str): # Basic ISO format check
                    fill_timestamp = datetime.datetime.fromisoformat(fill_timestamp.replace("Z", "+00:00"))
                
                if fill_timestamp.tzinfo is None: 
                    fill_timestamp = fill_timestamp.replace(tzinfo=datetime.timezone.utc)
            except Exception as e:
                self.logger.error(f"Error processing fill_timestamp '{payload.get('timestamp')}': {e}. Using current time.", exc_info=True)
                fill_timestamp = datetime.datetime.now(datetime.timezone.utc)

        self.current_cash -= commission
        active_regime = self._current_market_regime 

        if symbol not in self.open_positions: 
            if direction == 'BUY':
                self.open_positions[symbol] = {
                    'quantity': quantity, 'direction': 'LONG',
                    'initial_entry_price': fill_price, 'initial_entry_timestamp': fill_timestamp,
                    'trade_id': str(uuid.uuid4()), 
                    'current_segment_entry_price': fill_price, 'current_segment_entry_timestamp': fill_timestamp,
                    'current_segment_regime': active_regime, 'initial_entry_regime': active_regime
                }
                self.current_cash -= quantity * fill_price
                self.logger.info(f"Opened LONG {quantity} {symbol} at {fill_price} in regime '{active_regime}'. Trade ID: {self.open_positions[symbol]['trade_id']}")
            elif direction == 'SELL': 
                self.open_positions[symbol] = {
                    'quantity': quantity, 'direction': 'SHORT',
                    'initial_entry_price': fill_price, 'initial_entry_timestamp': fill_timestamp,
                    'trade_id': str(uuid.uuid4()),
                    'current_segment_entry_price': fill_price, 'current_segment_entry_timestamp': fill_timestamp,
                    'current_segment_regime': active_regime, 'initial_entry_regime': active_regime
                }
                self.current_cash += quantity * fill_price
                self.logger.info(f"Opened SHORT {quantity} {symbol} at {fill_price} in regime '{active_regime}'. Trade ID: {self.open_positions[symbol]['trade_id']}")
        else: 
            position = self.open_positions[symbol]
            
            if position['direction'] == 'LONG':
                if direction == 'SELL': 
                    sold_quantity = min(quantity, position['quantity'])
                    trade_pnl = (fill_price - position['current_segment_entry_price']) * sold_quantity
                    
                    self._log_trade_segment(
                        symbol=symbol, trade_id=position['trade_id'],
                        segment_entry_timestamp=position['current_segment_entry_timestamp'],
                        segment_exit_timestamp=fill_timestamp, direction=position['direction'],
                        segment_entry_price=position['current_segment_entry_price'],
                        segment_exit_price=fill_price, quantity=sold_quantity,
                        commission=commission * (sold_quantity / quantity) if quantity != 0 else commission, 
                        pnl=trade_pnl, regime=position['current_segment_regime'] 
                    )
                    self.current_cash += sold_quantity * fill_price
                    position['quantity'] -= sold_quantity
                    if position['quantity'] < 1e-9: 
                        del self.open_positions[symbol]
                        self.logger.info(f"Closed LONG {symbol}. Segment PnL: {trade_pnl:.2f} in regime '{position['current_segment_regime']}'.")
                    else: 
                        self.logger.info(f"Partially closed LONG {sold_quantity} {symbol}. Segment PnL: {trade_pnl:.2f}. Remaining: {position['quantity']}. New segment starts in regime '{active_regime}'.")
                        position['current_segment_entry_price'] = fill_price 
                        position['current_segment_entry_timestamp'] = fill_timestamp
                        position['current_segment_regime'] = active_regime
                elif direction == 'BUY': 
                    segment_close_pnl = (fill_price - position['current_segment_entry_price']) * position['quantity']
                    self._log_trade_segment(symbol, position['trade_id'], position['current_segment_entry_timestamp'], fill_timestamp, position['direction'], position['current_segment_entry_price'], fill_price, position['quantity'], 0, segment_close_pnl, position['current_segment_regime'])

                    new_total_quantity = position['quantity'] + quantity
                    new_avg_initial_price = ((position['initial_entry_price'] * position['quantity']) + (fill_price * quantity)) / new_total_quantity
                    position['initial_entry_price'] = new_avg_initial_price
                    position['quantity'] = new_total_quantity
                    self.current_cash -= quantity * fill_price
                    
                    position['current_segment_entry_price'] = fill_price 
                    position['current_segment_entry_timestamp'] = fill_timestamp
                    position['current_segment_regime'] = active_regime
                    self.logger.info(f"Increased LONG {symbol}. New avg initial entry: {new_avg_initial_price:.2f}. New segment in '{active_regime}'.")

            elif position['direction'] == 'SHORT':
                if direction == 'BUY': 
                    covered_quantity = min(quantity, position['quantity'])
                    trade_pnl = (position['current_segment_entry_price'] - fill_price) * covered_quantity
                    
                    self._log_trade_segment(
                        symbol=symbol, trade_id=position['trade_id'],
                        segment_entry_timestamp=position['current_segment_entry_timestamp'],
                        segment_exit_timestamp=fill_timestamp, direction=position['direction'],
                        segment_entry_price=position['current_segment_entry_price'],
                        segment_exit_price=fill_price, quantity=covered_quantity,
                        commission=commission * (covered_quantity / quantity) if quantity != 0 else commission, 
                        pnl=trade_pnl, regime=position['current_segment_regime']
                    )
                    self.current_cash -= covered_quantity * fill_price
                    position['quantity'] -= covered_quantity
                    if position['quantity'] < 1e-9: 
                        del self.open_positions[symbol]
                        self.logger.info(f"Covered SHORT {symbol}. Segment PnL: {trade_pnl:.2f} in regime '{position['current_segment_regime']}'.")
                    else: 
                        self.logger.info(f"Partially covered SHORT {covered_quantity} {symbol}. Segment PnL: {trade_pnl:.2f}. Remaining: {position['quantity']}. New segment starts in regime '{active_regime}'.")
                        position['current_segment_entry_price'] = fill_price
                        position['current_segment_entry_timestamp'] = fill_timestamp
                        position['current_segment_regime'] = active_regime
                elif direction == 'SELL': 
                    segment_close_pnl = (position['current_segment_entry_price'] - fill_price) * position['quantity']
                    self._log_trade_segment(symbol, position['trade_id'], position['current_segment_entry_timestamp'], fill_timestamp, position['direction'], position['current_segment_entry_price'], fill_price, position['quantity'], 0, segment_close_pnl, position['current_segment_regime'])

                    new_total_quantity = position['quantity'] + quantity
                    new_avg_initial_price = ((position['initial_entry_price'] * position['quantity']) + (fill_price * quantity)) / new_total_quantity
                    position['initial_entry_price'] = new_avg_initial_price
                    position['quantity'] = new_total_quantity
                    self.current_cash += quantity * fill_price

                    position['current_segment_entry_price'] = fill_price
                    position['current_segment_entry_timestamp'] = fill_timestamp
                    position['current_segment_regime'] = active_regime
                    self.logger.info(f"Increased SHORT {symbol}. New avg initial entry: {new_avg_initial_price:.2f}. New segment in '{active_regime}'.")
        
        self._update_portfolio_value(fill_timestamp)

    def _log_trade_segment(self, symbol: str, trade_id: str, 
                           segment_entry_timestamp: datetime.datetime, segment_exit_timestamp: datetime.datetime,
                           direction: str, segment_entry_price: float, segment_exit_price: float,
                           quantity: float, commission: float, pnl: float, regime: str):
        segment_id = str(uuid.uuid4())
        trade_record = {
            'symbol': symbol, 'trade_id': trade_id, 'segment_id': segment_id,
            'entry_timestamp': segment_entry_timestamp, 'exit_timestamp': segment_exit_timestamp,
            'direction': direction, 'segment_entry_price': segment_entry_price, 
            'segment_exit_price': segment_exit_price, 'quantity': quantity,
            'commission': commission, 'pnl': pnl, 'regime': regime
        }
        self._trade_log.append(trade_record)
        self.realized_pnl += (pnl - commission) 
        self.logger.debug(f"Logged trade segment for {symbol}: ID={segment_id}, PnL={pnl:.2f}, Commission={commission:.2f}, Net PnL={(pnl-commission):.2f}, Regime='{regime}'")

    def on_bar(self, bar_event: Event):
        payload = bar_event.payload
        symbol = payload['symbol']
        close_price = float(payload['close'])
        timestamp = payload.get('timestamp') 

        if not isinstance(timestamp, datetime.datetime):
            try:
                if hasattr(timestamp, 'to_pydatetime'): 
                    timestamp = timestamp.to_pydatetime()
                if timestamp.tzinfo is None: 
                    timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
            except Exception as e:
                self.logger.warning(f"Bar event for {symbol} has invalid timestamp format '{payload.get('timestamp')}': {e}. Using current time.")
                timestamp = datetime.datetime.now(datetime.timezone.utc)

        self._last_bar_prices[symbol] = close_price
        self._update_portfolio_value(timestamp)

    def _update_portfolio_value(self, timestamp: Optional[datetime.datetime]):
        if timestamp is None: 
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            self.logger.debug(f"Portfolio value update called with None timestamp, using current time: {timestamp}")

        self.unrealized_pnl = 0.0
        self.current_holdings_value = 0.0
        
        for symbol, position in self.open_positions.items():
            last_price = self._last_bar_prices.get(symbol)
            if last_price is not None:
                market_value = position['quantity'] * last_price
                # For short positions, market_value is negative, representing liability
                if position['direction'] == 'SHORT':
                    self.current_holdings_value -= market_value 
                else:
                    self.current_holdings_value += market_value
                
                if position['direction'] == 'LONG':
                    self.unrealized_pnl += (last_price - position['current_segment_entry_price']) * position['quantity']
                elif position['direction'] == 'SHORT':
                    self.unrealized_pnl += (position['current_segment_entry_price'] - last_price) * position['quantity']
        
        self.current_total_value = self.current_cash + self.current_holdings_value
        current_total_pnl = self.realized_pnl + self.unrealized_pnl

        if self._event_bus:
            portfolio_status = {
                'timestamp': timestamp, 'cash': self.current_cash,
                'holdings_value': self.current_holdings_value, 'total_value': self.current_total_value,
                'realized_pnl': self.realized_pnl, 'unrealized_pnl': self.unrealized_pnl,
                'total_pnl': current_total_pnl, 'open_positions_count': len(self.open_positions)
            }
            # self._event_bus.publish(Event(EventType.PORTFOLIO_UPDATE, portfolio_status)) 
        
        self.logger.info(
            f"Portfolio Update at {timestamp}: "
            f"Cash={self.current_cash:.2f}, Holdings Value={self.current_holdings_value:.2f}, "
            f"Total Value={self.current_total_value:.2f}, Realized PnL={self.realized_pnl:.2f}"
        )
        
    def close_all_open_positions(self, timestamp: datetime.datetime):
        self.logger.info(f"'{self.name}' initiating closing of all open positions at {timestamp}.")
        for symbol in list(self.open_positions.keys()): 
            position = self.open_positions[symbol]
            # Use current_segment_entry_price as a fallback if last_price is not available for some reason
            last_price = self._last_bar_prices.get(symbol, position['current_segment_entry_price']) 

            if last_price is None: 
                self.logger.warning(f"No price for {symbol} to close position. Position might remain open in records.")
                continue

            final_segment_pnl = 0
            commission_on_close = 0.0 # Actual commission would come from a fill event. For synthetic close, assume 0.

            if position['direction'] == 'LONG':
                final_segment_pnl = (last_price - position['current_segment_entry_price']) * position['quantity']
                self.current_cash += position['quantity'] * last_price
            elif position['direction'] == 'SHORT':
                final_segment_pnl = (position['current_segment_entry_price'] - last_price) * position['quantity']
                self.current_cash -= position['quantity'] * last_price # Cash out for buying back short
            
            self._log_trade_segment(
                symbol=symbol, trade_id=position['trade_id'],
                segment_entry_timestamp=position['current_segment_entry_timestamp'],
                segment_exit_timestamp=timestamp, direction=position['direction'],
                segment_entry_price=position['current_segment_entry_price'],
                segment_exit_price=last_price, quantity=position['quantity'],
                commission=commission_on_close, 
                pnl=final_segment_pnl, regime=position['current_segment_regime']
            )
            self.logger.info(f"Synthetically closed {position['direction']} {position['quantity']} {symbol} at {last_price}. Final segment PnL: {final_segment_pnl:.2f} in regime '{position['current_segment_regime']}'.")
            del self.open_positions[symbol]

        self._update_portfolio_value(timestamp) 
        self.logger.info(f"'{self.name}' finished attempting to close all open positions.")

    def get_performance_by_regime(self) -> Dict[str, Dict[str, Any]]:
        performance_by_regime: Dict[str, Dict[str, Any]] = {}
        for segment in self._trade_log:
            regime = segment.get('regime', 'unknown_regime')
            if regime not in performance_by_regime:
                performance_by_regime[regime] = {
                    'total_gross_pnl': 0.0, 'trade_segments': 0, 'winning_segments': 0,
                    'losing_segments': 0, 'total_commission': 0.0, 'net_pnl_values': [] 
                }
            
            stats = performance_by_regime[regime]
            segment_pnl = segment['pnl']
            segment_commission = segment.get('commission', 0.0)
            
            stats['total_gross_pnl'] += segment_pnl 
            stats['trade_segments'] += 1
            stats['total_commission'] += segment_commission
            net_segment_pnl = segment_pnl - segment_commission
            stats['net_pnl_values'].append(net_segment_pnl)

            if net_segment_pnl > 0:
                stats['winning_segments'] += 1
            elif net_segment_pnl < 0:
                stats['losing_segments'] += 1
        
        for regime, stats in performance_by_regime.items():
            stats['net_pnl_sum'] = sum(stats['net_pnl_values']) 
            if stats['trade_segments'] > 0:
                stats['win_rate'] = stats['winning_segments'] / stats['trade_segments'] if stats['trade_segments'] > 0 else 0.0
                
                pnl_values = stats.pop('net_pnl_values', []) 
                if len(pnl_values) > 1:
                    mean_pnl = statistics.mean(pnl_values)
                    std_dev_pnl = statistics.stdev(pnl_values)
                    stats['sharpe_ratio'] = (mean_pnl / std_dev_pnl) * (252**0.5) if std_dev_pnl > 1e-9 else 0.0 
                elif len(pnl_values) == 1: 
                    stats['sharpe_ratio'] = float('inf') if pnl_values[0] > 0 else float('-inf') if pnl_values[0] < 0 else 0.0
                else: 
                    stats['sharpe_ratio'] = 0.0
            else:
                stats['win_rate'] = 0.0
                stats['sharpe_ratio'] = 0.0
        return performance_by_regime
        
    def start(self):
        super().start()
        if self.state == BaseComponent.STATE_INITIALIZED:
            self.state = BaseComponent.STATE_STARTED
            self.logger.info(f"BasicPortfolio '{self.name}' started. Monitoring FILL and BAR events...")
        elif self.state == BaseComponent.STATE_STARTED:
             self.logger.info(f"BasicPortfolio '{self.name}' already started.")
        else:
            self.logger.warning(f"BasicPortfolio '{self.name}' not starting, current state: {self.state}")

    def stop(self):
        if self.state not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED, BaseComponent.STATE_STOPPED]:
            self.logger.info("--- BasicPortfolio Final Summary ---")
            self.logger.info(f"Initial Cash: {self.initial_cash:.2f}")
            self.logger.info(f"Final Cash: {self.current_cash:.2f}")
            if self.open_positions:
                self.logger.info("Final Open Positions (should be closed by close_all_open_positions):")
                for symbol, pos_data in self.open_positions.items():
                    self.logger.info(f"  {symbol}: Quantity={pos_data['quantity']}, Direction={pos_data['direction']}, Entry={pos_data['initial_entry_price']:.2f}")
            else:
                self.logger.info("Final Holdings: None (all positions closed)")
            self.logger.info(f"Final Portfolio Value: {self.current_total_value:.2f}")
            self.logger.info(f"Total Realized P&L: {self.realized_pnl:.2f}") 
            self.logger.info(f"Number of Trade Segments Logged: {len(self._trade_log)}")

            try:
                perf_by_regime = self.get_performance_by_regime()
                self.logger.info("--- Performance by Regime ---")
                if perf_by_regime:
                    for regime, stats in perf_by_regime.items():
                        self.logger.info(f"  Regime: {regime}")
                        for stat_name, stat_value in stats.items():
                            if isinstance(stat_value, float):
                                self.logger.info(f"    {stat_name.replace('_', ' ').title()}: {stat_value:.2f}")
                            else:
                                self.logger.info(f"    {stat_name.replace('_', ' ').title()}: {stat_value}")
                else:
                    self.logger.info("  No trade segments logged to report performance by regime.")
            except Exception as e:
                self.logger.error(f"Error generating performance by regime report: {e}", exc_info=True)

        super().stop() 
        self.logger.info(f"Stopping BasicPortfolio '{self.name}'...")
        if self._event_bus and hasattr(self._event_bus, 'unsubscribe'):
            try:
                self._event_bus.unsubscribe(EventType.FILL, self.on_fill)
                self._event_bus.unsubscribe(EventType.BAR, self.on_bar)
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.info(f"'{self.name}' unsubscribed from FILL, BAR, and CLASSIFICATION events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing {self.name} from events: {e}", exc_info=True)
        
        if self.state != BaseComponent.STATE_STOPPED:
             self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"BasicPortfolio '{self.name}' stopped. State: {self.state}")

    def get_current_holdings(self) -> List[Dict[str, Any]]:
        holdings = []
        for symbol, data in self.open_positions.items():
            holdings.append({
                "symbol": symbol,
                "quantity": data['quantity'],
                "direction": data['direction'],
                "initial_entry_price": data['initial_entry_price'], 
                "current_segment_entry_price": data['current_segment_entry_price'],
                "current_segment_regime": data['current_segment_regime']
            })
        return holdings

    def get_current_cash(self) -> float:
        return self.current_cash

    def get_last_processed_timestamp(self) -> Optional[datetime.datetime]:
        if self._trade_log:
            latest_exit_ts = None
            for segment in self._trade_log:
                exit_ts = segment.get('exit_timestamp')
                if isinstance(exit_ts, datetime.datetime):
                    if latest_exit_ts is None or exit_ts > latest_exit_ts:
                        latest_exit_ts = exit_ts
            return latest_exit_ts
            
        elif self.open_positions:
            latest_entry_ts = None
            for pos in self.open_positions.values():
                entry_ts = pos.get('current_segment_entry_timestamp')
                if isinstance(entry_ts, datetime.datetime):
                    if latest_entry_ts is None or entry_ts > latest_entry_ts:
                        latest_entry_ts = entry_ts
            return latest_entry_ts
        return None

