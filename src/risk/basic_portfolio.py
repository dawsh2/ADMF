# src/risk/basic_portfolio.py
import logging
import datetime
import uuid 
from typing import Dict, List, Any, Tuple, Optional

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError

class BasicPortfolio(BaseComponent):
    """
    A basic portfolio component that tracks cash, holdings, and portfolio value.
    It listens to FILL events to update positions and BAR events to mark-to-market.
    Refined P&L and cost basis calculations.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str):
        super().__init__(instance_name, config_loader, component_config_key)
        
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error("EventBus instance is required for BasicPortfolio.")
            raise ConfigurationError("EventBus instance is required for BasicPortfolio.")

        self._initial_cash: float = self.get_specific_config("initial_cash")
        if self._initial_cash is None or not isinstance(self._initial_cash, (int, float)) or self._initial_cash < 0:
            raise ConfigurationError(f"Portfolio '{self.name}': 'initial_cash' must be a non-negative number. Got: {self._initial_cash}")

        self._current_cash: float = self._initial_cash
        self._holdings: Dict[str, Dict[str, float]] = {} # symbol -> {"quantity": float, "avg_cost_price": float, "last_market_price": float}
        self._market_prices: Dict[str, float] = {}
        self._realized_pnl: float = 0.0
        self._trade_log: List[Dict[str, Any]] = []
        self._portfolio_value_history: List[Tuple[datetime.datetime, float]] = []

        self.logger.info(
            f"BasicPortfolio '{self.name}' initialized with initial cash: {self._initial_cash:.2f}"
        )

    def setup(self):
        self.logger.info(f"Setting up BasicPortfolio '{self.name}'...")
        self._current_cash = self._initial_cash 
        self._holdings.clear()
        self._market_prices.clear()
        self._realized_pnl = 0.0
        self._trade_log.clear()
        self._portfolio_value_history.clear()

        self._event_bus.subscribe(EventType.FILL, self._on_fill_event)
        self._event_bus.subscribe(EventType.BAR, self._on_bar_event)

        self.logger.info(f"'{self.name}' subscribed to FILL and BAR events.")
        self._update_and_log_portfolio_value(datetime.datetime.now(datetime.timezone.utc))
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"BasicPortfolio '{self.name}' setup complete. State: {self.state}")        

    def _on_fill_event(self, fill_event: Event):
        if fill_event.event_type != EventType.FILL:
            return

        fill_data = fill_event.payload
        symbol = fill_data.get("symbol")
        quantity_filled = float(fill_data.get("quantity_filled", 0)) # Absolute quantity in the fill
        fill_price = float(fill_data.get("fill_price", 0))
        commission = float(fill_data.get("commission", 0)) # Total commission for this fill
        direction = fill_data.get("direction", "").upper() # "BUY" or "SELL"
        timestamp = fill_data.get("timestamp", datetime.datetime.now(datetime.timezone.utc))

        if not all([symbol, quantity_filled > 0, fill_price > 0, direction in ["BUY", "SELL"]]):
            self.logger.error(f"'{self.name}' received invalid FILL event data: {fill_data}")
            return

        self.logger.info(
            f"'{self.name}' processing FILL: {direction} {quantity_filled} {symbol} at {fill_price:.2f}, "
            f"Commission: {commission:.2f}"
        )

        trade_details = {**fill_data, "realized_pnl_from_this_fill": 0.0}
        
        # Ensure holding structure exists
        if symbol not in self._holdings:
            self._holdings[symbol] = {"quantity": 0.0, "avg_cost_price": 0.0, "last_market_price": fill_price}

        original_quantity = self._holdings[symbol]["quantity"]
        original_avg_cost = self._holdings[symbol]["avg_cost_price"]
        
        current_fill_pnl = 0.0

        if direction == "BUY":
            self._current_cash -= (quantity_filled * fill_price + commission)

            if original_quantity < 0: # Was short, this BUY is (partially or fully) covering
                qty_to_cover = min(abs(original_quantity), quantity_filled)
                # P&L from covering short: (short_sale_price - buy_back_price) * quantity
                # original_avg_cost for a short position is the average price it was sold at.
                pnl_on_cover = (original_avg_cost - fill_price) * qty_to_cover
                current_fill_pnl += pnl_on_cover
                self.logger.info(f"Realized P&L from covering {qty_to_cover} short {symbol}: {pnl_on_cover:.2f}")
                
                # Update quantity from covering
                self._holdings[symbol]["quantity"] += qty_to_cover
                remaining_fill_qty = quantity_filled - qty_to_cover

                if self._holdings[symbol]["quantity"] == 0: # Exactly covered the short, now flat
                    self._holdings[symbol]["avg_cost_price"] = 0.0
                
                if remaining_fill_qty > 0: # Flipped from short to long
                    self._holdings[symbol]["quantity"] = remaining_fill_qty # This is the new long quantity
                    self._holdings[symbol]["avg_cost_price"] = fill_price # Cost basis of new long
            
            else: # Was flat or long, this BUY adds to long or opens long
                new_total_value_of_holding = (original_quantity * original_avg_cost) + (quantity_filled * fill_price)
                self._holdings[symbol]["quantity"] += quantity_filled
                if self._holdings[symbol]["quantity"] != 0:
                    self._holdings[symbol]["avg_cost_price"] = new_total_value_of_holding / self._holdings[symbol]["quantity"]
                else: # Should not happen if original_quantity >=0 and quantity_filled > 0
                    self._holdings[symbol]["avg_cost_price"] = 0.0
        
        elif direction == "SELL":
            self._current_cash += (quantity_filled * fill_price - commission)

            if original_quantity > 0: # Was long, this SELL is (partially or fully) closing
                qty_to_close_long = min(original_quantity, quantity_filled)
                # P&L from closing long: (sell_price - buy_price) * quantity
                pnl_on_close_long = (fill_price - original_avg_cost) * qty_to_close_long
                current_fill_pnl += pnl_on_close_long
                self.logger.info(f"Realized P&L from closing {qty_to_close_long} long {symbol}: {pnl_on_close_long:.2f}")

                self._holdings[symbol]["quantity"] -= qty_to_close_long
                remaining_fill_qty = quantity_filled - qty_to_close_long

                if self._holdings[symbol]["quantity"] == 0: # Exactly closed the long, now flat
                    self._holdings[symbol]["avg_cost_price"] = 0.0
                
                if remaining_fill_qty > 0: # Flipped from long to short
                    self._holdings[symbol]["quantity"] = -remaining_fill_qty # This is the new short quantity
                    self._holdings[symbol]["avg_cost_price"] = fill_price # Avg price this short was entered at
            
            else: # Was flat or short, this SELL adds to short or opens short
                # For short positions, avg_cost_price stores the average price shares were sold at.
                new_total_sell_value = (abs(original_quantity) * original_avg_cost) + (quantity_filled * fill_price)
                self._holdings[symbol]["quantity"] -= quantity_filled # Becomes more negative
                if self._holdings[symbol]["quantity"] != 0:
                    self._holdings[symbol]["avg_cost_price"] = new_total_sell_value / abs(self._holdings[symbol]["quantity"])
                else: # Should not happen here
                     self._holdings[symbol]["avg_cost_price"] = 0.0

        self._realized_pnl += current_fill_pnl
        trade_details["realized_pnl_from_this_fill"] = current_fill_pnl
        
        self._holdings[symbol]["last_market_price"] = fill_price
        self._trade_log.append(trade_details)
        self._update_and_log_portfolio_value(timestamp)


    def _on_bar_event(self, bar_event: Event):
        if bar_event.event_type != EventType.BAR:
            return
        
        bar_data = bar_event.payload
        symbol = bar_data.get("symbol")
        close_price_val = bar_data.get("close")
        timestamp = bar_data.get("timestamp")

        if not symbol or close_price_val is None or timestamp is None:
            self.logger.debug(f"'{self.name}' ignoring BAR event with missing data: {bar_data}")
            return 

        try:
            close_price = float(close_price_val)
        except ValueError:
            self.logger.warning(f"Could not convert close price '{close_price_val}' to float for {symbol}")
            return

        self._market_prices[symbol] = close_price 

        if symbol in self._holdings and self._holdings[symbol]["quantity"] != 0:
            self._holdings[symbol]["last_market_price"] = close_price
        # Always update portfolio value on new bar for consistent equity curve,
        # even if holdings for this specific symbol are zero or don't exist.
        self._update_and_log_portfolio_value(timestamp)


    def _update_and_log_portfolio_value(self, timestamp: datetime.datetime):
        current_positions_value = 0.0
        for symbol, data in self._holdings.items():
            quantity = data.get("quantity", 0.0)
            market_price = data.get("last_market_price", self._market_prices.get(symbol, 0.0)) 
            current_positions_value += quantity * market_price # quantity can be negative for shorts

        total_value = self._current_cash + current_positions_value
        
        if timestamp is None: 
            timestamp = datetime.datetime.now(datetime.timezone.utc)

        self._portfolio_value_history.append((timestamp, total_value))
        
        self.logger.info(
            f"Portfolio Update at {timestamp}: Cash={self._current_cash:.2f}, "
            f"Positions Value={current_positions_value:.2f}, Total Value={total_value:.2f}, "
            f"Realized PnL={self._realized_pnl:.2f}"
        )

    def _generate_unique_id(self, prefix=""):
        return f"{prefix}{uuid.uuid4().hex[:10]}"

    def close_all_open_positions(self, closing_timestamp: datetime.datetime):
        self.logger.info(f"'{self.name}' initiating closing of all open positions at {closing_timestamp}.")
        if not self._event_bus:
            self.logger.error(f"'{self.name}' cannot close positions: EventBus is not available.")
            return

        symbols_with_positions = list(self._holdings.keys())

        for symbol in symbols_with_positions:
            holding_data = self._holdings.get(symbol)
            if not holding_data: continue

            current_quantity = holding_data.get("quantity", 0.0)
            if current_quantity == 0:
                continue

            order_direction = "SELL" if current_quantity > 0 else "BUY"
            quantity_to_close = abs(current_quantity)
            closing_price = holding_data.get("last_market_price", self._market_prices.get(symbol))

            if closing_price is None:
                self.logger.warning(
                    f"'{self.name}': Cannot determine closing price for {symbol}. Skipping closing order."
                )
                continue
            
            order_id = self._generate_unique_id(prefix=f"close_{symbol}_")
            order_payload = {
                "order_id": order_id,
                "symbol": symbol,
                "order_type": "MARKET",
                "direction": order_direction,
                "quantity": quantity_to_close,
                "simulated_fill_price": closing_price,
                "timestamp": closing_timestamp, 
                "strategy_id": "PortfolioClosure"
            }
            
            close_order_event = Event(EventType.ORDER, order_payload)
            self._event_bus.publish(close_order_event)
            self.logger.info(
                f"'{self.name}' published ORDER to close {order_direction} {quantity_to_close} {symbol} "
                f"at estimated price {closing_price:.2f}. OrderID: {order_id}"
            )
        self.logger.info(f"'{self.name}' finished attempting to close all open positions.")

    def get_current_position_quantity(self, symbol: str) -> float:
        holding = self._holdings.get(symbol)
        if holding:
            return holding.get("quantity", 0.0)
        return 0.0

    def get_last_processed_timestamp(self) -> Optional[datetime.datetime]:
        if self._portfolio_value_history:
            return self._portfolio_value_history[-1][0]
        return None

    def get_final_portfolio_value(self) -> Optional[float]:
        """
        Returns the final total portfolio value recorded in the history.
        Returns None if no history is available.
        """
        if self._portfolio_value_history:
            return self._portfolio_value_history[-1][1] # Get the value from the last tuple (timestamp, value)
        elif self.state == BaseComponent.STATE_INITIALIZED or self.state == BaseComponent.STATE_STOPPED :
             # If history is empty but component is initialized/stopped, it means only initial cash state might exist
             # or no BAR events leading to history. Default to current total value.
            current_positions_value = 0.0
            for symbol, data in self._holdings.items():
                quantity = data.get("quantity", 0.0)
                market_price = data.get("last_market_price", self._market_prices.get(symbol, 0.0)) 
                current_positions_value += quantity * market_price
            return self._current_cash + current_positions_value
        return None

    def get_total_realized_pnl(self) -> float:
        """Returns the total realized P&L accumulated."""
        return self._realized_pnl    

    def start(self):
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start {self.name} from state '{self.state}'. Expected INITIALIZED.")
            return
        self.logger.info(f"{self.name} started. Monitoring FILL and BAR events...")
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"Stopping {self.name}...")
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.FILL, self._on_fill_event)
                self._event_bus.unsubscribe(EventType.BAR, self._on_bar_event)
                self.logger.info(f"'{self.name}' attempted to unsubscribe from FILL and BAR events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.name}': {e}")
        
        final_log_timestamp = self.get_last_processed_timestamp() or datetime.datetime.now(datetime.timezone.utc)
        self._update_and_log_portfolio_value(final_log_timestamp)

        self.logger.info(f"--- {self.name} Final Summary ---")
        self.logger.info(f"Initial Cash: {self._initial_cash:.2f}")
        self.logger.info(f"Final Cash: {self._current_cash:.2f}")
        
        final_holdings_display = {
            sym: data for sym, data in self._holdings.items() if data.get("quantity", 0.0) != 0.0
        }
        if not final_holdings_display:
             self.logger.info("Final Holdings: None (all positions closed)")
        else:
            self.logger.info(f"Final Holdings (should be empty if closure worked): {final_holdings_display}")

        if self._portfolio_value_history:
            self.logger.info(f"Final Portfolio Value: {self._portfolio_value_history[-1][1]:.2f}")
        else:
            self.logger.info(f"Final Portfolio Value: {self._current_cash:.2f} (initial cash, no history)")

        self.logger.info(f"Total Realized P&L: {self._realized_pnl:.2f}")
        self.logger.info(f"Number of Trades Logged (Fill Events): {len(self._trade_log)}")
        
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"{self.name} stopped. State: {self.state}")
