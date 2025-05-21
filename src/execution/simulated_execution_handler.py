# src/execution/simulated_execution_handler.py
import logging
import datetime
import uuid # For generating unique IDs

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError

class SimulatedExecutionHandler(BaseComponent):
    """
    Simulates the execution of orders based on ORDER events.
    It receives ORDER events, simulates their fill based on data within the order
    (like a simulated_fill_price for market orders), and then publishes FILL events.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str):
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error("EventBus instance is required.")
            raise ConfigurationError("EventBus instance is required for SimulatedExecutionHandler.")

        # Default quantity might be less relevant if orders always specify quantity.
        # Kept for now but its usage might change.
        self._default_quantity = self.get_specific_config("default_quantity", 100)
        self._commission_per_trade = self.get_specific_config("commission_per_trade", 0.0)
        self._passthrough_mode = self.get_specific_config("passthrough", False) # Passthrough might mean not generating fills.

        if not isinstance(self._default_quantity, int) or self._default_quantity <= 0:
            # This validation might be removed if orders always contain explicit quantities.
            self.logger.warning(f"SimExecHandler: 'default_quantity' ({self._default_quantity}) might not be used if orders always specify quantity.")
        if not isinstance(self._commission_per_trade, (int, float)) or self._commission_per_trade < 0:
            raise ConfigurationError(f"SimExecHandler: 'commission_per_trade' must be a non-negative number. Got {self._commission_per_trade}")

        self.logger.info(
            f"{self.name} configured. Default Qty (if order lacks it): {self._default_quantity}, "
            f"Commission: {self._commission_per_trade}, Passthrough: {self._passthrough_mode}"
        )

    def setup(self):
        self.logger.info(f"Setting up {self.name}...")
        # Unsubscribe from SIGNAL if previously subscribed (e.g. during development)
        # This is more of a safeguard if you were experimenting. Clean setup subscribes once.
        # try:
        #     self._event_bus.unsubscribe(EventType.SIGNAL, self._on_signal_event) # Old method name
        # except ValueError: # If it was never subscribed to SIGNAL with that specific method
        #     pass
            
        self._event_bus.subscribe(EventType.ORDER, self._on_order_event) # NEW: Subscribe to ORDER
        self.logger.info(f"'{self.name}' subscribed to ORDER events.")
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"{self.name} setup complete. State: {self.state}")

    def _generate_unique_id(self, prefix=""):
        return f"{prefix}{uuid.uuid4().hex[:8]}"

    def _on_order_event(self, order_event: Event): # Renamed from _on_signal_event
        self.logger.debug(f"{self.name} received ORDER event")
        if order_event.event_type != EventType.ORDER:
            return

        order_data = order_event.payload
        order_id = order_data.get("order_id", self._generate_unique_id("ord_missing_"))
        symbol = order_data.get("symbol")
        order_type = order_data.get("order_type", "MARKET").upper()
        direction = order_data.get("direction") # Should be 'BUY' or 'SELL'
        quantity = order_data.get("quantity")
        
        # For simulated fill price:
        # 1. Check for 'simulated_fill_price' (used by portfolio close_all)
        # 2. Fallback to 'price_limit' if it's a limit order (though we mostly use MARKET for now)
        # 3. Fallback to a general 'price' field if provided in the order.
        # 4. If none, this simulation handler would need a market data feed or error out for market orders.
        #    For MVP, we'll require some price indication in the order for simulation.
        fill_price = order_data.get("simulated_fill_price")
        if fill_price is None:
            fill_price = order_data.get("price") # A general price field
        if fill_price is None and order_type == "LIMIT":
            fill_price = order_data.get("price_limit")


        order_timestamp = order_data.get("timestamp", datetime.datetime.now(datetime.timezone.utc))
        strategy_id = order_data.get("strategy_id", "UnknownStrategy")

        self.logger.info(
            f"{self.name} received ORDER: ID={order_id}, {direction} {quantity} {symbol} "
            f"Type={order_type} at (target/simulated price) {fill_price if fill_price is not None else 'N/A'} "
            f"from {strategy_id}"
        )

        if not all([symbol, direction, quantity is not None and quantity > 0]):
            self.logger.error(f"{self.name} received invalid or incomplete ORDER data: {order_data}")
            return

        if fill_price is None and order_type == "MARKET":
            self.logger.error(
                f"{self.name}: Market order ID={order_id} for {symbol} has no fill price indication "
                f"('simulated_fill_price' or 'price'). Cannot simulate fill. Order: {order_data}"
            )
            # Optionally, publish a REJECTED_ORDER event or similar, or just log and return.
            return
        elif fill_price is None and order_type == "LIMIT":
             self.logger.error(
                f"{self.name}: Limit order ID={order_id} for {symbol} has no price_limit. "
                f"Cannot simulate fill. Order: {order_data}"
            )
             return


        if self._passthrough_mode:
            self.logger.info(f"{self.name} in PASSTHROUGH mode. Not generating real FILL for order ID {order_id}.")
            # Passthrough means do nothing beyond logging the order.
            return

        # Simulate fill and publish FILL event
        # For MVP, assume immediate fill at the specified price (no slippage beyond what's implied by fill_price)
        fill_id = self._generate_unique_id(prefix="fill_")
        
        fill_payload = {
            "fill_id": fill_id,
            "order_id": order_id,
            "symbol": symbol,
            "timestamp": datetime.datetime.now(datetime.timezone.utc), # Fill timestamp (can also use order_timestamp for perfect backtest fill)
            "quantity_filled": quantity, # Assuming full fill for simulation
            "fill_price": float(fill_price), # Ensure it's a float
            "commission": self._commission_per_trade, # Use the configured commission
            "direction": direction,
            "exchange": "SIMULATED_EXCHANGE"
        }
        fill_event = Event(EventType.FILL, fill_payload)
        self.logger.debug(f"{self.name} publishing FILL: {quantity} {symbol} at {fill_price}")
        self._event_bus.publish(fill_event)
        self.logger.info(
            f"{self.name} published FILL Event: ID={fill_id} for OrderID={order_id}, "
            f"Filled {quantity} {symbol} at {fill_price:.2f}"
        )

    def start(self):
        if self.state not in [BaseComponent.STATE_INITIALIZED, BaseComponent.STATE_STOPPED]:
            self.logger.warning(f"Cannot start {self.name} from state '{self.state}'. Expected INITIALIZED or STOPPED.")
            return
            
        # Ensure we're subscribed to ORDER events (needed for restarts)
        if self._event_bus:
            self._event_bus.subscribe(EventType.ORDER, self._on_order_event)
            self.logger.debug(f"{self.name} re-subscribed to ORDER events on start/restart")
            
        self.logger.info(f"{self.name} started. Listening for ORDER events...") # Updated log message
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"Stopping {self.name}...")
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.ORDER, self._on_order_event) # Unsubscribe from ORDER
                self.logger.info(f"'{self.name}' attempted to unsubscribe from ORDER events.")
            except Exception as e: # Catch generic exception, though EventBus usually raises ValueError
                self.logger.error(f"Error unsubscribing '{self.name}' from ORDER events: {e}")
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"{self.name} stopped. State: {self.state}")
