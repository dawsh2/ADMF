# src/execution/simulated_execution_handler.py
import logging
import datetime
import uuid # For generating unique IDs

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError

class SimulatedExecutionHandler(BaseComponent):
    """
    Simulates the execution of orders based on SIGNAL events.
    It receives SIGNAL events, creates ORDER events, simulates their fill,
    and then publishes FILL events.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str):
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error("EventBus instance is required.")
            raise ConfigurationError("EventBus instance is required for SimulatedExecutionHandler.")

        self._default_quantity = self.get_specific_config("default_quantity", 100)
        self._commission_per_trade = self.get_specific_config("commission_per_trade", 0.0) # Corrected from just "commission"
        self._passthrough_mode = self.get_specific_config("passthrough", False)

        if not isinstance(self._default_quantity, int) or self._default_quantity <= 0:
            raise ConfigurationError(f"SimExecHandler: 'default_quantity' must be a positive integer. Got {self._default_quantity}")
        if not isinstance(self._commission_per_trade, (int, float)) or self._commission_per_trade < 0:
            raise ConfigurationError(f"SimExecHandler: 'commission_per_trade' must be a non-negative number. Got {self._commission_per_trade}")

        self.logger.info(
            f"{self.name} configured. Default Qty: {self._default_quantity}, "
            f"Commission: {self._commission_per_trade}, Passthrough: {self._passthrough_mode}"
        )

    def setup(self):
        self.logger.info(f"Setting up {self.name}...")
        self._event_bus.subscribe(EventType.SIGNAL, self._on_signal_event)
        self.logger.info(f"'{self.name}' subscribed to SIGNAL events.")
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"{self.name} setup complete. State: {self.state}")

    def _generate_unique_id(self, prefix=""):
        return f"{prefix}{uuid.uuid4().hex[:8]}"

    def _on_signal_event(self, signal_event: Event):
        if signal_event.event_type != EventType.SIGNAL:
            return

        signal_data = signal_event.payload
        symbol = signal_data.get("symbol")
        signal_type = signal_data.get("signal_type") # BUY or SELL
        price_at_signal = signal_data.get("price_at_signal")
        signal_timestamp = signal_data.get("timestamp")
        strategy_id = signal_data.get("strategy_id", "UnknownStrategy")

        self.logger.info(f"{self.name} received SIGNAL: {signal_type} {symbol} at {price_at_signal} from {strategy_id}")

        if self._passthrough_mode:
            self.logger.info(f"{self.name} in PASSTHROUGH mode. Not creating real ORDER/FILL.")
            # Optionally, create dummy/passthrough ORDER and FILL events if needed for downstream testing
            # For now, passthrough means do nothing beyond logging the signal.
            return

        # 1. Create and publish ORDER event
        order_id = self._generate_unique_id(prefix="ord_")
        order_payload = {
            "order_id": order_id,
            "symbol": symbol,
            "order_type": "MARKET", # For MVP, always market order
            "direction": signal_type, # Should be 'BUY' or 'SELL'
            "quantity": self._default_quantity,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "strategy_id": strategy_id
        }
        order_event = Event(EventType.ORDER, order_payload)
        self._event_bus.publish(order_event)
        self.logger.info(f"{self.name} published ORDER Event: ID={order_id}, {signal_type} {self._default_quantity} {symbol}")

        # 2. Simulate fill and publish FILL event
        # For MVP, assume immediate fill at the signal price (no slippage)
        fill_id = self._generate_unique_id(prefix="fill_")
        fill_payload = {
            "fill_id": fill_id,
            "order_id": order_id,
            "symbol": symbol,
            "timestamp": datetime.datetime.now(datetime.timezone.utc), # Fill timestamp
            "quantity_filled": self._default_quantity,
            "fill_price": price_at_signal,
            "commission": self._commission_per_trade,
            "direction": signal_type, # From original signal/order
            "exchange": "SIMULATED"
        }
        fill_event = Event(EventType.FILL, fill_payload)
        self._event_bus.publish(fill_event)
        self.logger.info(
            f"{self.name} published FILL Event: ID={fill_id} for OrderID={order_id}, "
            f"Filled {self._default_quantity} {symbol} at {price_at_signal:.2f}"
        )

    def start(self):
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start {self.name} from state '{self.state}'. Expected INITIALIZED.")
            return
        self.logger.info(f"{self.name} started. Listening for SIGNAL events...")
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"Stopping {self.name}...")
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.SIGNAL, self._on_signal_event)
                self.logger.info(f"'{self.name}' attempted to unsubscribe from SIGNAL events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.name}' from SIGNAL events: {e}")
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"{self.name} stopped. State: {self.state}")
