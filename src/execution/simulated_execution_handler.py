# src/execution/simulated_execution_handler.py
import logging
import datetime
import uuid # For generating unique IDs

from typing import Optional

from src.core.component_base import ComponentBase
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError

class SimulatedExecutionHandler(ComponentBase):
    """
    Simulates the execution of orders based on ORDER events.
    It receives ORDER events, simulates their fill based on data within the order
    (like a simulated_fill_price for market orders), and then publishes FILL events.
    """

    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Initialize internal state (no external dependencies)
        self._default_quantity: int = 100
        self._commission_per_trade: float = 0.0
        self._passthrough_mode: bool = False

    def _initialize(self):
        """Component-specific initialization logic."""
        # Load configuration
        self._default_quantity = self.get_specific_config("default_quantity", 100)
        self._commission_per_trade = self.get_specific_config("commission_per_trade", 0.0)
        self._passthrough_mode = self.get_specific_config("passthrough", False)
        
        if not isinstance(self._default_quantity, int) or self._default_quantity <= 0:
            self.logger.warning(f"SimExecHandler: 'default_quantity' ({self._default_quantity}) might not be used if orders always specify quantity.")
        if not isinstance(self._commission_per_trade, (int, float)) or self._commission_per_trade < 0:
            raise ConfigurationError(f"SimExecHandler: 'commission_per_trade' must be a non-negative number. Got {self._commission_per_trade}")
        
        self.logger.info(
            f"{self.instance_name} configured. Default Qty (if order lacks it): {self._default_quantity}, "
            f"Commission: {self._commission_per_trade}, Passthrough: {self._passthrough_mode}"
        )
    
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
    
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        if self.subscription_manager:
            self.subscription_manager.subscribe(EventType.ORDER, self._on_order_event)
            self.logger.info(f"'{self.instance_name}' subscribed to ORDER events.")
    
    def setup(self):
        """Set up the execution handler."""
        self.logger.info(f"Setting up {self.instance_name}...")
        # Event subscriptions handled by initialize_event_subscriptions
        self.logger.info(f"{self.instance_name} setup complete.")

    def _generate_unique_id(self, prefix=""):
        return f"{prefix}{uuid.uuid4().hex[:8]}"

    def _on_order_event(self, order_event: Event): # Renamed from _on_signal_event
        self.logger.debug(f"{self.instance_name} received ORDER event")
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
            f"{self.instance_name} received ORDER: ID={order_id}, {direction} {quantity} {symbol} "
            f"Type={order_type} at (target/simulated price) {fill_price if fill_price is not None else 'N/A'} "
            f"from {strategy_id}"
        )

        if not all([symbol, direction, quantity is not None and quantity > 0]):
            self.logger.error(f"{self.instance_name} received invalid or incomplete ORDER data: {order_data}")
            return

        if fill_price is None and order_type == "MARKET":
            self.logger.error(
                f"{self.instance_name}: Market order ID={order_id} for {symbol} has no fill price indication "
                f"('simulated_fill_price' or 'price'). Cannot simulate fill. Order: {order_data}"
            )
            # Optionally, publish a REJECTED_ORDER event or similar, or just log and return.
            return
        elif fill_price is None and order_type == "LIMIT":
             self.logger.error(
                f"{self.instance_name}: Limit order ID={order_id} for {symbol} has no price_limit. "
                f"Cannot simulate fill. Order: {order_data}"
            )
             return


        if self._passthrough_mode:
            self.logger.info(f"{self.instance_name} in PASSTHROUGH mode. Not generating real FILL for order ID {order_id}.")
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
        self.logger.debug(f"{self.instance_name} publishing FILL: {quantity} {symbol} at {fill_price}")
        self.event_bus.publish(fill_event)
        self.logger.info(
            f"{self.instance_name} published FILL Event: ID={fill_id} for OrderID={order_id}, "
            f"Filled {quantity} {symbol} at {fill_price:.2f}"
        )

    def start(self):
        """Start the execution handler."""
        super().start()
        self.logger.info(f"{self.instance_name} started. Listening for ORDER events...")

    def stop(self):
        """Stop the execution handler."""
        self.logger.info(f"Stopping {self.instance_name}...")
        super().stop()
        self.logger.info(f"{self.instance_name} stopped.")
    
    def teardown(self):
        """Clean up resources."""
        super().teardown()
