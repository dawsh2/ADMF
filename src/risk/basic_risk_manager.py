# src/risk/basic_risk_manager.py
import logging
import datetime
import uuid
from typing import Optional, Dict, Any

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError, DependencyNotFoundError # Added DependencyNotFoundError
from src.risk.basic_portfolio import BasicPortfolio # For type hinting
from src.core.container import Container # For type hinting and resolving

class BasicRiskManager(BaseComponent):
    """
    A very basic risk manager that translates strategy signals (-1, 0, 1)
    into orders to achieve a target position (e.g., fixed quantity long or short).
    It considers the current position to determine the required trade.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str,
                 container: Container, portfolio_manager_key: str = "portfolio_manager"): # Added container and portfolio_manager_key
        super().__init__(instance_name, config_loader, component_config_key)
        
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error(f"EventBus instance is required for {self.name}.")
            raise ConfigurationError(f"EventBus instance is required for {self.name}.")

        self._container = container # Store container
        self._portfolio_manager_key = portfolio_manager_key # Store key/name of portfolio service
        self._portfolio_manager: Optional[BasicPortfolio] = None # Will be resolved in setup

        self._target_trade_quantity: int = self.get_specific_config("target_trade_quantity", 100)
        if not isinstance(self._target_trade_quantity, int) or self._target_trade_quantity <= 0:
            raise ConfigurationError(
                f"'{self.name}': 'target_trade_quantity' must be a positive integer. "
                f"Got: {self._target_trade_quantity}"
            )

        self.logger.info(
            f"{self.name} initialized. Target trade quantity: {self._target_trade_quantity}. "
            f"Portfolio manager key: '{self._portfolio_manager_key}'."
        )

    def setup(self):
        self.logger.info(f"Setting up {self.name}...")
        # Resolve portfolio manager dependency using the stored container and key
        try:
            resolved_pm = self._container.resolve(self._portfolio_manager_key)
            if not isinstance(resolved_pm, BasicPortfolio): # Ensure it's the correct type
                self.logger.error(f"{self.name} resolved '{self._portfolio_manager_key}' but it is not a BasicPortfolio instance. Type: {type(resolved_pm)}")
                self.state = BaseComponent.STATE_FAILED
                raise ComponentError(f"{self.name} resolved incorrect type for portfolio_manager.")
            self._portfolio_manager = resolved_pm # Assign if correct type
            self.logger.info(f"Successfully resolved and linked portfolio_manager: {self._portfolio_manager.name}")

        except DependencyNotFoundError as e:
            self.logger.error(f"Critical: {self.name} could not resolve portfolio_manager dependency '{self._portfolio_manager_key}'. Error: {e}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            raise ComponentError(f"{self.name} failed setup due to missing portfolio_manager dependency.") from e
        except Exception as e: # Catch any other unexpected errors during resolution or type check
            self.logger.error(f"Critical: Unexpected error resolving portfolio_manager for {self.name}. Error: {e}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            raise ComponentError(f"{self.name} failed setup due to an unexpected error with portfolio_manager dependency.") from e

        self._event_bus.subscribe(EventType.SIGNAL, self._on_signal_event)
        self.logger.info(f"'{self.name}' subscribed to SIGNAL events.")
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"{self.name} setup complete. State: {self.state}")

    # ... (keep _generate_unique_id, _on_signal_event, start, stop methods as defined before) ...
    def _generate_unique_id(self, prefix="ord_rm_"): # RiskManager order prefix
        return f"{prefix}{uuid.uuid4().hex[:10]}"

    def _on_signal_event(self, signal_event: Event):
        if signal_event.event_type != EventType.SIGNAL:
            return

        # Ensure portfolio manager is available (should have been resolved in setup)
        if not self._portfolio_manager:
            self.logger.error(f"'{self.name}' cannot process signal: PortfolioManager not available.")
            return

        signal_data: Dict[str, Any] = signal_event.payload
        symbol = signal_data.get("symbol")
        signal_type_int = signal_data.get("signal_type") # Expected to be -1, (0), or 1
        price_at_signal = signal_data.get("price_at_signal")
        # signal_timestamp = signal_data.get("timestamp", datetime.datetime.now(datetime.timezone.utc)) # Already there
        strategy_id = signal_data.get("strategy_id", "UnknownStrategy")

        if not all([symbol, signal_type_int is not None, price_at_signal is not None]):
            self.logger.error(f"'{self.name}' received incomplete SIGNAL data: {signal_data}")
            return
        
        self.logger.info(
            f"'{self.name}' received SIGNAL: Type={signal_type_int} for {symbol} at {price_at_signal} "
            f"from {strategy_id}."
        )

        current_quantity = self._portfolio_manager.get_current_position_quantity(symbol)
        self.logger.info(f"Current position for {symbol}: {current_quantity}")

        target_position = 0
        if signal_type_int == 1: # Buy signal -> Aim for long position
            target_position = self._target_trade_quantity
        elif signal_type_int == -1: # Sell signal -> Aim for short position
            target_position = -self._target_trade_quantity
        elif signal_type_int == 0: # Neutral/Flat signal
            target_position = 0
        else:
            self.logger.warning(f"'{self.name}': Unknown signal type {signal_type_int} for {symbol}. No action taken.")
            return

        self.logger.info(f"Target position for {symbol} based on signal {signal_type_int}: {target_position}")

        quantity_to_order = target_position - current_quantity
        
        if quantity_to_order == 0:
            self.logger.info(
                f"'{self.name}': No order needed for {symbol}. Current position ({current_quantity}) "
                f"already matches target ({target_position})."
            )
            return

        order_direction = "BUY" if quantity_to_order > 0 else "SELL"
        order_quantity_abs = abs(quantity_to_order)

        order_id = self._generate_unique_id()
        order_payload = {
            "order_id": order_id,
            "symbol": symbol,
            "order_type": "MARKET", 
            "direction": order_direction,
            "quantity": order_quantity_abs,
            "simulated_fill_price": price_at_signal, 
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "strategy_id": strategy_id, 
            "risk_manager_id": self.name
        }
        
        order_event = Event(EventType.ORDER, order_payload)
        self._event_bus.publish(order_event)
        self.logger.info(
            f"'{self.name}' published ORDER: ID={order_id}, {order_direction} {order_quantity_abs} {symbol} "
            f"at (intended) {price_at_signal:.2f} to reach target pos {target_position} "
            f"(current: {current_quantity})."
        )

    def start(self):
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start {self.name} from state '{self.state}'. Expected INITIALIZED.")
            return
        if not self._portfolio_manager: # Double check portfolio manager linkage before starting fully
            self.logger.error(f"Cannot start {self.name}: PortfolioManager not linked. Setup may have failed.")
            self.state = BaseComponent.STATE_FAILED
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
