# src/risk/basic_risk_manager.py
import logging
import datetime
import uuid
from typing import Optional, Dict, Any

from src.core.component_base import ComponentBase
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError, ComponentError, DependencyNotFoundError
from src.risk.basic_portfolio import BasicPortfolio  # For type hinting

class BasicRiskManager(ComponentBase):
    """
    A very basic risk manager that translates strategy signals (-1, 0, 1)
    into orders to achieve a target position (e.g., fixed quantity long or short).
    It considers the current position to determine the required trade.
    """

    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Initialize internal state (no external dependencies)
        self._portfolio_manager_key: str = "portfolio_manager"
        self._portfolio_manager: Optional[BasicPortfolio] = None
        self._target_trade_quantity: int = 100
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Load configuration
        self._target_trade_quantity = self.get_specific_config("target_trade_quantity", 100)
        if not isinstance(self._target_trade_quantity, int) or self._target_trade_quantity <= 0:
            raise ConfigurationError(
                f"'{self.instance_name}': 'target_trade_quantity' must be a positive integer. "
                f"Got: {self._target_trade_quantity}"
            )
        
        self._portfolio_manager_key = self.get_specific_config("portfolio_manager_key", "portfolio_manager")
        
        # Resolve portfolio manager dependency using the container
        try:
            resolved_pm = self.container.resolve(self._portfolio_manager_key)
            if not isinstance(resolved_pm, BasicPortfolio):
                self.logger.error(f"{self.instance_name} resolved '{self._portfolio_manager_key}' but it is not a BasicPortfolio instance. Type: {type(resolved_pm)}")
                raise ComponentError(f"{self.instance_name} resolved incorrect type for portfolio_manager.")
            self._portfolio_manager = resolved_pm
            self.logger.info(f"Successfully resolved and linked portfolio_manager: {self._portfolio_manager.instance_name}")

        except DependencyNotFoundError as e:
            self.logger.error(f"Critical: {self.instance_name} could not resolve portfolio_manager dependency '{self._portfolio_manager_key}'. Error: {e}", exc_info=True)
            raise ComponentError(f"{self.instance_name} failed initialization due to missing portfolio_manager dependency.") from e
        except Exception as e:
            self.logger.error(f"Critical: Unexpected error resolving portfolio_manager for {self.instance_name}. Error: {e}", exc_info=True)
            raise ComponentError(f"{self.instance_name} failed initialization due to an unexpected error with portfolio_manager dependency.") from e
        
        self.logger.info(
            f"{self.instance_name} initialized. Target trade quantity: {self._target_trade_quantity}. "
            f"Portfolio manager key: '{self._portfolio_manager_key}'."
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
            self.subscription_manager.subscribe(EventType.SIGNAL, self._on_signal_event)
            self.logger.info(f"'{self.instance_name}' subscribed to SIGNAL events.")

    def setup(self):
        self.logger.info(f"Setting up {self.instance_name}...")
        # Resolve portfolio manager dependency using the stored container and key
        try:
            resolved_pm = self.container.resolve(self._portfolio_manager_key)
            if not isinstance(resolved_pm, BasicPortfolio): # Ensure it's the correct type
                self.logger.error(f"{self.instance_name} resolved '{self._portfolio_manager_key}' but it is not a BasicPortfolio instance. Type: {type(resolved_pm)}")
                # Mark as failed
                raise ComponentError(f"{self.instance_name} resolved incorrect type for portfolio_manager.")
            self._portfolio_manager = resolved_pm # Assign if correct type
            self.logger.info(f"Successfully resolved and linked portfolio_manager: {self._portfolio_manager.instance_name}")

        except DependencyNotFoundError as e:
            self.logger.error(f"Critical: {self.instance_name} could not resolve portfolio_manager dependency '{self._portfolio_manager_key}'. Error: {e}", exc_info=True)
            # Mark as failed
            raise ComponentError(f"{self.instance_name} failed setup due to missing portfolio_manager dependency.") from e
        except Exception as e: # Catch any other unexpected errors during resolution or type check
            self.logger.error(f"Critical: Unexpected error resolving portfolio_manager for {self.instance_name}. Error: {e}", exc_info=True)
            # Mark as failed
            raise ComponentError(f"{self.instance_name} failed setup due to an unexpected error with portfolio_manager dependency.") from e

        # Event subscriptions handled by initialize_event_subscriptions
        self.logger.info(f"{self.instance_name} setup complete.")

    # ... (keep _generate_unique_id, _on_signal_event, start, stop methods as defined before) ...
    def _generate_unique_id(self, prefix="ord_rm_"): # RiskManager order prefix
        return f"{prefix}{uuid.uuid4().hex[:10]}"

    def _on_signal_event(self, signal_event: Event):
        self.logger.debug(f"{self.instance_name} received SIGNAL event")
        if signal_event.event_type != EventType.SIGNAL:
            return

        # Ensure portfolio manager is available (should have been resolved in setup)
        if not self._portfolio_manager:
            self.logger.error(f"'{self.instance_name}' cannot process signal: PortfolioManager not available.")
            return

        signal_data: Dict[str, Any] = signal_event.payload
        symbol = signal_data.get("symbol")
        signal_type_int = signal_data.get("signal_type") # Expected to be -1, (0), or 1
        price_at_signal = signal_data.get("price_at_signal")
        # signal_timestamp = signal_data.get("timestamp", datetime.datetime.now(datetime.timezone.utc)) # Already there
        strategy_id = signal_data.get("strategy_id", "UnknownStrategy")


        if not all([symbol, signal_type_int is not None, price_at_signal is not None]):
            self.logger.error(f"'{self.instance_name}' received incomplete SIGNAL data: {signal_data}")
            return
        
        self.logger.info(
            f"'{self.instance_name}' received SIGNAL: Type={signal_type_int} for {symbol} at {price_at_signal} "
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
            self.logger.warning(f"'{self.instance_name}': Unknown signal type {signal_type_int} for {symbol}. No action taken.")
            return

        self.logger.info(f"Target position for {symbol} based on signal {signal_type_int}: {target_position}")

        quantity_to_order = target_position - current_quantity
        
        if quantity_to_order == 0:
            self.logger.info(
                f"'{self.instance_name}': No order needed for {symbol}. Current position ({current_quantity}) "
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
            "risk_manager_id": self.instance_name
        }
        
        order_event = Event(EventType.ORDER, order_payload)
        self.logger.debug(f"{self.instance_name} publishing ORDER: {order_direction} {order_quantity_abs} {symbol}")
        self.event_bus.publish(order_event)
        self.logger.info(
            f"'{self.instance_name}' published ORDER: ID={order_id}, {order_direction} {order_quantity_abs} {symbol} "
            f"at (intended) {price_at_signal:.2f} to reach target pos {target_position} "
            f"(current: {current_quantity})."
        )

    def start(self):
        """Start the risk manager."""
        super().start()
        if not self._portfolio_manager: # Double check portfolio manager linkage before starting fully
            self.logger.error(f"Cannot start {self.instance_name}: PortfolioManager not linked. Setup may have failed.")
            # Mark as failed
            return
            
        self.logger.info(f"{self.instance_name} started. Listening for SIGNAL events...")

    def stop(self):
        """Stop the risk manager."""
        self.logger.info(f"Stopping {self.instance_name}...")
        super().stop()
        self.logger.info(f"{self.instance_name} stopped.")
    
    def teardown(self):
        """Clean up resources."""
        super().teardown()
        self._portfolio_manager = None
