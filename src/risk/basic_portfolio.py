# src/risk/basic_portfolio.py
import logging
import datetime # For type hinting and operations
from typing import Dict, Any, Optional, List, Tuple # Ensure List and Tuple are imported
import uuid

from ..core.component import BaseComponent
from ..core.event import Event, EventType
from ..core.exceptions import ComponentError, DependencyNotFoundError # Added DependencyNotFoundError
# Assuming RegimeDetector is imported for type hinting if passed directly,
# otherwise, we'll resolve it by key.
# from ..strategy.regime_detector import RegimeDetector 


class BasicPortfolio(BaseComponent):
    """
    Manages the portfolio's positions, cash, and tracks performance.
    This version will be enhanced to be regime-aware.
    """
    def __init__(self, 
                 instance_name: str, 
                 config_loader, 
                 event_bus, 
                 container, # Added container for dependency resolution
                 component_config_key: Optional[str] = None):
        """
        Initialize the BasicPortfolio.

        Args:
            instance_name (str): Name of this portfolio instance.
            config_loader: Configuration loader instance.
            event_bus: Event bus for subscribing to events.
            container: The application's dependency injection container.
            component_config_key (str, optional): Key for specific config.
        """
        super().__init__(instance_name, config_loader, component_config_key)
        self._event_bus = event_bus
        self._container = container # Store the container
        self.initial_cash: float = self.get_specific_config('initial_cash', 100000.0)
        self.current_cash: float = self.initial_cash
        
        # Stores open positions: {symbol: {'quantity': float, 'direction': str, 'entry_price': float, 
        #                                'entry_timestamp': datetime, 'trade_id': str,
        #                                'current_segment_entry_price': float, 
        #                                'current_segment_regime': str,
        #                                'entry_regime': str}}
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        
        # Stores closed trades or trade segments: 
        # [{'symbol': str, 'trade_id': str, 'segment_id': str, 
        #   'entry_timestamp': datetime, 'exit_timestamp': datetime, 
        #   'direction': str ('LONG'/'SHORT'), 'entry_price': float, 'exit_price': float, 
        #   'quantity': float, 'commission': float, 'pnl': float, 'regime': str}]
        self._trade_log: List[Dict[str, Any]] = []
        
        self.total_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.current_holdings_value: float = 0.0
        self.current_total_value: float = self.initial_cash
        self._last_bar_prices: Dict[str, float] = {} # Stores last close price for unrealized PnL

        self._regime_detector: Optional[Any] = None # Will hold RegimeDetector instance
        self.regime_detector_key: str = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        self._current_market_regime: Optional[str] = "default" # Store the latest global market regime

        self.logger.info(f"BasicPortfolio '{self.name}' initialized with initial cash: {self.initial_cash:.2f}")

    def setup(self):
        """
        Subscribe to necessary events and resolve dependencies.
        """
        super().setup() # Calls parent's setup
        self.logger.info(f"Setting up BasicPortfolio '{self.name}'...")
        if self._event_bus:
            self._event_bus.subscribe(EventType.FILL, self.on_fill)
            self._event_bus.subscribe(EventType.BAR, self.on_bar)
            # --- ADD SUBSCRIPTION TO CLASSIFICATION EVENT ---
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.name}' subscribed to FILL, BAR, and CLASSIFICATION events.")
        else:
            self.logger.error(f"Event bus not available for '{self.name}'. Cannot subscribe to events.")
            self.state = BaseComponent.STATE_FAILED
            return

        # Resolve RegimeDetector
        try:
            self._regime_detector = self._container.resolve(self.regime_detector_key)
            if self._regime_detector:
                self.logger.info(f"Successfully resolved and linked RegimeDetector: {self._regime_detector.name}")
                # Get initial regime if detector is ready
                if hasattr(self._regime_detector, 'get_current_classification') and callable(getattr(self._regime_detector, 'get_current_classification')):
                    initial_regime = self._regime_detector.get_current_classification()
                    if initial_regime:
                        self._current_market_regime = initial_regime
                        self.logger.info(f"Initial market regime set to: {self._current_market_regime}")
                    else:
                        self.logger.warning(f"RegimeDetector '{self._regime_detector.name}' returned no initial classification. Defaulting to 'default'.")
                        self._current_market_regime = "default"
            else: # Should not happen if resolve doesn't raise DependencyNotFoundError
                self.logger.error(f"Failed to resolve RegimeDetector with key '{self.regime_detector_key}'. Regime-aware tracking will be disabled.")
        except DependencyNotFoundError:
            self.logger.error(f"Dependency '{self.regime_detector_key}' (RegimeDetector) not found in container. Regime-aware tracking will be disabled.")
            # self._regime_detector will remain None. Subsequent checks for its existence will handle this.
        except Exception as e:
            self.logger.error(f"Error resolving RegimeDetector '{self.regime_detector_key}': {e}", exc_info=True)
            # self._regime_detector will remain None.

        self._update_portfolio_value(datetime.datetime.now(datetime.timezone.utc)) # Initial update
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"BasicPortfolio '{self.name}' setup complete. State: {self.state}")


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
