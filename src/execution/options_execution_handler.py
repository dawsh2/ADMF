#!/usr/bin/env python3
"""
Options Execution Handler - Converts stock signals to 1DTE options trades.

This handler receives stock trading signals and converts them to options orders:
- BUY signal → Buy ATM Call options
- SELL signal → Buy ATM Put options
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import math

from ..core.component_base import ComponentBase
from ..core.event_bus import Event, EventType
from ..core.subscription_manager import SubscriptionManager
from .brokers.alpaca_broker import AlpacaBroker
from .brokers.base_broker import OrderSide, OrderType


class OptionsExecutionHandler(ComponentBase):
    """
    Options execution handler that converts stock signals to options trades.
    
    Specifically designed for 1DTE (1 Day To Expiration) SPY options trading.
    """
    
    def __init__(self, instance_name="options_execution_handler"):
        """Initialize with instance name."""
        super().__init__(instance_name)
        self._broker_type = None
        self._broker = None
        self._paper_trading = True
        self._pending_orders = {}
        self._subscription_manager = SubscriptionManager()
        self._order_check_interval = 1.0
        
        # Options-specific settings
        self._underlying_symbol = None  # e.g., "SPY"
        self._days_to_expiration = 1    # 1DTE options
        self._option_quantity = 1       # Number of contracts per signal
        self._strike_selection = "atm"  # "atm", "itm1", "otm1" etc.
        
        # Cache for current market data
        self._last_underlying_price = None
        self._available_strikes = {}
        
        # Position tracking
        self._current_position = None  # 'call', 'put', or None
        self._current_option_symbol = None
        self._exit_current_before_new = True  # Exit existing position before entering opposite
        
    def initialize(self, context: Dict[str, Any]):
        """Initialize from context."""
        super().initialize(context)
        
        # Get configuration
        config = self._config_loader.get(f'components.{self.instance_name}.config', {})
        self._broker_type = config.get('broker_type', 'alpaca')
        
        # Options-specific config
        self._underlying_symbol = config.get('underlying_symbol', 'SPY')
        self._days_to_expiration = config.get('days_to_expiration', 1)
        self._option_quantity = config.get('option_quantity', 1)
        self._strike_selection = config.get('strike_selection', 'atm')
        
        # Check if paper trading is enabled
        cli_args = context.get('metadata', {}).get('cli_args', {})
        self._paper_trading = cli_args.get('paper', True)
        
        self.logger.info(f"OptionsExecutionHandler initialized")
        self.logger.info(f"Trading {self._days_to_expiration}DTE {self._underlying_symbol} options")
        self.logger.info(f"Strike selection: {self._strike_selection}, Quantity: {self._option_quantity}")
        self.logger.info(f"Paper trading: {self._paper_trading}")
        
    def initialize_event_subscriptions(self):
        """Subscribe to ORDER events."""
        self._subscription_manager.subscribe(
            event_type=EventType.ORDER,
            handler=self._on_order_event,
            context=self
        )
        
        # Also subscribe to BAR events to track underlying price
        self._subscription_manager.subscribe(
            event_type=EventType.BAR,
            handler=self._on_bar_event,
            context=self
        )
        
    def start(self):
        """Start order execution."""
        super().start()
        
        # Create broker instance
        if self._broker_type == 'alpaca':
            self._broker = AlpacaBroker(paper_trading=self._paper_trading)
        else:
            raise ValueError(f"Unsupported broker type: {self._broker_type}")
            
        # Get credentials
        credentials = self._get_broker_credentials()
        
        # Connect to broker
        if self._broker.connect(credentials):
            self.logger.info(f"Connected to {self._broker_type} broker")
            if self._paper_trading:
                self.logger.info("Using paper trading account")
            else:
                self.logger.warning("Using LIVE trading account - be careful!")
                
            # Start order monitoring thread
            self._start_order_monitoring()
            
            # Get initial underlying price
            self._update_underlying_price()
            
            # Reconcile positions on startup
            self._reconcile_positions()
        else:
            raise RuntimeError(f"Failed to connect to {self._broker_type}")
            
    def stop(self):
        """Stop order execution."""
        if self._broker:
            # Cancel all pending orders
            for order_id in list(self._pending_orders.keys()):
                self._broker.cancel_order(order_id)
            
            # Disconnect from broker
            self._broker.disconnect()
            
        self.logger.info("Disconnected from broker API")
        super().stop()
        
    def _on_bar_event(self, event: Event):
        """Track underlying price from BAR events."""
        bar = event.payload
        if bar['symbol'] == self._underlying_symbol:
            self._last_underlying_price = bar['close']
            
    def _on_order_event(self, event: Event):
        """
        Handle incoming ORDER events and convert to options orders.
        
        Args:
            event: ORDER event from risk manager
        """
        order = event.payload
        
        # Only process orders for our underlying symbol
        if order['symbol'] != self._underlying_symbol:
            self.logger.warning(f"Ignoring order for {order['symbol']}, only trading {self._underlying_symbol} options")
            return
            
        self.logger.info(f"Received ORDER: {order['side']} signal for {order['symbol']}")
        
        # Determine new position type
        new_position_type = 'call' if order['side'] == 'buy' else 'put'
        
        # Check if we need to exit current position first
        if self._current_position and self._current_position != new_position_type:
            self.logger.info(f"Signal flip detected: {self._current_position} → {new_position_type}")
            
            if self._exit_current_before_new:
                # Exit current position before entering new one
                self._exit_current_position()
                
        # If we're already in the same position type, skip (no pyramiding)
        if self._current_position == new_position_type:
            self.logger.info(f"Already holding {new_position_type} position, skipping new order")
            return
            
        # Convert stock signal to options order
        try:
            option_symbol = self._select_option_contract(order['side'])
            
            if not option_symbol:
                self.logger.error("Could not find suitable option contract")
                return
                
            # Submit options order
            # For options, we always BUY (calls for bullish, puts for bearish)
            broker_order_id = self._broker.submit_order(
                symbol=option_symbol,
                side=OrderSide.BUY,
                quantity=self._option_quantity,
                order_type=OrderType.MARKET,
                time_in_force='day'
            )
            
            if broker_order_id:
                # Store order for monitoring
                self._pending_orders[broker_order_id] = {
                    **order,
                    'option_symbol': option_symbol,
                    'option_quantity': self._option_quantity,
                    'position_type': new_position_type
                }
                self.logger.info(f"Options order submitted: BUY {self._option_quantity} {option_symbol}")
            else:
                self.logger.error(f"Failed to submit options order")
                
        except Exception as e:
            self.logger.error(f"Error processing options order: {e}")
            
    def _select_option_contract(self, signal_side: str) -> Optional[str]:
        """
        Select the appropriate option contract based on signal.
        
        Args:
            signal_side: 'buy' for calls, 'sell' for puts
            
        Returns:
            Option symbol (OCC format) or None if not found
        """
        if not self._last_underlying_price:
            self._update_underlying_price()
            
        if not self._last_underlying_price:
            self.logger.error("Cannot get underlying price")
            return None
            
        # Determine option type based on signal
        option_type = 'C' if signal_side == 'buy' else 'P'
        
        # Calculate target expiration date (1DTE)
        today = datetime.now().date()
        target_expiry = today + timedelta(days=self._days_to_expiration)
        
        # Find the next valid trading day (skip weekends)
        while target_expiry.weekday() >= 5:  # Saturday = 5, Sunday = 6
            target_expiry += timedelta(days=1)
            
        # Get available strikes for this expiration
        strikes = self._get_available_strikes(target_expiry)
        
        if not strikes:
            self.logger.error(f"No strikes available for {target_expiry}")
            return None
            
        # Select ATM strike (closest to current price)
        atm_strike = self._find_atm_strike(strikes, self._last_underlying_price)
        
        # Adjust strike based on selection strategy
        selected_strike = self._adjust_strike_selection(strikes, atm_strike, option_type)
        
        # Format option symbol in OCC format
        # Example: SPY231229C450 (SPY, Dec 29 2023, Call, $450 strike)
        option_symbol = f"{self._underlying_symbol}{target_expiry.strftime('%y%m%d')}{option_type}{int(selected_strike * 1000):08d}"
        
        self.logger.info(f"Selected option: {option_symbol} (underlying @ ${self._last_underlying_price:.2f})")
        
        return option_symbol
        
    def _update_underlying_price(self):
        """Update the current price of the underlying."""
        try:
            quote = self._broker.get_latest_quote(self._underlying_symbol)
            if quote:
                self._last_underlying_price = quote['last']
                self.logger.info(f"{self._underlying_symbol} current price: ${self._last_underlying_price:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to get underlying price: {e}")
            
    def _get_available_strikes(self, expiry_date: datetime.date) -> List[float]:
        """
        Get available option strikes for a given expiration date.
        
        Note: This is a simplified implementation. In production, you'd
        query the broker's options chain API.
        """
        if not self._last_underlying_price:
            return []
            
        # For SPY, strikes are typically $1 apart near ATM
        # This is a simplified approximation
        center = round(self._last_underlying_price)
        strikes = []
        
        # Generate strikes around current price
        for offset in range(-10, 11):  # +/- $10 from current price
            strike = center + offset
            if strike > 0:
                strikes.append(float(strike))
                
        return sorted(strikes)
        
    def _find_atm_strike(self, strikes: List[float], underlying_price: float) -> float:
        """Find the at-the-money (closest) strike."""
        if not strikes:
            return underlying_price
            
        # Find strike with minimum distance to current price
        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        return atm_strike
        
    def _adjust_strike_selection(self, strikes: List[float], atm_strike: float, option_type: str) -> float:
        """
        Adjust strike selection based on strategy.
        
        Args:
            strikes: Available strikes
            atm_strike: The ATM strike
            option_type: 'C' for calls, 'P' for puts
            
        Returns:
            Selected strike price
        """
        if self._strike_selection == "atm":
            return atm_strike
            
        # Find index of ATM strike
        try:
            atm_index = strikes.index(atm_strike)
        except ValueError:
            return atm_strike
            
        # For "closest to ITM" selection
        if self._strike_selection == "itm1":
            # Calls: one strike below ATM (lower strike = ITM for calls)
            # Puts: one strike above ATM (higher strike = ITM for puts)
            if option_type == 'C' and atm_index > 0:
                return strikes[atm_index - 1]
            elif option_type == 'P' and atm_index < len(strikes) - 1:
                return strikes[atm_index + 1]
                
        return atm_strike
        
    def _exit_current_position(self):
        """Exit the current options position by selling it."""
        if not self._current_option_symbol:
            return
            
        self.logger.info(f"Exiting current {self._current_position} position: {self._current_option_symbol}")
        
        try:
            # Submit SELL order to close position
            broker_order_id = self._broker.submit_order(
                symbol=self._current_option_symbol,
                side=OrderSide.SELL,
                quantity=self._option_quantity,
                order_type=OrderType.MARKET,
                time_in_force='day'
            )
            
            if broker_order_id:
                # Store exit order for monitoring
                self._pending_orders[broker_order_id] = {
                    'symbol': self._underlying_symbol,
                    'side': 'sell',  # Closing position
                    'option_symbol': self._current_option_symbol,
                    'option_quantity': self._option_quantity,
                    'position_type': 'exit',
                    'exit_from': self._current_position
                }
                self.logger.info(f"Exit order submitted: SELL {self._option_quantity} {self._current_option_symbol}")
                
                # Clear current position (will be confirmed when fill comes in)
                # For now, assume it will fill and clear to allow new position
                self._current_position = None
                self._current_option_symbol = None
            else:
                self.logger.error(f"Failed to submit exit order for {self._current_option_symbol}")
                
        except Exception as e:
            self.logger.error(f"Error exiting position: {e}")
        
    def _get_broker_credentials(self) -> Dict[str, str]:
        """Get broker API credentials from environment or config."""
        import os
        
        # First try environment variables
        if self._broker_type == 'alpaca':
            api_key = os.environ.get('ALPACA_API_KEY')
            secret_key = os.environ.get('ALPACA_SECRET_KEY')
            
            if api_key and secret_key:
                return {
                    'api_key': api_key,
                    'secret_key': secret_key
                }
                
        # Try config file
        broker_config = self._config_loader.get('broker', {})
        if 'credentials' in broker_config:
            return broker_config['credentials']
            
        raise ValueError(f"No credentials found for {self._broker_type}. "
                        "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
                        
    def _start_order_monitoring(self):
        """Start monitoring pending orders for fills."""
        import threading
        
        def monitor_orders():
            while self.running:
                # Check each pending order
                for order_id, original_order in list(self._pending_orders.items()):
                    status = self._broker.get_order_status(order_id)
                    
                    if status:
                        if status['status'].value == 'filled':
                            # Create FILL event for the option trade
                            self._publish_fill_event(original_order, status)
                            
                            # Update current position tracking
                            if original_order.get('position_type') == 'exit':
                                # Confirmed exit
                                self.logger.info(f"Position exit confirmed for {original_order['option_symbol']}")
                            else:
                                # New position opened
                                self._current_position = original_order.get('position_type')
                                self._current_option_symbol = original_order.get('option_symbol')
                                self.logger.info(f"New {self._current_position} position: {self._current_option_symbol}")
                            
                            # Remove from pending
                            del self._pending_orders[order_id]
                        elif status['status'].value in ['cancelled', 'rejected']:
                            self.logger.warning(f"Options order {order_id} was {status['status'].value}")
                            
                            # If this was an exit that failed, restore position tracking
                            if original_order.get('position_type') == 'exit':
                                self._current_position = original_order.get('exit_from')
                                self.logger.warning(f"Exit failed, still holding {self._current_position} position")
                                
                            del self._pending_orders[order_id]
                            
                # Sleep before next check
                import time
                time.sleep(self._order_check_interval)
                
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_orders, daemon=True)
        monitor_thread.start()
        
    def _reconcile_positions(self):
        """Reconcile positions with broker on startup."""
        positions = self._broker.get_positions()
        if positions:
            self.logger.info(f"Current broker positions: {len(positions)}")
            for pos in positions:
                # Check if it's an option position
                if len(pos['symbol']) > 10:  # Options symbols are longer
                    self.logger.info(f"  Option: {pos['symbol']}: {pos['quantity']} contracts @ ${pos['avg_price']}")
                    
                    # Check if this is one of our SPY options
                    if pos['symbol'].startswith(self._underlying_symbol):
                        # Parse option type from symbol (C or P)
                        if 'C' in pos['symbol'][len(self._underlying_symbol)+6:]:
                            self._current_position = 'call'
                        elif 'P' in pos['symbol'][len(self._underlying_symbol)+6:]:
                            self._current_position = 'put'
                            
                        self._current_option_symbol = pos['symbol']
                        self.logger.warning(f"Found existing {self._current_position} position: {self._current_option_symbol}")
                else:
                    self.logger.info(f"  Stock: {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']}")
        else:
            self.logger.info("No open positions at broker")
            self._current_position = None
            self._current_option_symbol = None
            
    def _publish_fill_event(self, original_order: Dict[str, Any], broker_status: Dict[str, Any]):
        """Publish FILL event from broker order status."""
        # For options, we need to translate back to the underlying symbol
        # so the portfolio manager tracks it correctly
        fill_event = Event(
            event_type=EventType.FILL,
            payload={
                'timestamp': broker_status.get('filled_at', datetime.now()),
                'symbol': original_order['symbol'],  # Original underlying symbol
                'side': original_order['side'],  # Original signal side
                'quantity': original_order['quantity'],  # Original quantity (not used for options)
                'price': broker_status.get('filled_avg_price', 0),  # Option premium paid
                'commission': 0.0,  # TODO: Calculate from broker
                'order_id': original_order.get('order_id'),
                # Additional options-specific data
                'option_symbol': original_order.get('option_symbol'),
                'option_quantity': original_order.get('option_quantity'),
                'option_type': 'call' if original_order['side'] == 'buy' else 'put'
            }
        )
        
        self._event_bus.publish(fill_event)
        self.logger.info(f"Options order filled: {original_order['option_symbol']} "
                        f"{original_order['option_quantity']} contracts @ ${broker_status['filled_avg_price']}")